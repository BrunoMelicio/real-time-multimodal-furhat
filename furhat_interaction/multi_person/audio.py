"""One microphone, Silero, two-person Light-ASD and isolated cpp ASR."""
from furhat_interaction.paths import ROOT
from collections import deque
from queue import Queue,Empty,Full
from time import perf_counter
import numpy as np
from furhat_interaction.dialogue import Utterance
from furhat_interaction.reflection import BargeEvidence
from .policies import speaker_choice,freeze_speaker


class AudioInbox:
    """Bounded callback queue; never block the mic or report a fatal overflow."""
    def __init__(self,size=128):
        from threading import Lock
        self.queue=Queue(size);self.lock=Lock();self.dropped=0

    def put(self,packet):
        try: self.queue.put_nowait(packet)
        except Full:
            try: self.queue.get_nowait()
            except Empty: pass
            with self.lock: self.dropped+=1
            try: self.queue.put_nowait(packet)
            except Full:
                with self.lock: self.dropped+=1

    def get(self,timeout=.05):
        packet=self.queue.get(timeout=timeout)
        with self.lock:
            lost=self.dropped;self.dropped=0
        if lost:
            # Once samples are lost, keep the newest packet, not seconds of
            # stale sound. The caller resets VAD/utterance/identity evidence.
            while True:
                try: packet=self.queue.get_nowait();lost+=1
                except Empty: break
        return packet,lost


def monitoring(mode,end,after):
    return mode in (2,3) or (mode==1 and end>=after)


def capture(mic,stop,mode,epoch,after,robot_started,forced,crops,jobs,events):
    try:
        import sounddevice as sd
        from furhat_interaction.microphone import select_microphone
        from furhat_interaction.audio.vad import SileroVAD
        from furhat_interaction.active_speaker import ActiveSpeakerModel
        from pathlib import Path
        if not (ROOT/'models/speaker_association/light_talkset.model').exists():
            raise FileNotFoundError('Cached Light-ASD weights missing; no automatic downloads in interaction scenes.')
        model=ActiveSpeakerModel('light',threads=1);vad=SileroVAD(.6)
        device,info=select_microphone(sd,mic);pending=AudioInbox()
        history,audio,votes=deque(maxlen=90),deque(maxlen=45),deque(maxlen=100)
        segment=Utterance();evidence=BargeEvidence(.65)
        old_epoch,old_mode=-1,-1;next_asd,last_meter=0.,0.;last_identity=None
        def callback(data,frames,timing,status):
            age=max(0.,timing.currentTime-timing.inputBufferAdcTime-frames/16000)
            pending.put((perf_counter()-age,data[:,0].copy(),str(status) if status else ''))
        with sd.InputStream(device=device,samplerate=16000,channels=1,blocksize=512,dtype='float32',callback=callback):
            events.put(dict(type='ready',worker='audio',mic=info['name']))
            while not stop.is_set():
                while True:
                    try: history.append(crops.get_nowait())
                    except Empty: break
                try: (end,values,status),dropped=pending.get(timeout=.05)
                except Empty: continue
                if dropped or status:
                    segment.reset();vad.reset_states();audio.clear();votes.clear();evidence.reset();last_identity=None
                    events.put(dict(type='warning',message=f'Audio gap: {dropped} packets discarded; {status or "processing backlog"}. Partial speech reset; please repeat if needed.'))
                token,current=epoch.value,mode.value
                visual_transition=(token==old_epoch and old_mode==2 and current==3 and forced.value in (1,2))
                if token!=old_epoch or (current!=old_mode and (old_mode,current)!=(2,3)):
                    segment.reset();vad.reset_states();audio.clear();votes.clear();evidence.reset();last_identity=None
                old_epoch,old_mode=token,current
                # No ASR/ASD is consumed while the scene is muted. Previously
                # robot speech itself could trigger expensive two-face ASD.
                if not monitoring(current,end,after.value):
                    if perf_counter()-last_meter>.15:
                        events.put(dict(type='meter',epoch=token,voiced=False,db=float(20*np.log10(max(1e-8,np.sqrt(np.mean(values**2)))))))
                        last_meter=perf_counter()
                    continue
                voiced=vad.is_speech(np.rint(np.clip(values,-1,1)*32767).astype('<i2').tobytes(),16000)
                audio.append((end,values))
                if visual_transition and voiced and not segment.frames:
                    for at,chunk in list(audio)[-25:]: segment.feed(chunk,at,True)
                identity=None;scores={}
                if voiced and perf_counter()>=next_asd and len(audio)>=20 and history:
                    times=np.array([at for at,_ in history]);targets=end-.6+(np.arange(15)+.5)/25
                    indices=np.abs(times[:,None]-targets[None,:]).argmin(axis=0)
                    if np.max(np.abs(times[indices]-targets))<=.12:
                        ids=set(history[int(indices[0])][1])
                        for index in indices: ids.intersection_update(history[int(index)][1])
                        ids=sorted(ids)
                        if ids:
                            sound=np.concatenate([x for _,x in audio])[-9600:]
                            videos=[np.stack([history[int(i)][1][slot] for i in indices]) for slot in ids]
                            values_scores=model.score([sound],videos)[0]
                            scores={slot:float(score) for slot,score in zip(ids,values_scores)}
                            identity=speaker_choice(scores)
                    votes.append((end,identity))
                    events.put(dict(type='speaker',epoch=token,at=end,person=identity,scores=scores))
                    # Schedule from completion so a slow inference cannot
                    # trigger another inference immediately on queued audio.
                    next_asd=perf_counter()+.2
                if current==2:
                    force=forced.value
                    accepted=force in (1,2)
                    if not accepted and identity is not None and end-robot_started.value>.3:
                        if identity!=last_identity: evidence.reset()
                        last_identity=identity
                        accepted=evidence.update(scores[identity],end,True)
                    elif not voiced:
                        evidence.reset()
                    if accepted:
                        mode.value=3;current=3;segment.reset()
                        if voiced:
                            for at,chunk in list(audio)[-25:]: segment.feed(chunk,at,True)
                        events.put(dict(type='barge',epoch=token,person=force if force in (1,2) else identity,
                                        source='hand motion' if force in (1,2) else 'Light-ASD'))
                    else: continue
                else:
                    job=segment.feed(values,end,voiced)
                    if job:
                        job.update(epoch=token,person=freeze_speaker([slot for at,slot in votes if job['start']-.2<=at<=job['end']+.1]))
                        if job['final']:
                            jobs.put(job,timeout=2)
                            if epoch.value==token: mode.value=0
                        else:
                            try: jobs.put_nowait(job)
                            except Full: pass
                if perf_counter()-last_meter>.15:
                    events.put(dict(type='meter',epoch=token,voiced=bool(voiced),db=float(20*np.log10(max(1e-8,np.sqrt(np.mean(values**2)))))))
                    last_meter=perf_counter()
    except BaseException as exc:
        events.put(dict(type='error',message=f'Audio: {type(exc).__name__}: {exc}'))


def transcribe(model_name,stop,jobs,events):
    try:
        from types import SimpleNamespace
        from furhat_interaction.asr import WhisperBackend
        from furhat_interaction.text import clean_transcript
        backend=WhisperBackend(SimpleNamespace(engine='cpp',model=model_name,threads=2,cpp_device='cpu'))
        events.put(dict(type='ready',worker='asr'))
        while not stop.is_set():
            try: job=jobs.get(timeout=.1)
            except Empty: continue
            start=perf_counter();text=clean_transcript(backend.decode(job['audio']))
            events.put(dict(type='transcript',epoch=job['epoch'],person=job['person'],turn=job['turn'],
                            start=job['start'],end=job['end'],final=job['final'],text=text,
                            decode_ms=(perf_counter()-start)*1000))
    except BaseException as exc:
        events.put(dict(type='error',message=f'ASR: {type(exc).__name__}: {exc}'))
