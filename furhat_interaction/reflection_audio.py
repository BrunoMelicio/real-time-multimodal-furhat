"""Part 3 streaming audio. Stop on validated speech onset; ASR finishes later."""
from collections import deque
from queue import Queue,Empty,Full
from time import perf_counter
import numpy as np
from furhat_interaction.dialogue import Utterance
from furhat_interaction.reflection import BargeEvidence


def audio_worker(mic,guard,vad_threshold,asd_threshold,stop,mode,epoch,listen_after,robot_started,crops,jobs,events,force_barge=None):
    try:
        import sounddevice as sd
        from furhat_interaction.microphone import select_microphone
        from furhat_interaction.audio.vad import SileroVAD
        model=None
        if guard=='visual':
            from furhat_interaction.active_speaker import ActiveSpeakerModel
            model=ActiveSpeakerModel('light',threads=1)
        vad=SileroVAD(vad_threshold)
        device,info=select_microphone(sd,mic)
        pending=Queue(128)
        faces,audio=deque(maxlen=70),deque(maxlen=45)
        segment,evidence=Utterance(),BargeEvidence(asd_threshold)
        seen_epoch,seen_mode,last_meter,next_asd,onset=-1,-1,0.,0.,0
        def callback(data,frames,timing,status):
            age=max(0.,timing.currentTime-timing.inputBufferAdcTime-frames/16000)
            try: pending.put_nowait((perf_counter()-age,data[:,0].copy(),str(status) if status else ''))
            except Full: events.put(dict(type='error',message='Microphone callback overflow.'))
        with sd.InputStream(device=device,samplerate=16000,channels=1,blocksize=512,dtype='float32',callback=callback):
            events.put(dict(type='ready',worker='audio',mic=info['name'],echo_guard=guard))
            while not stop.is_set():
                while True:
                    try: faces.append(crops.get_nowait())
                    except Empty: break
                try: end,values,status=pending.get(timeout=.05)
                except Empty: continue
                if status: raise RuntimeError('Microphone discontinuity: '+status)
                token,current=epoch.value,mode.value
                # Preserve only the validated 2->3 barge-in buffer. Other phase
                # changes must not carry robot-voice VAD state into listening.
                if token!=seen_epoch or (current!=seen_mode and (seen_mode,current)!=(2,3)):
                    segment.reset();vad.reset_states();audio.clear();evidence.reset();onset=0
                seen_epoch,seen_mode=token,current
                db=20*np.log10(max(1e-8,float(np.sqrt(np.mean(values**2)))))
                voiced=vad.is_speech(np.rint(np.clip(values,-1,1)*32767).astype('<i2').tobytes(),16000)
                audio.append((end,values))
                onset=onset+1 if voiced else 0
                score=None
                # Monitor active-speaker evidence in every scene phase. Only
                # the interruptible speaking phase may act on it as a barge-in.
                if model is not None and voiced and perf_counter()>=next_asd and len(audio)>=20 and faces:
                    next_asd=perf_counter()+.2
                    times=np.array([t for t,_ in faces])
                    targets=end-.6+(np.arange(15)+.5)/25
                    indices=np.abs(times[:,None]-targets[None,:]).argmin(axis=0)
                    if np.max(np.abs(times[indices]-targets))<=.1:
                        sound=np.concatenate([x for _,x in audio])[-9600:]
                        video=np.stack([faces[int(i)][1] for i in indices])
                        score=float(model.score([sound],[video])[0,0])
                        events.put(dict(type='asd',score=score,at=end))
                if current==0 or (current==1 and end<listen_after.value):
                    if perf_counter()-last_meter>.15:
                        events.put(dict(type='meter',active=False,db=db,mode=current))
                        last_meter=perf_counter()
                    continue
                if current==2:
                    accepted=False
                    forced=force_barge is not None and force_barge.value==token
                    if forced:
                        accepted=True
                    elif end-robot_started.value>=.3 and onset>=6:
                        if guard=='vad':
                            accepted=True
                        elif score is not None:
                            accepted=evidence.update(score,end,voiced)
                    if not voiced: evidence.reset()
                    if accepted:
                        # Retain a little onset audio. This is not source separation/AEC.
                        segment.reset()
                        if voiced:
                            for at,chunk in list(audio)[-9:]: segment.feed(chunk,at,True)
                        mode.value=3;current=3
                        events.put(dict(type='barge',epoch=token,at=perf_counter(),audio_at=end,
                                        score=score,guard='hand_motion' if forced else guard))
                    else:
                        if perf_counter()-last_meter>.15:
                            events.put(dict(type='meter',active=False,db=db,mode=2))
                            last_meter=perf_counter()
                        continue
                else:
                    job=segment.feed(values,end,voiced)
                    if job:
                        job['epoch']=token
                        if job['final']:
                            jobs.put(job,timeout=2)
                            # One completed utterance per listening turn.
                            if epoch.value==token: mode.value=0
                        else:
                            try: jobs.put_nowait(job)
                            except Full: pass
                if perf_counter()-last_meter>.15:
                    events.put(dict(type='meter',active=bool(segment.frames),voiced=bool(voiced),db=db,mode=current))
                    last_meter=perf_counter()
    except BaseException as exc:
        events.put(dict(type='error',message=f'Audio: {type(exc).__name__}: {exc}'))


def asr_worker(model_name,stop,jobs,events):
    try:
        from types import SimpleNamespace
        from furhat_interaction.asr import WhisperBackend
        from furhat_interaction.text import clean_transcript
        backend=WhisperBackend(SimpleNamespace(engine='cpp',model=model_name,threads=2,cpp_device='cpu'))
        events.put(dict(type='ready',worker='asr'))
        while not stop.is_set():
            try: job=jobs.get(timeout=.1)
            except Empty: continue
            started=perf_counter()
            text=clean_transcript(backend.decode(job['audio']))
            events.put(dict(type='transcript',epoch=job['epoch'],turn=job['turn'],start=job['start'],end=job['end'],
                            final=job['final'],text=text,decode_ms=(perf_counter()-started)*1000))
    except BaseException as exc:
        events.put(dict(type='error',message=f'Whisper: {type(exc).__name__}: {exc}'))
