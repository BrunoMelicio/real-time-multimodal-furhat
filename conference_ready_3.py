"""Part 3: disappointment, encouragement, speech/hand-motion interruption, reassurance.

Space starts; Q/Esc exits. Uses the latest completed Part 2 game, local Llama,
Whisper.cpp, Silero and one face model. Light-ASD guards interruptions by default.
Face and both hands run throughout; no object, body or separation models.
Rehearse with a mime near the face, without contact. Native Furhat speech/lip sync.
"""
import argparse
import asyncio
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
from queue import Queue,Empty,Full
from threading import Thread,Event
from time import perf_counter,sleep

from conference_ready_support import extract_name
from conference_ready_reflection import ReflectionLLM,CueHistory,game_memory,HandNearFace,HAND_CUE,reassuring_reply,reflective_reply,acknowledgment_action
from conference_ready_reflection_audio import audio_worker,asr_worker

ROOT=Path(__file__).resolve().parent


def speech_worker(robot,llm,jobs,events,stop,interrupted,mode,epoch,listen_after,robot_started):
    """Poll an interruptible SDK future; never wait for Whisper before stopping."""
    while not stop.is_set():
        try: job=jobs.get(timeout=.1)
        except Empty: continue
        token=job['epoch']
        try:
            text=job.get('text')
            if text is None:
                events.put(dict(type='thinking',epoch=token))
                try:
                    text=llm.respond(job['kind'],job['name'],job['speech'],job['memory'],job['previous'],job['cues'])
                except Exception as exc:
                    events.put(dict(type='warning',message=f'LLM fallback: {type(exc).__name__}: {exc}'))
                    text={'first':'It sounds like the game was disappointing. It is okay to lose; we can learn at your own pace.',
                          'support':'Thank you for telling me. We can pause and take this at your own pace.',
                          'choice':'Thank you for telling me. We can take this at your own pace.'}[job['kind']]
                if job['kind']=='support': text=reassuring_reply(text,job['cues'],job['speech'])
                if job['kind']=='first': text=reflective_reply(text,job['cues'])
            if stop.is_set() or token!=epoch.value: continue
            if job.get('gesture'):
                robot.call(robot.client.async_client.request_gesture_start(job['gesture'],intensity=.35,duration=.7,wait=False))
            events.put(dict(type='robot',epoch=token,text=text,interruptible=job.get('interruptible',False)))
            robot_started.value=perf_counter()
            mode.value=2 if job.get('interruptible') else 0
            future=asyncio.run_coroutine_threadsafe(robot.client.async_client.request_speak_text(text,wait=True,abort=True),robot.client._loop)
            while not future.done() and not stop.is_set() and not interrupted.is_set() and token==epoch.value and mode.value!=3:
                stop.wait(.015)
            if stop.is_set() or interrupted.is_set() or token!=epoch.value or mode.value==3:
                future.cancel()
                events.put(dict(type='interrupted',epoch=token))
                continue
            future.result()
            # Audio may validate a barge-in just as response_speak_end arrives.
            if interrupted.is_set() or mode.value==3:
                events.put(dict(type='interrupted',epoch=token)); continue
            at=perf_counter()+.5
            listen_after.value=at
            mode.value=1 if job['next']!='complete' else 0
            events.put(dict(type='stage',epoch=token,stage=job['next'],at=at))
        except Exception as exc:
            events.put(dict(type='error',message=f'Dialogue: {type(exc).__name__}: {exc}'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mic',default='builtin')
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--name',help='Optional override; otherwise use the completed Part 2 name.')
    p.add_argument('--model',default='llama3.2:1b')
    p.add_argument('--ollama-url',default='http://127.0.0.1:11434')
    p.add_argument('--whisper',choices=['base.en','small.en'],default='base.en')
    p.add_argument('--echo-guard',choices=['visual','vad'],default='visual',
                   help='visual: Silero + Light-ASD. vad: Silero only; use headphones to keep Furhat out of the mic.')
    p.add_argument('--vad-threshold',type=float,default=.6)
    p.add_argument('--asd-threshold',type=float,default=.65)
    p.add_argument('--no-mirror',action='store_true')
    args=p.parse_args()
    if not 0<args.vad_threshold<1 or not 0<args.asd_threshold<1: p.error('Thresholds must be between 0 and 1.')
    if args.mic.isdecimal(): args.mic=int(args.mic)
    memory=game_memory(ROOT);name=args.name or memory.get('name')
    if name: memory['name']=name
    ctx=mp.get_context('spawn')
    stop,mode,epoch=ctx.Event(),ctx.Value('i',0),ctx.Value('i',0)
    listen_after,robot_started=ctx.Value('d',float('inf')),ctx.Value('d',0.)
    force_barge=ctx.Value('i',-1)
    crops,asr_jobs,events=ctx.Queue(24),ctx.Queue(8),ctx.Queue()
    speech_jobs,interrupted=Queue(),Event()
    processes,thread,robot,vision,camera=[],None,None,None,None
    journal,chat,had_interruption,failed=[],[],False,False
    output=ROOT/'text_output/conference_ready_3'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    def record(kind,**data):
        row=dict(time=perf_counter(),type=kind,**data);journal.append(row)
        with (output/'events.jsonl').open('a') as handle: handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(f'[{kind.upper()}] {json.dumps(data,ensure_ascii=False)}',flush=True)
    try:
        from conference_ready_vision import Camera
        from conference_ready_reflection_vision import ReflectionVision
        from group_interaction.robot import Robot
        llm=ReflectionLLM(args.model,args.ollama_url);llm.check()
        vision=ReflectionVision();vision.enable_hands();cv2=vision.cv2
        for title,x in (('Camera - clean',20),('Perception and dialogue',680)):
            cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE);cv2.moveWindow(title,x,40)
        robot=Robot(args.host);robot.start()
        robot.call(robot.client.async_client.send_event(dict(type='request.attend.location',x=0.,y=0.,z=1.,slack_yaw=180.,slack_pitch=180.,slack_timeout=-1)))
        workers=((audio_worker,(args.mic,args.echo_guard,args.vad_threshold,args.asd_threshold,stop,mode,epoch,listen_after,robot_started,crops,asr_jobs,events,force_barge)),
                 (asr_worker,(args.whisper,stop,asr_jobs,events)))
        for target,values in workers:
            process=ctx.Process(target=target,args=values,daemon=True);processes.append(process);process.start()
        thread=Thread(target=speech_worker,args=(robot,llm,speech_jobs,events,stop,interrupted,mode,epoch,listen_after,robot_started),daemon=True)
        thread.start();camera=Camera(cv2,args.camera)
        phase,status,ready='loading','Loading audio models...',set()
        cues=CueHistory();lines=[];partial='';speaking=False;db=-160.;asd=0.;asd_at=0.
        motion=HandNearFace();hand_seen=False;hand_at=None;barge_token=-1
        sequence,last_final=-1,-1
        head,last_head,future=[0.,0.],0.,None
        voiced,voice_at,voice_since=False,0.,None
        nod_epoch,nod_until,next_nod,nod_future=-1,0.,0.,None
        frames,began=0,perf_counter()
        def queue_job(next_phase, text=None, kind=None, speech='', observations=(), gesture=None, interruptible=False):
            nonlocal phase,status,partial,speaking
            mode.value=0;epoch.value+=1;interrupted.clear()
            force_barge.value=-1
            phase,status,partial,speaking='busy','Thinking...' if kind else 'Furhat speaking...','',False
            speech_jobs.put(dict(epoch=epoch.value,next=next_phase,text=text,kind=kind,speech=speech,
                name=name or 'friend',memory=memory,previous=list(chat),cues=list(observations),
                gesture=gesture,interruptible=interruptible))
        def begin_interruption(source,score=None):
            nonlocal phase,status,had_interruption,barge_token
            if barge_token==epoch.value: return
            barge_token=epoch.value
            interrupted.set()
            if source=='hand_motion':
                # The audio worker retains pre-roll before changing 2 -> 3.
                force_barge.value=epoch.value
            else:
                mode.value=3
            phase,status='correction','INTERRUPTED: listening...'
            had_interruption=True
            if chat and chat[-1]['role']=='furhat': chat[-1]['interrupted']=True
            started=perf_counter()
            request=asyncio.run_coroutine_threadsafe(robot.client.async_client.request_speak_stop(),robot.client._loop)
            def stop_done(result):
                try:
                    result.result();events.put(dict(type='stop_ack',delay_ms=(perf_counter()-started)*1000))
                except Exception as exc: events.put(dict(type='error',message=f'Stop speech: {exc}'))
            request.add_done_callback(stop_done)
            record('interruption',guard=source,asd_score=score)
        def introduction():
            queue_job('first',text=f'{name}, how was the game for you?')
        def keypress(key):
            nonlocal phase,status,cues
            if key==32 and phase=='armed':
                phase,status,cues='calibrate','Look toward the camera briefly...',CueHistory()
                record('start',name=name,game_scores=memory.get('scores'),echo_guard=args.echo_guard)
            return key in (27,ord('q'))
        print(f'[OUTPUT] {output}\n[SETUP] Arrange windows. Space starts; Q/Esc exits. Llama={args.model}; echo guard={args.echo_guard}.',flush=True)
        while not stop.is_set():
            now=perf_counter()
            for process in processes:
                if process.exitcode is not None: raise RuntimeError(f'Audio worker exited unexpectedly ({process.exitcode}).')
            while True:
                try: event=events.get_nowait()
                except Empty: break
                kind=event['type']
                if kind=='error': raise RuntimeError(event['message'])
                if kind=='warning': record('warning',message=event['message'])
                elif kind=='ready':
                    ready.add(event['worker']);record('ready',**{k:v for k,v in event.items() if k!='type'})
                    if ready=={'audio','asr'}: phase,status='armed','Arrange windows; SPACE starts the scene'
                elif kind=='meter':
                    speaking,db=event['active'],event['db']
                    voiced,voice_at=event.get('voiced',False),perf_counter()
                elif kind=='asd': asd,asd_at=event['score'],event['at']
                elif kind in ('thinking','robot','stage','interrupted','barge') and event['epoch']!=epoch.value:
                    continue
                elif kind=='thinking': status='Thinking...'
                elif kind=='robot':
                    lines.append('Furhat: '+event['text']);chat.append(dict(role='furhat',text=event['text'],interrupted=False))
                    record('furhat',text=event['text'],interruptible=event['interruptible'])
                    status='Furhat speaking; interruption enabled' if event['interruptible'] else 'Furhat speaking...'
                elif kind=='barge':
                    begin_interruption(event['guard'],event['score'])
                elif kind=='stop_ack': record('stop_request',dispatch_ms=round(event['delay_ms'],1))
                elif kind=='interrupted':
                    # The next state belongs to retained speech, never the cancelled reply.
                    if phase=='busy': phase,status='correction','Listening to your correction...'
                elif kind=='stage':
                    phase=event['stage'];partial='';speaking=False
                    status={'name':'Listening for your name','first':'Tell me how the game felt',
                            'ack':'Listening for your acknowledgment',
                            'correction':'Listening; encouragement ended if not interrupted',
                            'final_ack':'Listening for your final response',
                            'complete':'Scene complete; Q/Esc exits'}[phase]
                elif kind=='transcript':
                    if event['epoch']!=epoch.value or event['turn']<=last_final or phase not in ('name','first','ack','correction','final_ack'):
                        continue
                    if not event['final']:
                        partial=f"{name or 'Person 1'} (partial): {event['text']}";continue
                    last_final=event['turn'];partial=''
                    text=event['text'];observations=cues.context(event['start'],event['end'])
                    if phase in ('ack','correction') and hand_seen: observations.append(HAND_CUE)
                    record('transcript',speaker=name or 'Person 1',text=text,visual_context=observations,
                           hand_detection=motion.diagnostics(),
                           decode_ms=round(event['decode_ms'],1),final_after_voice_ms=round((perf_counter()-event['end'])*1000,1))
                    if not text:
                        epoch.value+=1;listen_after.value=perf_counter();mode.value=1
                        status='No clear words; please try again';continue
                    lines.append(f"{name or 'Person 1'}: {text}");chat.append(dict(role='person',text=text))
                    if phase=='name':
                        name=extract_name(text)
                        if name: memory['name']=name;introduction()
                        else: queue_job('name',text='Sorry, what should I call you?')
                    elif phase=='first':
                        motion.reset();hand_seen=False;hand_at=None
                        queue_job('ack',kind='first',speech=text,observations=observations,gesture='Thoughtful')
                    elif phase=='ack':
                        action=acknowledgment_action(text,observations)
                        if action=='pause':
                            queue_job('complete',text='Of course. We can pause here.')
                        elif action=='support':
                            queue_job('final_ack',kind='support',speech=text,observations=observations,gesture='Thoughtful')
                        else:
                            motion.reset();hand_seen=False;hand_at=None
                            queue_job('correction',text='The more we practise, the more familiar the game can become. '
                                'We can try slowly, without keeping score, and focus on learning together rather than winning every round.',interruptible=True)
                    elif phase=='correction':
                        queue_job('final_ack',kind='support',speech=text,observations=observations,gesture='Thoughtful')
                    else:
                        queue_job('complete',kind='choice',speech=text,observations=observations,gesture='Smile')
            if camera.error: raise RuntimeError(camera.error)
            snapshot=camera.snapshot()
            if snapshot is None or snapshot[0]==sequence:
                if keypress(cv2.waitKey(1)&255): break
                sleep(.002);continue
            sequence,at,frame=snapshot
            observation=vision.analyze(frame,at);h,w=frame.shape[:2]
            if phase=='calibrate' and cues.calibrate(observation['angles'],observation['uv'],at):
                if name: introduction()
                else: queue_job('name',text='Before we talk, what should I call you?')
            visible_cues=cues.add(observation['angles'],observation['uv'],observation['scores'],at)
            if motion.update(observation['motion_box'],observation['hands'],at):
                hand_seen=True;hand_at=now
                record('visual_cue',observation=HAND_CUE)
                if mode.value==2: begin_interruption('hand_motion')
            if hand_seen: visible_cues.append(HAND_CUE)
            # A silent mime can also pause the robot; do not wait forever for ASR.
            if phase=='correction' and hand_seen and mode.value==3 and not speaking and hand_at is not None and now-hand_at>2.:
                queue_job('final_ack',text=reassuring_reply('',[HAND_CUE]),gesture='Thoughtful')
            if args.echo_guard=='visual' and observation['crop'] is not None:
                try: crops.put_nowait((at,observation['crop']))
                except Full: pass
            # A small acknowledgment of listening, never agreement with negative
            # self-talk. No nod during correction, robot speech or hand-safety cues.
            listening=(phase in ('first','ack') and mode.value==1 and now>=listen_after.value
                       and voiced and now-voice_at<.4 and not hand_seen and observation['box'] is not None)
            if not listening:
                voice_since=None
            elif voice_since is None:
                voice_since=now
            elif now-voice_since>=.55 and epoch.value!=nod_epoch and now>=next_nod and (future is None or future.done()):
                nod_future=asyncio.run_coroutine_threadsafe(
                    robot.client.async_client.request_gesture_start('Nod',intensity=.3,duration=.65,wait=True),robot.client._loop)
                nod_epoch=epoch.value;nod_until=now+.85;next_nod=now+5.
                record('backchannel',gesture='Nod')
            if nod_future is not None and nod_future.done():
                nod_future.result();nod_future=None
            # Let the native nod finish before resuming camera-driven head poses.
            if now>=nod_until and phase not in ('loading','armed','calibrate') and observation['box'] is not None and now-last_head>=.1:
                if future is None or future.done():
                    if future is not None: future.result()
                    x1,y1,x2,y2=observation['box']
                    target=(-25*((x1+x2)/w-1),15*((y1+y2)/h-1))
                    head=[.75*a+.25*max(-limit,min(limit,b)) for a,b,limit in zip(head,target,(25,15))]
                    future=asyncio.run_coroutine_threadsafe(robot.client.async_client.request_face_headpose(float(head[0]),float(head[1]),0.,False),robot.client._loop)
                    last_head=now
            frames+=1
            current=mode.value
            speaker=(name or 'Person 1') if speaking or current==3 else 'Furhat' if current==2 or status=='Furhat speaking...' else 'None'
            hand_status=(f"Hands: {len(observation['hands'])} | near face: {motion.near} | "
                         f"motion: {motion.motion_range:.2f}/0.18 | reversals: {motion.reversals}/2 | "
                         f"cue: {'DETECTED' if motion.fired else 'not detected'}")
            extras=([partial] if partial else ['Cues: '+(', '.join(visible_cues) if visible_cues else 'no sustained cue')])+[hand_status]
            lines_to_show=lines[-2:]+extras
            view_status=f'{status} | {frames/max(.01,now-began):.1f} FPS'
            clean,canvas=vision.render(frame,observation,name,speaker,view_status,lines_to_show,not args.no_mirror)
            asd_text=f'{asd:.2f}' if now-asd_at<1. else 'waiting for speech'
            cv2.putText(canvas,f'Mic {db:.0f} dBFS | Light-ASD: {asd_text}',(10,22),0,.48,(90,240,90),1,cv2.LINE_AA)
            cv2.imshow('Camera - clean',clean);cv2.imshow('Perception and dialogue',canvas)
            if keypress(cv2.waitKey(1)&255): break
    except KeyboardInterrupt:
        print('Stopped.')
    except Exception as exc:
        failed=True;record('error',message=f'{type(exc).__name__}: {exc}')
    finally:
        stop.set();interrupted.set();mode.value=0
        if robot:
            try: robot.stop_speech()
            except Exception: pass
        if thread: thread.join(2)
        for process in processes:
            process.join(3)
            if process.is_alive(): process.terminate();process.join(2)
        if camera: camera.close()
        if vision: vision.close();vision.cv2.destroyAllWindows()
        if robot: robot.close()
        (output/'session.json').write_text(json.dumps(dict(name=name,game=memory,
            interruption_detected=had_interruption,dialogue=chat,events=journal),indent=2,ensure_ascii=False))
        print(f'[SAVED] {output}',flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    mp.freeze_support()
    raise SystemExit(main())
