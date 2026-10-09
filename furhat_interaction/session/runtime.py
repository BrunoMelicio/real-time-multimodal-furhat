"""Continuous interaction using shared camera/audio/model owners."""
import argparse
import asyncio
import json
import multiprocessing as mp
from pathlib import Path
from queue import Queue,Empty,Full
from threading import Thread,Event
from time import perf_counter,sleep
from furhat_interaction.multi_person.performance import FrameStats
from furhat_interaction.multi_person.cues import acknowledge_downward
from furhat_interaction.dialogue import Ollama,clean_reply
from furhat_interaction.reflection import HAND_CUE,reassuring_reply,reflective_reply
from furhat_interaction.multi_person.audio import capture,transcribe
from furhat_interaction.multi_person.policies import center
from .controller import Session, decide

from furhat_interaction.paths import ROOT


def release_queues(*channels):
    # Readers are already stopped. Unread camera packets cannot be flushed;
    # waiting for their feeder threads here would hang Python's atexit phase.
    for channel in channels:
        channel.cancel_join_thread()
        channel.close()


def attention_target(part,state,confirmed,focus):
    if part==2 and state in ('release','playing','complete'): return None
    return confirmed or focus


def dialogue(robot,llm,jobs,events,stop,mode,epoch,after,robot_started):
    while not stop.is_set():
        try: job=jobs.get(timeout=.1)
        except Empty: continue
        token=job['epoch']
        try:
            text=job['text']
            if text is None:
                try:
                    action,text=decide(llm,job)
                    if action!='chat':
                        if not stop.is_set() and token==epoch.value:
                            events.put(dict(type='decision',epoch=token,action=action,person=job['target']))
                        continue
                except Exception as exc:
                    events.put(dict(type='warning',message=f'LLM fallback: {type(exc).__name__}: {exc}'))
                    text={'conversation':'Could you tell me a little more?', 'day':'Thank you for telling me about your day.',
                          'first':'It is okay to feel disappointed about losing; learning takes practice.',
                          'support':'Thank you for telling me. We can pause and take this at your own pace.',
                          'choice':'Thank you for telling me. We can take this at your own pace.'}[job['kind']]
                if job['kind']=='conversation':
                    text=acknowledge_downward(text,job.get('cues',[]))
                    text=reassuring_reply(text,job.get('cues',[]),job.get('speech',''))
                if job['kind']=='first': text=acknowledge_downward(text,job.get('cues',[]))
                if job['kind']=='support': text=reassuring_reply(text,job.get('cues',[]),job.get('speech',''))
            if stop.is_set() or token!=epoch.value: continue
            finished=True
            for segment in job.get('segments',[dict(text=text)]):
                if 'look_at' in segment:
                    events.put(dict(type='attention',epoch=token,target=segment['look_at']))
                events.put(dict(type='robot',epoch=token,text=segment['text'],interruptible=bool(job.get('interruptible'))))
                robot_started.value=perf_counter();mode.value=2 if job.get('interruptible') else 0
                future=asyncio.run_coroutine_threadsafe(robot.client.async_client.request_speak_text(segment['text'],wait=True,abort=True),robot.client._loop)
                while not future.done() and not stop.is_set() and token==epoch.value and mode.value!=3: stop.wait(.015)
                if stop.is_set() or token!=epoch.value or mode.value==3:
                    future.cancel();finished=False;break
                future.result()
            if not finished: continue
            if job.get('wink'):
                robot.call(robot.client.async_client.request_gesture_start('Wink',intensity=.5,duration=.7,wait=True))
            if job.get('after_text'):
                if 'segments' in job: events.put(dict(type='attention',epoch=token,target=job['target'] if job['next']=='after_game' else None))
                robot.say(job['after_text']);events.put(dict(type='robot',epoch=token,text=job['after_text']))
            after.value=perf_counter()+.5
            mode.value=0 if job['next']=='complete' else 1
            events.put(dict(type='stage',epoch=token,stage=job['next'],at=after.value))
        except Exception as exc: events.put(dict(type='error',message=f'Dialogue: {type(exc).__name__}: {exc}'))


def run(capacity=2,argv=None):
    parser=argparse.ArgumentParser(description='Continuous Furhat interaction: Space starts, Q/Esc exits.')
    parser.add_argument('--people',type=int,choices=(1,2),default=capacity,help='Participant capacity; larger groups coming soon.')
    parser.add_argument('--mic',default='builtin')
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--whisper',choices=['base.en','small.en'],default='base.en')
    parser.add_argument('--model',default='llama3.2:1b')
    parser.add_argument('--ollama-url',default='http://127.0.0.1:11434')
    parser.add_argument('--object-model',choices=['ssd','lite0','lite2'],default='lite0')
    parser.add_argument('--names',nargs='+',help='Optional names, left to right in the preview. Otherwise ask each newcomer.')
    parser.add_argument('--no-mirror',action='store_true')
    parser.add_argument('--profile',action='store_true',help='Print frame-stage timings; no files saved.')
    args=parser.parse_args(argv)
    capacity=args.people
    if args.mic.isdecimal(): args.mic=int(args.mic)
    if args.names and len(args.names)!=capacity:
        parser.error('--names must have one name per configured participant.')
    lines=[]
    def log(kind,**data):
        print(f'[{kind.upper()}] {json.dumps(data,ensure_ascii=False)}',flush=True)
    ctx=mp.get_context('spawn');stop=ctx.Event();mode=ctx.Value('i',0);epoch=ctx.Value('i',0)
    after=ctx.Value('d',float('inf'));robot_started=ctx.Value('d',0.);forced=ctx.Value('i',0)
    crops,jobs,events=ctx.Queue(24),ctx.Queue(8),ctx.Queue();speech_jobs=Queue()
    # Also covers an error during startup, before normal cleanup can run.
    for channel in (crops,jobs,events): channel.cancel_join_thread()
    processes=[];worker=robot=vision=camera=None
    scene=Session(capacity);failed=False
    try:
        from furhat_interaction.vision import Camera
        from furhat_interaction.group.robot import Robot
        from furhat_interaction.multi_person.vision import Vision
        llm=Ollama(args.model,args.ollama_url)
        if llm: llm.check()
        vision=Vision(3,args.object_model,continuous=True,max_people=capacity);cv2=vision.cv2
        vision.profile=args.profile
        for title,x in (('Camera - clean',20),('Perception and dialogue',680)):
            cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE);cv2.moveWindow(title,x,40)
        robot=Robot(args.host);robot.start()
        robot.call(robot.client.async_client.send_event(dict(type='request.attend.location',x=0.,y=0.,z=1.,
                   slack_yaw=180.,slack_pitch=180.,slack_timeout=-1)))
        for function,values in ((capture,(args.mic,stop,mode,epoch,after,robot_started,forced,crops,jobs,events)),
                                (transcribe,(args.whisper,stop,jobs,events))):
            process=ctx.Process(target=function,args=values,daemon=True);process.start();processes.append(process)
        worker=Thread(target=dialogue,args=(robot,llm,speech_jobs,events,stop,mode,epoch,after,robot_started),daemon=True);worker.start()
        camera=Camera(cv2,args.camera);ready=set();started=False;status='Loading audio...'
        active=None;active_at=0.;partial='';voiced=False;voice_at=0.;voice_since=None
        last_final=-1;sequence=-1;obs=None;focus=None;head_future=None;head_at=0.;head=[0.,0.]
        nod_until=0.;nod_epoch=-1;next_nod=0.;nod_future=None
        glance=None;glance_until=0.;glanced=set();barge_epoch=-1;hand_at=None
        stats=FrameStats(args.profile)
        def submit(action):
            nonlocal status,partial,active,voice_since
            if action is None: return
            mode.value=0;epoch.value+=1;forced.value=0;partial='';active=None;voice_since=None
            status='Thinking...' if action['text'] is None else 'Furhat speaking...'
            speech_jobs.put(dict(action,epoch=epoch.value,name=scene.label(action['target']),
                                 game=scene.memory,history=list(scene.history)))
        def resume():
            epoch.value+=1;forced.value=0;after.value=perf_counter();mode.value=1
        def interrupt(person,source):
            nonlocal barge_epoch,status,hand_at
            if person not in (1,2) or barge_epoch==epoch.value: return
            barge_epoch=epoch.value;mode.value=3;scene.state='correction';scene.focus=person;scene.loser=person;scene.part=3
            status=f'Interrupted: listening to {scene.label(person)}'
            if source=='hand motion': forced.value=person;hand_at=perf_counter()
            request=asyncio.run_coroutine_threadsafe(robot.client.async_client.request_speak_stop(),robot.client._loop)
            def stopped(f):
                try: f.result()
                except Exception as exc: events.put(dict(type='error',message=f'Stop speech: {exc}'))
            request.add_done_callback(stopped)
            log('interruption',person=scene.label(person),source=source)
        def keypress(key):
            nonlocal started,status
            if key==32 and not started and ready=={'audio','asr'} and obs:
                stable=[p for p in obs['people'] if vision.seats.stable(p['id'],perf_counter())]
                if args.names:
                    if len(stable)!=capacity:
                        status='All named participants must be visible before Space.';return False
                    order=sorted(stable,key=lambda p:center(p['box'])[0],reverse=not args.no_mirror)
                    scene.names={p['id']:n for p,n in zip(order,args.names)}
                for slot,detector in vision.cues.items():
                    refreshed=detector.rebase(perf_counter())
                    log('posture_baseline',person=scene.label(slot),refreshed=refreshed,**detector.diagnostics(perf_counter()))
                started=True;log('start',capacity=capacity,names=scene.names)
                action=scene.start(obs)
                if action: submit(action)
                else: resume();status='Listening; welcome to Furhat'
            return key in (27,ord('q'))
        print('[SETUP] Space starts; Q/Esc exits. Terminal logs only; no run files are saved.',flush=True)
        while not stop.is_set():
            now=perf_counter()
            for p in processes:
                if p.exitcode is not None: raise RuntimeError(f'Worker exited ({p.exitcode}).')
            while True:
                try: event=events.get_nowait()
                except Empty: break
                kind=event['type']
                if kind=='error': raise RuntimeError(event['message'])
                if kind=='warning': log('warning',message=event['message']);continue
                if kind=='ready':
                    ready.add(event['worker']);log('ready',**{k:v for k,v in event.items() if k!='type'})
                    if ready=={'audio','asr'}: status='Arrange windows; SPACE starts'
                    continue
                if event.get('epoch')!=epoch.value: continue
                if kind=='speaker': active,active_at=event['person'],event['at']
                elif kind=='attention': scene.focus=event['target']
                elif kind=='meter': voiced,voice_at=event['voiced'],perf_counter()
                elif kind=='robot':
                    lines.append('Furhat: '+event['text']);lines[:]=lines[-12:];log('furhat',text=event['text'])
                    scene.history.append(dict(person='Furhat',speech=event['text']))
                    scene.history=scene.history[-12:]
                    if scene.part==3 and event.get('interruptible'):
                        status='Furhat speaking; INTERRUPTION ENABLED'
                        log('interruption_ready',person=scene.label(scene.loser),guard='Light-ASD or repeated hand motion near face')
                elif kind=='decision':
                    action=scene.apply_decision(event['action'],event['person'])
                    if action: submit(action)
                    else: scene.stage('chat',perf_counter());resume()
                elif kind=='barge': interrupt(event['person'],event['source'])
                elif kind=='stage':
                    scene.stage(event['stage'],event['at']);partial=''
                    status=f'Listening: {scene.state}'
                    if scene.state=='rules':
                        for detector in vision.head.values(): detector.reset()
                        vision.head_cooldown={1:0.,2:0.}
                    if scene.state=='correction':
                        scene.hand_seen.clear()
                        for detector in vision.motion.values(): detector.reset()
                elif kind=='transcript':
                    if event['turn']<=last_final or scene.state=='busy': continue
                    if scene.expected() is None and scene.state not in ('rules','correction','chat','reflect','release','playing'):
                        if event['final']:
                            last_final=event['turn']
                            if not scene.done: resume()
                        continue
                    slot=event['person'];text=event['text']
                    if not event['final']:
                        partial=f"{scene.label(slot) if slot else 'Person uncertain'} (partial): {text}";continue
                    last_final=event['turn'];partial=''
                    cues=vision.cues[slot].context(event['start'],event['end']) if slot in (1,2) else []
                    if slot in scene.hand_seen: cues.append(HAND_CUE)
                    log('transcript',person=scene.names.get(slot,slot),text=text,visual_context=cues,
                        object_context=vision.object_for(slot,event['end']) if slot in (1,2) else None,
                        head_detection={scene.label(i):d.metrics for i,d in vision.head.items()} if scene.part==2 and scene.state=='rules' else None,
                        decode_ms=round(event['decode_ms'],1),final_after_voice_ms=round((perf_counter()-event['end'])*1000,1),
                        hand_detection=vision.motion[slot].diagnostics() if slot in (1,2) and scene.part==3 else None)
                    if scene.part==3 and slot in (1,2):
                        log('posture',person=scene.label(slot),used_cues=cues,**vision.cues[slot].diagnostics(perf_counter()))
                    if not text or slot is None:
                        state=scene.state
                        submit(scene.say('I could not confidently tell who spoke. Please face the camera and repeat your full sentence.',state,scene.expected()))
                        continue
                    lines.append(f'{scene.label(slot)}: {text}');lines[:]=lines[-12:]
                    action=scene.transcript(slot,text,cues,vision.object_for(slot,event['end']))
                    scene.hand_seen.clear()
                    if action: submit(action)
                    else: resume()
            if camera.error: raise RuntimeError(camera.error)
            snapshot=camera.snapshot()
            if snapshot is None or snapshot[0]==sequence:
                if keypress(cv2.waitKey(1)&255): break
                sleep(.002);continue
            sequence,at,frame=snapshot
            observe_at=perf_counter() if args.profile else 0.
            obs=vision.observe(frame,at)
            observe_end=perf_counter() if args.profile else 0.
            try: crops.put_nowait((at,obs['crops']))
            except Full: pass
            # No eye/contact or hand evidence is fabricated for a hidden person.
            for slot in obs['motion']:
                scene.hand_seen.add(slot);log('visual_cue',person=scene.label(slot),observation=HAND_CUE)
                if mode.value==2: interrupt(slot,'hand motion')
            if started:
                if scene.part==2 and scene.state=='rules' and now>=after.value:
                    for slot,answer in obs['head_events'].items():
                        log('head_gesture',person=scene.label(slot),gesture='nod' if answer else 'head shake',**vision.head[slot].metrics)
                if scene.state!='busy' and mode.value==1 and not voiced:
                    submit(scene.observe(obs,now,vision,now>=after.value))
            if scene.part==3 and scene.state=='correction' and mode.value==3 and hand_at and now-hand_at>2. and not voiced:
                submit(scene.say(reassuring_reply('',[HAND_CUE]),'reflect',scene.loser))
            visible={p['id']:p for p in obs['people']}
            if started:
                new=[i for i in visible if i not in scene.names and i not in glanced and vision.seats.stable(i,now)]
                if new:
                    glance=new[0];glanced.add(glance);glance_until=now+.4
            confirmed=active if now-active_at<.8 and active in visible and mode.value in (1,3) else None
            focus=glance if now<glance_until else attention_target(scene.part,scene.state,confirmed,scene.focus)
            listening=(started and scene.state in ('chat','reflect') and mode.value==1 and voiced
                       and now-voice_at<.4 and confirmed is not None and confirmed not in scene.hand_seen)
            if not listening: voice_since=None
            elif voice_since is None: voice_since=now
            elif now-voice_since>.55 and nod_epoch!=epoch.value and now>next_nod:
                nod_future=asyncio.run_coroutine_threadsafe(robot.client.async_client.request_gesture_start('Nod',intensity=.3,duration=.65,wait=True),robot.client._loop)
                nod_epoch=epoch.value;nod_until=now+.85;next_nod=now+5.;log('backchannel',gesture='Nod')
            if nod_future is not None and nod_future.done(): nod_future.result();nod_future=None
            neutral=scene.part==2 and focus is None
            if (focus in visible or neutral) and started and now>nod_until and now-head_at>.1 and (head_future is None or head_future.done()):
                if head_future: head_future.result()
                if neutral: target=(0.,0.)
                else:
                    box=obs['faces'].get(focus,visible[focus])['box'];x,y=center(box)
                    target=(-25*(x/(obs['width']/2)-1),15*(y/(obs['height']/2)-1))
                head=[.7*a+.3*max(-limit,min(limit,b)) for a,b,limit in zip(head,target,(25,15))]
                head_future=asyncio.run_coroutine_threadsafe(robot.client.async_client.request_face_headpose(*map(float,head),0.,False),robot.client._loop)
                head_at=now
            detail=(f"Scores: "+' | '.join(f'{scene.label(i)} {s}' for i,s in scene.game.scores.items())) if scene.part==2 else ''
            render_at=perf_counter() if args.profile else 0.
            clean,canvas=vision.render(frame,obs,scene.names,confirmed,focus,
                f'{status} | {stats.fps:.1f} FPS',lines[-2:]+([partial] if partial else [detail]),not args.no_mirror)
            render_end=perf_counter() if args.profile else 0.
            cv2.imshow('Camera - clean',clean);cv2.imshow('Perception and dialogue',canvas)
            key=cv2.waitKey(1)&255
            end=perf_counter()
            timings=dict(vision.timings,vision_ms=(observe_end-observe_at)*1000,
                         render_ms=(render_end-render_at)*1000,display_ms=(end-render_end)*1000,
                         frame_ms=(end-now)*1000,camera_age_ms=max(0.,end-at)*1000) if args.profile else {}
            report=stats.record(end,timings)
            if report is not None: log('perf',**report)
            if keypress(key): break
    except KeyboardInterrupt: print('Stopped.')
    except Exception as exc:
        failed=True;log('error',message=f'{type(exc).__name__}: {exc}')
    finally:
        stop.set();mode.value=0
        if robot:
            try: robot.stop_speech()
            except Exception: pass
        if worker: worker.join(2)
        for p in processes:
            p.join(3)
            if p.is_alive(): p.terminate();p.join(2)
        if camera: camera.close()
        if vision: vision.close();vision.cv2.destroyAllWindows()
        if robot: robot.close()
        release_queues(crops,jobs,events)
        print('[CLOSED] No run files saved.',flush=True)
    return int(failed)
