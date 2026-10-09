"""Single-person conference Part 2: speech, silent head answer, three-round RPS.

Space starts after arranging windows; Q/Esc exits. Rehearsed 2-1 Furhat win:
show scissors, paper, scissors. Furhat commits to paper, scissors, rock in advance.
Actual observed gestures determine scores. No object model, ASD or LLM.
"""
import argparse
import asyncio
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Thread
from time import perf_counter, sleep
import numpy as np

from conference_ready_support import Utterance, asr_worker, extract_name
from conference_ready_game import HeadAnswer, Match, remembered_name, yes_no

ROOT = Path(__file__).resolve().parent


def audio_worker(mic, stop, mute_until, jobs, events):
    """Silero only; no speaker network is needed for the sole participant."""
    try:
        import sounddevice as sd
        from furhat_interaction.microphone import select_microphone
        from social_interaction.vad import SileroVAD
        device,info = select_microphone(sd,mic)
        vad,segment,pending = SileroVAD(.6),Utterance(),Queue(128)
        muted,last_meter = True,0.
        def callback(data, frames, timing, status):
            age = max(0.,timing.currentTime-timing.inputBufferAdcTime-frames/16000)
            try:
                pending.put_nowait((perf_counter()-age,data[:,0].copy(),str(status) if status else ''))
            except Full:
                events.put(dict(type='error',message='Microphone buffer overflow.'))
        with sd.InputStream(device=device,samplerate=16000,channels=1,blocksize=512,dtype='float32',callback=callback):
            events.put(dict(type='ready',worker='audio',mic=info['name']))
            while not stop.is_set():
                try:
                    end,values,status = pending.get(timeout=.05)
                except Empty:
                    continue
                if status:
                    raise RuntimeError('Microphone discontinuity: '+status)
                if end<mute_until.value:
                    if not muted:
                        segment.reset(); vad.reset_states()
                        events.put(dict(type='speech',active=False))
                    muted = True
                    continue
                muted = False
                pcm = np.rint(np.clip(values,-1,1)*32767).astype('<i2').tobytes()
                job = segment.feed(values,end,vad.is_speech(pcm,16000))
                if job:
                    if job['final']:
                        jobs.put(job,timeout=2)
                    else:
                        try:
                            jobs.put_nowait(job)
                        except Full:
                            pass
                if perf_counter()-last_meter>.1:
                    events.put(dict(type='speech',active=bool(segment.frames)))
                    last_meter = perf_counter()
    except BaseException as exc:
        events.put(dict(type='error',message=f'Audio: {type(exc).__name__}: {exc}'))


def speech_worker(robot, jobs, events, stop, mute_until):
    while not stop.is_set():
        try:
            job = jobs.get(timeout=.1)
        except Empty:
            continue
        try:
            mute_until.value = float('inf')
            events.put(dict(type='robot',text=job['text']))
            robot.say(job['text'])
            if stop.is_set():
                break
            if job.get('wink'):
                # SDK wait=True above confirms the ENTIRE win sentence finished.
                robot.call(robot.client.async_client.request_gesture_start('Wink',intensity=.6,duration=.7,wait=True))
            if stop.is_set():
                break
            if job.get('after_text'):
                events.put(dict(type='robot',text=job['after_text']))
                robot.say(job['after_text'])
            at = perf_counter()+.5
            mute_until.value = at if job['next'] in ('name','hobby','consent','rules') else float('inf')
            events.put(dict(type='stage',stage=job['next'],at=at))
        except Exception as exc:
            events.put(dict(type='error',message=f'Furhat: {type(exc).__name__}: {exc}'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mic',default='builtin')
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--name',help='Optional override; otherwise use the latest Part 1 name or ask.')
    parser.add_argument('--whisper',choices=['base.en','small.en'],default='base.en')
    parser.add_argument('--no-mirror',action='store_true')
    args = parser.parse_args()
    if args.mic.isdecimal():
        args.mic = int(args.mic)
    name = args.name or remembered_name(ROOT)
    ctx = mp.get_context('spawn')
    stop,mute_until = ctx.Event(),ctx.Value('d',float('inf'))
    asr_jobs,events,speech_jobs = ctx.Queue(8),ctx.Queue(),Queue()
    processes,thread,vision,camera,robot = [],None,None,None,None
    output = ROOT/'text_output/conference_ready_2'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    journal,failed,match = [],False,Match()
    def record(kind, **data):
        row = dict(time=perf_counter(),type=kind,**data)
        journal.append(row)
        with (output/'events.jsonl').open('a') as handle:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(f'[{kind.upper()}] {json.dumps(data,ensure_ascii=False)}',flush=True)
    try:
        from conference_ready_vision import Camera
        from conference_ready_game_vision import GameVision
        from group_interaction.robot import Robot
        print(f'[LOAD] Face and hands active throughout, Silero and Whisper.cpp {args.whisper}.',flush=True)
        vision = GameVision(); vision.enable_hands(); cv2 = vision.cv2
        for title,x in (('Camera - clean',20),('Perception and dialogue',680)):
            cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE); cv2.moveWindow(title,x,40)
        robot = Robot(args.host); robot.start()
        robot.call(robot.client.async_client.send_event(dict(type='request.attend.location',
            x=0.,y=0.,z=1.,slack_yaw=180.,slack_pitch=180.,slack_timeout=-1)))
        for target,values in ((audio_worker,(args.mic,stop,mute_until,asr_jobs,events)),
                              (asr_worker,(args.whisper,stop,asr_jobs,events))):
            process = ctx.Process(target=target,args=values,daemon=True)
            processes.append(process); process.start()
        thread = Thread(target=speech_worker,args=(robot,speech_jobs,events,stop,mute_until),daemon=True)
        thread.start(); camera = Camera(cv2,args.camera)
        ready,stage,status = set(),'loading','Loading audio...'
        lines,partial,speaking = [],'',False
        listen_since,last_final,sequence = float('inf'),0,-1
        head_answer = HeadAnswer()
        deadline,head,last_head,future = 0.,[0.,0.],0.,None
        frames,began = 0,perf_counter()
        def say(text,next_stage,wink=False,after_text=None):
            nonlocal stage,status,partial,speaking
            stage,status,partial,speaking = 'busy','Furhat speaking...','',False
            mute_until.value = float('inf')
            speech_jobs.put(dict(text=text,next=next_stage,wink=wink,after_text=after_text))
        def rules_answer(answer,source):
            record('rules_answer',answer=answer,source=source)
            if source=='head': vision.mark_head(answer,perf_counter())
            prefix = ('I saw your head shake. ' if source=='head' and not answer else
                      'I saw your nod. ' if source=='head' else '')
            if answer:
                say(prefix+'Great. We will play three rounds.','release')
            else:
                say(prefix+f'No problem, {name}. A fist is rock, an open hand is paper, and a V sign is scissors. '
                    'Rock beats scissors, scissors beats paper, and paper beats rock. We will play three rounds.','release')
        def keypress(key):
            if key==32 and stage=='armed':
                record('start',name=name,rehearsed=True,robot_moves=['paper','scissors','rock'])
                if name:
                    say(f'{name}, what do you like doing for fun?','hobby')
                else:
                    say('Before we play, what is your name?','name')
            return key in (27,ord('q'))
        print(f'[OUTPUT] {output}\n[SETUP] Arrange windows; click either camera window and press Space when ready.',flush=True)
        while not stop.is_set():
            now = perf_counter()
            for process in processes:
                if process.exitcode is not None:
                    raise RuntimeError(f'Audio worker exited unexpectedly ({process.exitcode}).')
            while True:
                try:
                    event = events.get_nowait()
                except Empty:
                    break
                kind = event['type']
                if kind=='error':
                    raise RuntimeError(event['message'])
                if kind=='ready':
                    ready.add(event['worker']); record('ready',**{k:v for k,v in event.items() if k!='type'})
                    if ready=={'audio','asr'}:
                        stage,status = 'armed','Arrange windows; SPACE starts the scene'
                elif kind=='speech':
                    speaking = event['active']
                elif kind=='robot':
                    lines.append('Furhat: '+event['text']); record('furhat',text=event['text'])
                elif kind=='stage':
                    stage,listen_since = event['stage'],event['at']
                    speaking,partial = False,''
                    status = {'name':'Listening for your name','hobby':'Listening...',
                              'consent':'Would you like to play? Answer aloud',
                              'rules':'Answer aloud, nod or shake your head',
                              'release':'Lower your hand out of view for the next round',
                              'capture':'Show ONE hand: fist / open palm / V sign',
                              'complete':'Scene complete. Q/Esc exits',
                              'paused':'Game declined. Q/Esc exits'}[stage]
                    if stage=='rules':
                        head_answer.reset()
                    if stage=='capture':
                        match.reset_hold(); deadline=listen_since+8.
                    if stage=='release':
                        match.release_since=None
                elif kind=='transcript':
                    if event['turn']<=last_final or event['start']<listen_since or stage not in ('name','hobby','consent','rules'):
                        continue
                    if not event['final']:
                        partial=f"{name or 'Person 1'} (partial): {event['text']}"; continue
                    last_final=event['turn']; partial=''
                    text=event['text']
                    record('transcript',speaker=name or 'Person 1',text=text,decode_ms=event['decode_ms'])
                    if not text:
                        status='No clear words; please try again'; continue
                    lines.append(f"{name or 'Person 1'}: {text}")
                    if stage=='name':
                        candidate=extract_name(text)
                        if candidate:
                            name=candidate; say(f'{name}, what do you like doing for fun?','hobby')
                        else:
                            say('Sorry, what should I call you?','name')
                    elif stage=='hobby':
                        say('We could try rock, paper, scissors. Would you like to play?','consent')
                    else:
                        answer=yes_no(text)
                        if answer is None:
                            say('You can say yes or no, or nod or shake your head.' if stage=='rules' else
                                'Would you like to play? Please say yes or no.',stage)
                        elif stage=='rules':
                            rules_answer(answer,'speech')
                        elif answer:
                            say('Do you know how to play?','rules')
                        else:
                            say('No problem. We can choose another activity later.','paused')
            if camera.error:
                raise RuntimeError(camera.error)
            snapshot=camera.snapshot()
            if snapshot is None or snapshot[0]==sequence:
                if keypress(cv2.waitKey(1)&255): break
                sleep(.002); continue
            sequence,at,frame=snapshot
            observation=vision.analyze(frame,at)
            h,w=frame.shape[:2]
            if stage=='rules' and at>=listen_since:
                answer=head_answer.update(observation['angles'],at)
                if answer is not None:
                    lines.append(f'{name}: [head shake / no]' if not answer else f'{name}: [nod / yes]')
                    rules_answer(answer,'head')
            if stage=='release' and at>=listen_since and observation['box'] is not None:
                if match.released(observation['hands'],at):
                    # robot_move is fixed BEFORE the gesture capture opens.
                    record('round_start',round=len(match.results)+1,committed_move=match.robot_move)
                    say(f'Round {len(match.results)+1}. Get ready. Rock, paper, scissors!','capture')
            elif stage=='capture' and at>=listen_since:
                if now>=deadline:
                    say('I could not see a clear move. Let us retry this round.','release')
                else:
                    result=match.observe(observation['hands'],at) if observation['box'] is not None else match.observe([],at)
                    if result:
                        record('round',**result)
                        outcome={'furhat':'I win this round.','person':'You win this round.','draw':'It is a draw.'}[result['winner']]
                        text=f"You showed {result['person']}. I chose {result['furhat']}. {outcome}"
                        next_stage='release'
                        after_text=None
                        celebrate=result['winner']=='furhat'
                        if len(match.results)==3:
                            scores=match.scores(); next_stage='complete'
                            celebrate=celebrate or scores['furhat']>scores['person']
                            winner='I won the game. Yay!' if scores['furhat']>scores['person'] else 'You won the game.' if scores['person']>scores['furhat'] else 'The game is a draw.'
                            text+=f" Final score: {scores['furhat']} to {scores['person']}. {winner}"
                            after_text=f'Thank you for playing, {name}.'
                        elif celebrate:
                            text+=' Yay!'
                        say(text,next_stage,wink=celebrate,after_text=after_text)
            if stage not in ('loading','armed') and observation['box'] is not None and now-last_head>=.1:
                if future is None or future.done():
                    if future is not None: future.result()
                    x1,y1,x2,y2=observation['box']
                    target=(-25*((x1+x2)/w-1),15*((y1+y2)/h-1))
                    head=[.75*a+.25*max(-limit,min(limit,b)) for a,b,limit in zip(head,target,(25,15))]
                    future=asyncio.run_coroutine_threadsafe(robot.client.async_client.request_face_headpose(float(head[0]),float(head[1]),0.,False),robot.client._loop)
                    last_head=now
            frames+=1
            scores=match.scores()
            detail=f"Round {min(3,len(match.results)+1)}/3 | {name or 'You'} {scores['person']} : Furhat {scores['furhat']}"
            display_lines=[detail]+lines[-(1 if partial else 2):]+([partial] if partial else [])
            # Keep score visible even when three dialogue lines are displayed.
            view_status=f'{status} | {frames/max(.01,now-began):.1f} FPS'
            speaker='Furhat' if stage=='busy' else (name or 'Person 1') if speaking else 'None'
            clean,canvas=vision.render(frame,observation,name,speaker,view_status,display_lines,not args.no_mirror)
            cv2.imshow('Camera - clean',clean); cv2.imshow('Perception and dialogue',canvas)
            if keypress(cv2.waitKey(1)&255): break
    except KeyboardInterrupt:
        print('Stopped.')
    except Exception as exc:
        failed=True; record('error',message=f'{type(exc).__name__}: {exc}')
    finally:
        stop.set()
        if robot:
            try: robot.stop_speech()
            except Exception: pass
        if thread: thread.join(2)
        for process in processes:
            process.join(3)
            if process.is_alive(): process.terminate(); process.join(2)
        if camera: camera.close()
        if vision: vision.close(); vision.cv2.destroyAllWindows()
        if robot: robot.close()
        (output/'session.json').write_text(json.dumps(dict(name=name,rehearsed=True,
            rounds=match.results,scores=match.scores(),events=journal),indent=2,ensure_ascii=False))
        print(f'[SAVED] {output}',flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    mp.freeze_support()
    raise SystemExit(main())
