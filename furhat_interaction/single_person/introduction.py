"""Single-person interaction scene: arrival, introduction, object-aware dialogue.

Run: .venv/bin/python main.py --people=1 --scenario=introduction --mic=builtin
Q/Esc closes both views. Uses the configured Furhat voice and local Ollama.
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

from furhat_interaction.dialogue import Arrival, VisualHistory, Ollama, extract_name, audio_worker, asr_worker, object_acknowledgment
from furhat_interaction.vision import OBJECT_MODELS

from furhat_interaction.paths import ROOT


def dialogue_worker(robot, llm, jobs, events, stop, mute_until):
    while not stop.is_set():
        try:
            job = jobs.get(timeout=.1)
        except Empty:
            continue
        try:
            # This scene is turn based: suppress robot echo during generation/speech.
            mute_until.value = float('inf')
            if 'context' in job:
                events.put(dict(type='thinking'))
                text = llm.reply(job['name'], job['speech'], job['context'])
                text, fallback = object_acknowledgment(text,job['context'])
                events.put(dict(type='reply_context',object=job['context'][0]['object'] if job['context'] else None,
                                acknowledgment_fallback=fallback))
            else:
                text = job['text']
            if stop.is_set():
                break
            events.put(dict(type='robot', text=text))
            robot.say(text)
            at = perf_counter()+.5
            mute_until.value = at
            events.put(dict(type='stage', stage=job['next'], at=at))
        except Exception as exc:
            events.put(dict(type='error', message=f'Dialogue: {type(exc).__name__}: {exc}'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mic', default='builtin', help='builtin, iphone, name, or current input ID')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--model', default='llama3.2:1b')
    parser.add_argument('--whisper', choices=['base.en','small.en'], default='base.en')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--no-mirror', action='store_true')
    parser.add_argument('--object-model', choices=OBJECT_MODELS, default='ssd',
                        help='One cached detector: MediaPipe ssd/lite0/lite2 or YOLO26 n/s/m.')
    parser.add_argument('--object-confidence', type=float, default=.3)
    parser.add_argument('--object-device', choices=['auto','cpu','mps'],default='auto',
                        help='YOLO only: auto uses Apple GPU when available. MediaPipe uses CPU.')
    parser.add_argument('--object-imgsz',type=int,default=640,help='YOLO input size; MediaPipe uses checkpoint input size.')
    args = parser.parse_args()
    if not 0 < args.object_confidence < 1 or args.object_imgsz < 32:
        parser.error('object-confidence must be between 0 and 1; object-imgsz must be >=32.')
    if not args.object_model.startswith('yolo-') and args.object_device == 'mps':
        parser.error('MediaPipe uses CPU here; choose --object-device=auto or cpu.')
    if args.mic.isdecimal():
        args.mic = int(args.mic)
    ctx = mp.get_context('spawn')
    stop, mute_until = ctx.Event(), ctx.Value('d', float('inf'))
    crops, asr_jobs, events = ctx.Queue(24), ctx.Queue(8), ctx.Queue()
    dialogue_jobs = Queue()
    processes, thread, robot, vision, camera = [], None, None, None, None
    output = ROOT/'text_output/single_person/introduction'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    journal, failed = [], False
    def record(kind, **data):
        row = dict(time=perf_counter(), type=kind, **data)
        journal.append(row)
        with (output/'events.jsonl').open('a') as handle:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(f'[{kind.upper()}] {json.dumps(data,ensure_ascii=False)}',flush=True)
    try:
        llm = Ollama(args.model,args.ollama_url)
        llm.check()
        print(f'[LOAD] One face model + {args.object_model}; Silero/Light-ASD and Whisper.cpp in separate workers.',flush=True)
        from furhat_interaction.vision import Vision, Camera
        from furhat_interaction.group.robot import Robot
        vision = Vision(args.object_model,args.object_confidence,args.object_device,args.object_imgsz)
        record('detector',**vision.metadata)
        cv2 = vision.cv2
        for title,x in (('Camera - clean',20),('Perception and dialogue',680)):
            cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(title,x,40)
        robot = Robot(args.host); robot.start()
        # Independent head position commands; disable competing automatic head attention.
        robot.call(robot.client.async_client.send_event(dict(type='request.attend.location',
            x=0.,y=0.,z=1.,slack_yaw=180.,slack_pitch=180.,slack_timeout=-1)))
        for target, values in ((audio_worker,(args.mic,stop,mute_until,crops,asr_jobs,events)),
                               (asr_worker,(args.whisper,stop,asr_jobs,events))):
            process = ctx.Process(target=target,args=values,daemon=True)
            processes.append(process); process.start()
        thread = Thread(target=dialogue_worker,args=(robot,llm,dialogue_jobs,events,stop,mute_until),daemon=True)
        thread.start()
        camera = Camera(cv2,args.camera)
        arrival, history = Arrival(), VisualHistory()
        ready, stage, name = set(), 'loading', None
        lines, partial, status = [], '', 'Loading audio models...'
        speaking, asd, asd_at, db = False, 0., 0., -160.
        sequence, last_final, listen_since = -1, 0, float('inf')
        last_head, head, future = 0., [0.,0.], None
        last_log, frames, fps, began = perf_counter(), 0, 0., perf_counter()
        def queue_speech(text, next_stage, **extra):
            nonlocal stage, status, partial, speaking
            stage, status, partial, speaking = 'busy','Furhat responding...', '',False
            mute_until.value = float('inf')
            dialogue_jobs.put(dict(text=text,next=next_stage,**extra))
        def handle_key(key):
            nonlocal stage, status, arrival, history
            if key == 32 and stage == 'armed':
                arrival, history = Arrival(), VisualHistory()
                stage, status = 'arrival','Waiting for one person...'
                record('start',message='Space pressed; scenario started.')
            return key in (27,ord('q'))
        print(f'[OUTPUT] {output}\n[SETUP] Arrange windows, then focus either camera window and press Space when ready. Q/Esc exits.',flush=True)
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
                if kind == 'error':
                    raise RuntimeError(event['message'])
                if kind == 'ready':
                    ready.add(event['worker']); record('ready',**{k:v for k,v in event.items() if k!='type'})
                    if ready == {'audio','asr'}:
                        stage, status = 'armed','Arrange windows, then press SPACE to start'
                        mute_until.value = float('inf')
                        print('[READY] Press SPACE in either camera window to start.',flush=True)
                elif kind == 'speech':
                    speaking = event['active']
                    db = event.get('db',db)
                elif kind == 'asd':
                    asd, asd_at = event['score'],event['at']
                elif kind == 'thinking':
                    status = 'Thinking...'
                elif kind == 'reply_context':
                    record('reply_context',object=event['object'],acknowledgment_fallback=event['acknowledgment_fallback'])
                elif kind == 'robot':
                    lines.append('Furhat: '+event['text']); record('furhat',text=event['text'])
                elif kind == 'stage':
                    stage, listen_since = event['stage'],event['at']
                    speaking, partial = False,''
                    status = 'Listening...' if stage in ('name','answer','followup') else 'Please settle into your seat...'
                    if stage == 'settle':
                        arrival.still_since = now
                        mute_until.value = float('inf')
                    if stage == 'complete':
                        status = 'Scene complete. Q/Esc exits; head following stays active.'
                        mute_until.value = float('inf')
                elif kind == 'transcript':
                    if event['turn'] <= last_final or event['start'] < listen_since or stage not in ('name','answer','followup'):
                        continue
                    if not event['final']:
                        partial = f"{name or 'Person 1'} (partial): {event['text']}"
                        continue
                    last_final = event['turn']; partial = ''
                    text = event['text']
                    context = history.context(event['start'],event['end'])
                    record('transcript',speaker=name or 'Person 1',text=text,context=context,
                           vision_counts=history.summary(event['start'],event['end']),decode_ms=event['decode_ms'])
                    if not text:
                        status = 'No clear words. Please try again.'
                        continue
                    lines.append(f"{name or 'Person 1'}: {text}")
                    if stage == 'name':
                        candidate = extract_name(text)
                        if not candidate:
                            queue_speech('Sorry, what should I call you?','name')
                        else:
                            name = candidate
                            queue_speech(f'Nice to meet you, {name}. How has your day been?','answer')
                    elif stage == 'answer':
                        queue_speech('', 'followup',name=name,speech=text,context=context)
                    else:
                        queue_speech(f'Thanks for sharing, {name}. Let us try something fun next.','complete')
            if camera.error:
                raise RuntimeError(camera.error)
            snapshot = camera.snapshot()
            if snapshot is None or snapshot[0] == sequence:
                if handle_key(cv2.waitKey(1)&255):
                    break
                sleep(.002); continue
            sequence, at, frame = snapshot
            observation = vision.analyze(frame,at)
            h,w = frame.shape[:2]
            if observation['fresh']:
                history.add(at,observation['objects'])
            if observation['crop'] is not None:
                try:
                    crops.put_nowait((at,observation['crop']))
                except Full:
                    pass
            present, settled = arrival.update(observation['box'],w,h,at)
            if stage == 'arrival' and present:
                queue_speech('Hello! Welcome. Please take a seat.','settle')
            elif stage == 'settle' and present and settled and now >= listen_since:
                queue_speech('What is your name?','name')
            if stage not in ('loading','armed') and observation['box'] is not None and now-last_head >= .1:
                if future is None or future.done():
                    if future is not None:
                        future.result()  # Surface failed SDK commands.
                    x1,y1,x2,y2 = observation['box']
                    target = [-25*((x1+x2)/w-1), 15*((y1+y2)/h-1)]
                    head = [.75*a+.25*max(-limit,min(limit,b)) for a,b,limit in zip(head,target,(25,15))]
                    future = asyncio.run_coroutine_threadsafe(robot.client.async_client.request_face_headpose(
                        float(head[0]),float(head[1]),0.,False),robot.client._loop)
                    last_head = now
            confirmed = speaking and now-asd_at < .9 and asd >= .65
            speaker = 'Furhat' if stage=='busy' and status!='Thinking...' else (
                (name or 'Person 1') if confirmed else 'Speech heard; visual match uncertain' if speaking else 'None')
            frames += 1
            fps = frames/max(.01,now-began)
            view_status = f'{status} | {fps:.1f} FPS | mic {db:.0f} dB | ASD {asd:.2f}'
            clean, diagnostic = vision.render(frame,observation,name,speaker,view_status,lines+([partial] if partial else []),not args.no_mirror)
            cv2.imshow('Camera - clean',clean)
            cv2.imshow('Perception and dialogue',diagnostic)
            if now-last_log >= 5:
                record('health',stage=stage,fps=round(fps,1),mic_db=round(db,1),asd=round(asd,2))
                last_log = now
            if handle_key(cv2.waitKey(1)&255):
                break
    except KeyboardInterrupt:
        print('Stopped.')
    except Exception as exc:
        failed = True
        record('error',message=f'{type(exc).__name__}: {exc}')
    finally:
        stop.set()
        if robot:
            try:
                robot.stop_speech()
            except Exception:
                pass
        if thread:
            thread.join(2)
        for process in processes:
            process.join(3)
            if process.is_alive():
                process.terminate(); process.join(2)
        if camera:
            camera.close()
        if vision:
            vision.close(); vision.cv2.destroyAllWindows()
        if robot:
            robot.close()
        (output/'session.json').write_text(json.dumps(dict(name=locals().get('name'),events=journal),indent=2,ensure_ascii=False))
        print(f'[SAVED] {output}',flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
