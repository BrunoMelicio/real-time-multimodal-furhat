"""Lazy ASR adapters; importing this module never loads weights or hardware."""
import importlib
from pathlib import Path
from queue import Queue
from time import perf_counter


class WhisperBackend:
    streaming = False

    def __init__(self, args):
        self.engine, self.size = args.engine, args.model
        self.info = {'engine':args.engine, 'model':args.model}
        if args.engine == 'faster':
            from faster_whisper import WhisperModel
            self.model = WhisperModel(args.model, device='cpu', compute_type='int8',
                                      cpu_threads=args.threads, num_workers=1)
            self.info.update(device='CPU', precision='INT8')
        elif args.engine == 'mlx':
            import mlx.core as mx
            self.module = importlib.import_module('mlx_whisper.transcribe')
            self.repo = args.model_repo or ('mlx-community/whisper-large-v3-turbo'
                         if args.model in ('turbo','large-v3-turbo') else
                         f'mlx-community/whisper-{args.model}-mlx')
            self.model = self.module.ModelHolder.get_model(self.repo, mx.float16)
            mx.eval(self.model.parameters())
            mx.synchronize()
            self.info.update(device='Apple GPU / MLX', precision='FP16', repository=self.repo)
        else:
            from pywhispercpp.model import Model
            import inspect
            context = {}
            if 'context_params' in inspect.signature(Model).parameters:
                context['context_params'] = {'use_gpu':args.cpp_device != 'cpu'}
            elif args.cpp_device == 'cpu':
                raise RuntimeError('Installed pywhispercpp lacks context_params; update it for explicit CPU selection')
            self.model = Model('large-v3-turbo' if args.model=='turbo' else args.model,
                               n_threads=args.threads, language='en',
                               print_realtime=False, print_progress=False,
                               print_timestamps=False, no_context=True,
                               **context)
            self.info.update(device='whisper.cpp backend: see native startup logs',
                             gpu_requested=args.cpp_device != 'cpu')

    def decode(self, audio):
        if self.engine == 'faster':
            segments, _ = self.model.transcribe(audio, language='en', beam_size=1,
                            temperature=0, vad_filter=False, condition_on_previous_text=False)
            return ' '.join(s.text.strip() for s in segments).strip()
        if self.engine == 'mlx':
            return self.module.transcribe(audio, path_or_hf_repo=self.repo, language='en',
                         temperature=0., condition_on_previous_text=False, verbose=None)['text'].strip()
        return ' '.join(s.text.strip() for s in self.model.transcribe(audio)).strip()

    def close(self):
        pass


class MoonshineBackend:
    streaming = True

    def __init__(self, args):
        from moonshine_voice import Transcriber, ModelArch, get_model_for_language
        arch = {'tiny':ModelArch.TINY_STREAMING, 'small':ModelArch.SMALL_STREAMING,
                'medium':ModelArch.MEDIUM_STREAMING}[args.model]
        path, actual_arch = get_model_for_language('en', arch)
        if actual_arch != arch:
            raise RuntimeError(f'Moonshine returned {actual_arch}, requested {arch}')
        self.model = Transcriber(path, actual_arch, update_interval=args.update_interval)
        self.events, self.lines, self.error, self.active = [], {}, None, False
        self.model.add_listener(self.event)
        self.info = dict(engine='moonshine', model=args.model, device='native CPU / ONNX runtime',
                         path=str(path), update_interval=args.update_interval,
                         finalization='common Silero endpoint forces stream stop',
                         weights_MB=sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())/1e6)

    def event(self, event):
        if getattr(event, 'error', None):
            self.error = event.error
            return
        line = getattr(event,'line',None)
        if line is None:
            return
        text = line.text.strip()
        key = getattr(line,'line_id',0)
        previous = self.lines.get(key)
        self.lines[key] = text
        if text and previous != text:
            self.events.append(dict(time=perf_counter(), text=text,
                                    complete=bool(getattr(line,'is_complete',False))))
            print(f'[PARTIAL] {text}', flush=True)

    def begin(self):
        self.lines, self.events, self.error = {}, [], None
        self.model.start()
        self.active = True

    def feed(self, audio):
        self.model.add_audio(audio.tolist(),16000)
        if self.error:
            raise self.error

    def finish(self):
        transcript = self.model.stop()
        self.active = False
        if self.error:
            raise self.error
        if transcript is not None:
            return ' '.join(line.text.strip() for line in transcript.lines if line.text.strip())
        return ' '.join(self.lines.values()).strip()

    def decode(self, audio):
        result = self.model.transcribe_without_streaming(audio.tolist(),16000)
        return ' '.join(line.text.strip() for line in result.lines if line.text.strip())

    def close(self):
        if self.active:
            self.finish()
        self.model.close()


class FurhatBackend:
    """SDK recognition. No local ASR loaded; local Silero only measures audio timing."""
    streaming = True

    def __init__(self, args):
        from furhat_realtime_api import FurhatClient
        self.args, self.results = args, Queue()
        self.events = []
        self.robot = FurhatClient(args.host)
        self.robot.connect()
        self.robot.request_listen_config(languages=['en-US'])
        for name in ['response.hear.start','response.hear.partial','response.hear.end',
                     'response.listen.end','response.listen.start']:
            self.robot.async_client.add_handler(name,self.event)
        self.info = dict(engine='furhat', model='SDK configured recognizer',
                         recognizer_label=args.furhat_recognizer,
                         device='SDK/provider; internal inference not exposed',
                         note='Python --mic monitors timing; set SDK microphone separately to same physical input')

    async def event(self, event):
        item = dict(time=perf_counter(), **event)
        self.events.append(item)
        kind = event.get('type')
        if kind == 'response.hear.partial' and event.get('text'):
            print('[PARTIAL]',event['text'],flush=True)
        if kind == 'response.hear.end':
            self.results.put(item)
        elif kind == 'response.listen.end' and event.get('cause') not in ('speech_end',):
            self.results.put(dict(item,text=''))

    def begin(self):
        self.events = []
        while not self.results.empty():
            self.results.get_nowait()
        self.robot._run_coroutine(self.robot.async_client.send_event_and_wait(dict(
            type='request.listen.start', partial=True, concat=True,
            stop_no_speech=True, stop_user_end=True, no_speech_timeout=15.,
            end_speech_timeout=self.args.silence), 'response.listen.start',timeout=5.))

    def close(self):
        try:
            self.robot._run_coroutine(self.robot.async_client.request_listen_stop())
        finally:
            self.robot.disconnect()
