"""Live object detection comparison: one model, camera only, natural speed.

Examples:
  .venv/bin/python furhat_interaction.objects.py --method=yolo --size=n
  .venv/bin/python furhat_interaction.objects.py --method=mediapipe --size=lite0
  .venv/bin/python furhat_interaction.objects.py --method=deim --size=n
"""
import argparse
import csv
import json
import platform
from collections import deque
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from time import perf_counter, monotonic_ns
from urllib.request import urlopen

from furhat_interaction.paths import ROOT
CACHE = ROOT / 'models' / 'object_compare'
DEIM_MODELS = {
    'n': 'harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers',
    's': 'harshaljanjani/DEIMv2_DINOv3_S_COCO_Transformers',
}
MP_NAMES = {'ssd': 'ssd_mobilenet_v2', 'lite0': 'efficientdet_lite0', 'lite2': 'efficientdet_lite2'}


def positive(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError('Must be positive.')
    return value


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', choices=['yolo','mediapipe','deim'], default='yolo')
    p.add_argument('--size', help='YOLO: n/s/m/l/x; MediaPipe: ssd/lite0/lite2; DEIMv2: n/s')
    p.add_argument('--precision', choices=['int8','float16','float32'], default=None,
                   help='MediaPipe checkpoint precision. Default: int8 EfficientDet, float32 SSD. Others use float32.')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--width', type=positive, default=640, help='Processing/preview width, preserving aspect ratio.')
    p.add_argument('--imgsz', type=positive, default=640, help='YOLO network size. MediaPipe/DEIM retain checkpoint-native input sizes.')
    p.add_argument('--confidence', type=float, default=.3)
    p.add_argument('--device', choices=['auto','cpu','mps'], default='auto', help='YOLO/DEIM: auto selects MPS. MediaPipe uses CPU.')
    p.add_argument('--mirror', action='store_true', help='Selfie orientation; default is unmirrored.')
    p.add_argument('--seconds', type=float, default=0, help='0: run until Q/Esc; otherwise stop after this duration.')
    p.add_argument('--offline', action='store_true', help='Use cached weights only; fail rather than downloading.')
    a = p.parse_args(argv)
    allowed = {'yolo':('n','s','m','l','x'), 'mediapipe':('ssd','lite0','lite2'), 'deim':('n','s')}
    a.size = a.size or {'yolo':'n','mediapipe':'lite0','deim':'n'}[a.method]
    if a.size not in allowed[a.method]:
        p.error(f'{a.method} sizes: {", ".join(allowed[a.method])}')
    if not 0 < a.confidence < 1 or a.seconds < 0:
        p.error('confidence must be between 0 and 1; seconds must be nonnegative.')
    if a.method == 'mediapipe':
        a.precision = a.precision or ('float32' if a.size == 'ssd' else 'int8')
        if a.size == 'ssd' and a.precision == 'float16':
            p.error('SSD MobileNet V2 official options: int8 or float32.')
        if a.device == 'mps':
            p.error('MediaPipe runs on CPU here; use --device=auto or cpu.')
    elif a.precision not in (None,'float32'):
        p.error('--precision is for MediaPipe; YOLO/DEIM use float32 in this comparison.')
    return a


def download(url, path, offline=False):
    if path.is_file():
        return path
    if offline:
        raise FileNotFoundError(f'Checkpoint not cached: {path}. Run once without --offline.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.download')
    print(f'[DOWNLOAD] {url}\n[CACHE] {path}', flush=True)
    try:
        with urlopen(url, timeout=120) as response, temp.open('wb') as out:
            total = int(response.headers.get('Content-Length',0))
            received = report_at = 0
            while True:
                chunk = response.read(1024*1024)
                if not chunk:
                    break
                out.write(chunk)
                received += len(chunk)
                if received >= report_at:
                    print(f'[DOWNLOAD] {received/1e6:.1f} MB' + (f' / {total/1e6:.1f} MB' if total else ''),flush=True)
                    report_at = received + 8*1024*1024
        with temp.open('rb') as stream:
            if stream.read(8)[4:8] != b'TFL3':
                raise RuntimeError('Downloaded file is not a TFLite checkpoint.')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def normalize_detections(rows, width, height):
    """Common output format, clipped to the original processing frame."""
    from math import isfinite
    result = []
    for label,score,box in rows:
        coords = [float(v) for v in box]
        if not isfinite(float(score)) or not all(isfinite(v) for v in coords):
            continue
        x1,y1,x2,y2 = coords
        box = (max(0,min(width,round(x1))),max(0,min(height,round(y1))),
               max(0,min(width,round(x2))),max(0,min(height,round(y2))))
        if box[2] > box[0] and box[3] > box[1]:
            result.append((str(label),float(score),box))
    return result


class Detector:
    def __init__(self, a):
        self.a = a
        self.close = lambda: None
        self.metadata = {'method':a.method,'size':a.size,'device':'cpu'}
        if a.method == 'mediapipe':
            import mediapipe as mp
            self.mp = mp
            name = MP_NAMES[a.size]
            filename = f'{name}_{a.precision}.tflite'
            # Reuse the previous demos' cached models if available.
            path = ROOT/'models'/filename
            url = f'https://storage.googleapis.com/mediapipe-models/object_detector/{name}/{a.precision}/latest/{name}.tflite'
            download(url,path,a.offline)
            with path.open('rb') as stream:
                if stream.read(8)[4:8] != b'TFL3':
                    raise RuntimeError(f'Invalid cached TFLite model: {path}')
            options = mp.tasks.vision.ObjectDetectorOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(path), delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                score_threshold=a.confidence,max_results=-1)
            self.model = mp.tasks.vision.ObjectDetector.create_from_options(options)
            self.close = self.model.close
            self.timestamp = -1
            self.metadata.update(network_input={'ssd':256,'lite0':320,'lite2':448}[a.size],precision=a.precision)
        else:
            import torch
            self.torch = torch
            self.device = ('mps' if torch.backends.mps.is_available() else 'cpu') if a.device == 'auto' else a.device
            if self.device == 'mps' and not torch.backends.mps.is_available():
                raise RuntimeError('MPS is unavailable; try --device=cpu.')
            self.metadata.update(device=self.device,precision='float32')
            if a.method == 'yolo':
                from ultralytics import YOLO
                name = f'yolo26{a.size}.pt'
                candidates = [CACHE/name, ROOT/'models'/name, ROOT/name]
                path = next((p for p in candidates if p.is_file()), candidates[0])
                if a.offline and not path.is_file():
                    raise FileNotFoundError(f'YOLO checkpoint not cached: {path}')
                path.parent.mkdir(parents=True,exist_ok=True)
                self.model = YOLO(str(path))
                params = sum(p.numel() for p in self.model.model.parameters())
                self.metadata.update(parameters=params,network_input=f'YOLO imgsz={a.imgsz}')
            else:
                try:
                    from transformers import AutoImageProcessor, AutoModelForObjectDetection, Deimv2Config
                except ImportError as exc:
                    raise RuntimeError('DEIMv2 requires Transformers 5.19.0. Install with: .venv/bin/python -m pip install "transformers==5.19.0"') from exc
                model_id = DEIM_MODELS[a.size]
                path = CACHE/'deim_hf'
                kw = dict(cache_dir=str(path),local_files_only=a.offline)
                print(f'[DEIM] {model_id} | complete checkpoint; no separate backbone download',flush=True)
                self.processor = AutoImageProcessor.from_pretrained(model_id,**kw)
                self.model = AutoModelForObjectDetection.from_pretrained(model_id,use_safetensors=True,**kw)
                self.model.eval().to(self.device)
                self.metadata.update(model_id=model_id,parameters=sum(p.numel() for p in self.model.parameters()),
                                     network_input=str(self.processor.size),processor=type(self.processor).__name__)
        self.metadata['checkpoint'] = str(path)
        if path.is_file():
            self.metadata['checkpoint_mb'] = round(path.stat().st_size/1e6,2)
        elif a.method == 'deim':
            # Counts the safetensors file for this model, excluding other cached variants.
            model_cache = path/('models--'+model_id.replace('/','--'))
            snapshots = list(model_cache.glob('snapshots/*/model.safetensors'))
            if snapshots:
                self.metadata['checkpoint_mb'] = round(snapshots[-1].stat().st_size/1e6,2)

    def predict(self, frame):
        import cv2
        if self.a.method == 'mediapipe':
            self.timestamp = max(self.timestamp+1,monotonic_ns()//1_000_000)
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            result = self.model.detect_for_video(image,self.timestamp)
            rows = []
            for detection in result.detections:
                if not detection.categories:
                    continue
                c = max(detection.categories,key=lambda c:c.score)
                b = detection.bounding_box
                rows.append((c.category_name or c.display_name or str(c.index), c.score,
                             (b.origin_x,b.origin_y,b.origin_x+b.width,b.origin_y+b.height)))
        elif self.a.method == 'yolo':
            result = self.model.predict(frame,device=self.device,imgsz=self.a.imgsz,conf=self.a.confidence,verbose=False)[0]
            # CPU conversion completes GPU work before this timed call returns.
            boxes = result.boxes.xyxy.cpu().tolist()
            labels = result.boxes.cls.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            rows = [(result.names[int(label)],score,box) for label,score,box in zip(labels,scores,boxes)]
        else:
            with self.torch.inference_mode():
                inputs = self.processor(images=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),return_tensors='pt').to(self.device)
                outputs = self.model(**inputs)
                result = self.processor.post_process_object_detection(outputs,threshold=self.a.confidence,
                            target_sizes=[frame.shape[:2]])[0]
                boxes = result['boxes'].cpu().tolist()
                labels = result['labels'].cpu().tolist()
                scores = result['scores'].cpu().tolist()
                rows = [(self.model.config.id2label[int(label)],score,box) for label,score,box in zip(labels,scores,boxes)]
        return normalize_detections(rows,frame.shape[1],frame.shape[0])


def main(argv=None):
    a = arguments(argv)
    import cv2
    import numpy as np
    output = ROOT/'vision_output'/'object_compare'/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+f'_{a.method}_{a.size}')
    output.mkdir(parents=True,exist_ok=True)
    print(f'[OUTPUT] {output}\n[LOAD] Only the selected detector is loaded. First run may download weights.',flush=True)
    metadata = {'arguments':vars(a),'platform':platform.platform(),'packages':{}}
    for name in ('mediapipe','ultralytics','torch','torchvision','transformers','opencv-python'):
        try: metadata['packages'][name] = version(name)
        except PackageNotFoundError: pass
    detector = cap = None
    samples = []
    try:
        started = perf_counter()
        detector = Detector(a)
        metadata.update(detector.metadata,load_seconds=perf_counter()-started)
        print('[MODEL] '+json.dumps(metadata),flush=True)
        cap = cv2.VideoCapture(a.camera)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera {a.camera}; check camera permissions.')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,round(a.width*3/4))
        print('[LIVE] Q/Esc exits. First 5 frames excluded from summary. No FPS cap or tracking.',flush=True)
        recent = deque(maxlen=60)
        start = last_log = perf_counter()
        with (output/'frames.csv').open('w',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['frame','warmup','detections','capture_ms','resize_ms','detection_pipeline_ms','draw_display_ms','total_ms'])
            index = 0
            while not a.seconds or perf_counter()-start < a.seconds:
                t0 = perf_counter()
                ok,frame = cap.read()
                t1 = perf_counter()
                if not ok:
                    raise RuntimeError('Camera stopped returning frames.')
                h,w = frame.shape[:2]
                if w != a.width:
                    frame = cv2.resize(frame,(a.width,round(h*a.width/w)))
                if a.mirror: frame = cv2.flip(frame,1)
                t2 = perf_counter()
                detections = detector.predict(frame)
                t3 = perf_counter()
                for label,score,(x1,y1,x2,y2) in detections:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(60,220,70),2)
                    cv2.putText(frame,f'{label} {score:.0%}',(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(60,220,70),1,cv2.LINE_AA)
                index += 1
                if index == 1:
                    print(f'[WARMUP] First detection call: {(t3-t2)*1000:.1f}ms',flush=True)
                fps = 1000/np.mean([row[-1] for row in recent]) if recent else 0
                cv2.rectangle(frame,(0,0),(min(640,frame.shape[1]),62),(25,25,25),-1)
                cv2.putText(frame,f'{a.method} {a.size} | {len(detections)} objects | {fps:.1f} FPS',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(frame,f'Detection pipeline {(t3-t2)*1000:.1f} ms | Q/Esc exits',(8,49),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
                cv2.imshow('Live object detection comparison',frame)
                key = cv2.waitKey(1)&0xff
                t4 = perf_counter()
                timings = [(t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000,(t4-t3)*1000,(t4-t0)*1000]
                writer.writerow([index,index<=5,len(detections),*[round(t,3) for t in timings]])
                if index > 5:
                    samples.append(timings)
                    recent.append(timings)
                if t4-last_log >= 2 and recent:
                    means = np.mean(recent,axis=0)
                    labels = ', '.join(sorted({row[0] for row in detections})) or 'none'
                    print(f'[TIMING] FPS={1000/means[-1]:.2f} | detection={means[2]:.1f}ms | capture={means[0]:.1f}ms | resize={means[1]:.1f}ms | draw/display={means[3]:.1f}ms | labels={labels}',flush=True)
                    stream.flush()
                    last_log = t4
                if key in (27,ord('q')): break
    except KeyboardInterrupt:
        print('\nStopped.')
    except Exception as exc:
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if cap is not None: cap.release()
        if detector is not None: detector.close()
        cv2.destroyAllWindows()
        if samples:
            data = np.array(samples)
            metadata['summary'] = {'frames_after_warmup':len(samples),'fps':1000/float(data[:,-1].mean()),
                'detection_pipeline_mean_ms':float(data[:,2].mean()),'detection_pipeline_p50_ms':float(np.median(data[:,2])),
                'detection_pipeline_p95_ms':float(np.percentile(data[:,2],95)),
                'note':'Full detector call: preprocessing, inference, decoding, GPU completion/result transfer. FPS includes camera/GUI, not sensor-to-display latency. Precision/backends differ; no ground-truth accuracy measured.'}
            print('[SUMMARY] '+json.dumps(metadata['summary']),flush=True)
        (output/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(f'[SAVED] {output}',flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}',flush=True)
        raise SystemExit(1)
