"""One camera, two views, one face model and one small object detector."""
from pathlib import Path
from threading import Thread, Event, Lock
from time import perf_counter
import textwrap
import numpy as np

from furhat_interaction.paths import ROOT
OBJECT_MODELS = {
    'ssd': 'ssd_mobilenet_v2_float32.tflite',
    'lite0': 'efficientdet_lite0_int8.tflite',
    'lite2': 'efficientdet_lite2_int8.tflite',
    'yolo-n': 'yolo26n.pt', 'yolo-s': 'yolo26s.pt', 'yolo-m': 'yolo26m.pt',
}
OBJECT_LABELS = ['person', 'bottle', 'cup', 'book', 'cell phone']


class Camera:
    def __init__(self, cv2, index):
        self.cv2, self.index = cv2, index
        self.stop, self.lock = Event(), Lock()
        self.latest, self.error = None, None
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        cap = self.cv2.VideoCapture(self.index)
        try:
            if not cap.isOpened():
                raise RuntimeError('Camera could not be opened.')
            cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, 480)
            sequence = 0
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera stopped delivering frames.')
                sequence += 1
                with self.lock:
                    self.latest = sequence, perf_counter(), frame
        except Exception as exc:
            self.error = str(exc)
        finally:
            cap.release()

    def snapshot(self):
        with self.lock:
            return self.latest

    def close(self):
        self.stop.set()
        self.thread.join(2)


class Vision:
    def __init__(self, object_model='ssd', confidence=.3, device='auto', imgsz=640):
        import cv2
        import mediapipe as mp
        from furhat_interaction.head_gaze import head_rotation, head_angles, iris_offsets
        self.cv2, self.mp = cv2, mp
        self.rotation, self.angles, self.iris = head_rotation, head_angles, iris_offsets
        face = ROOT/'models/face_landmarker.task'
        objects = ROOT/'models'/OBJECT_MODELS[object_model]
        if object_model.startswith('yolo-') and not objects.exists():
            objects = ROOT/'models/object_compare'/OBJECT_MODELS[object_model]
        for path in (face, objects):
            if not path.exists():
                raise FileNotFoundError(f'Cached model missing: {path}')
        base = mp.tasks.BaseOptions
        video = mp.tasks.vision.RunningMode.VIDEO
        self.face = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(base_options=base(model_asset_path=str(face)),
                running_mode=video, num_faces=1, output_facial_transformation_matrixes=True))
        try:
            self.yolo = object_model.startswith('yolo-')
            if self.yolo:
                from types import SimpleNamespace
                from furhat_interaction.objects import Detector
                import torch
                torch.set_num_threads(2)
                self.detector = Detector(SimpleNamespace(method='yolo',size=object_model[-1],
                    device=device,imgsz=imgsz,confidence=confidence,offline=True))
                self.metadata = self.detector.metadata
            else:
                self.detector = mp.tasks.vision.ObjectDetector.create_from_options(
                    mp.tasks.vision.ObjectDetectorOptions(base_options=base(model_asset_path=str(objects)),
                        running_mode=video, max_results=8, score_threshold=confidence,
                        category_allowlist=OBJECT_LABELS))
                self.metadata = dict(method='mediapipe',size=object_model,device='cpu',checkpoint=str(objects))
            self.metadata['confidence'] = confidence
        except Exception:
            self.face.close()
            raise
        self.timestamp, self.next_objects, self.objects = -1, 0., []

    def analyze(self, frame, at):
        cv2, mp = self.cv2, self.mp
        h, w = frame.shape[:2]
        stamp = max(self.timestamp+1, int(at*1000)); self.timestamp = stamp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.face.detect_for_video(image, stamp)
        fresh = at >= self.next_objects
        if fresh:
            self.next_objects = at+.2  # Objects change slowly; face/head remain live.
            self.objects = []
            if self.yolo:
                for label, score, box in self.detector.predict(frame):
                    if label in OBJECT_LABELS:
                        self.objects.append(dict(label=label,score=score,box=box))
            else:
                detections = self.detector.detect_for_video(image, stamp)
                for item in detections.detections:
                    cat, b = item.categories[0], item.bounding_box
                    self.objects.append(dict(label=cat.category_name, score=float(cat.score),
                        box=(b.origin_x, b.origin_y, b.origin_x+b.width, b.origin_y+b.height)))
        points = box = crop = rotation = angles = uv = None
        eyes = []
        if result.face_landmarks:
            points = np.array([(p.x*w, p.y*h) for p in result.face_landmarks[0]])
            low, high = points.min(axis=0), points.max(axis=0)
            box = (max(0,int(low[0])), max(0,int(low[1])), min(w,int(high[0])+1), min(h,int(high[1])+1))
            x1,y1,x2,y2 = box
            if x2>x1 and y2>y1:
                crop = cv2.resize(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY),(112,112))
            if result.facial_transformation_matrixes:
                rotation = self.rotation(result.facial_transformation_matrixes[0])
                angles = self.angles(rotation)
            uv, eyes = self.iris(points)
        return dict(points=points, box=box, crop=crop, rotation=rotation, angles=angles,
                    uv=uv, eyes=eyes, objects=self.objects, fresh=fresh)

    def render(self, frame, observation, name, speaker, status, lines, mirror=True):
        cv2 = self.cv2
        clean = cv2.flip(frame,1) if mirror else frame.copy()
        h,w = frame.shape[:2]
        canvas = cv2.convertScaleAbs(clean, alpha=.22)
        def pt(p):
            return (int(w-1-p[0] if mirror else p[0]), int(p[1]))
        green, cyan = (90,240,90), (240,220,80)
        for item in observation['objects']:
            x1,y1,x2,y2 = item['box']
            a,b = pt((x1,y1)), pt((x2,y2))
            color = (green if speaker == (name or 'Person 1') else cyan) if item['label']=='person' else (80,190,255)
            cv2.rectangle(canvas,a,b,color,2)
            label = (name or 'Person 1') if item['label']=='person' else item['label']
            cv2.putText(canvas,f"{label} {item['score']:.0%}",(min(a[0],b[0]),max(18,y1-6)),0,.5,color,1)
        points = observation['points']
        if points is not None:
            for p in points:
                cv2.circle(canvas,pt(p),1,cyan,-1)
            r = observation['rotation']
            if r is not None:
                origin = points[1]
                endpoint = origin + np.array([r[0,2], -r[1,2]])*100
                cv2.arrowedLine(canvas,pt(origin),pt(endpoint),(255,150,80),2)
            uv = observation['uv']
            if uv is not None:
                for iris, horizontal, vertical in observation['eyes']:
                    tip = iris + (horizontal*uv[0]+vertical*uv[1])*220
                    cv2.arrowedLine(canvas,pt(iris),pt(tip),(0,230,255),2)
        panel = np.zeros((220,w,3),dtype=np.uint8)
        rows = [(f'Speaker: {speaker}',green), (status,cyan)]
        if observation['angles'] is not None:
            yaw,pitch,roll = observation['angles']
            rows.append((f'Head: yaw {yaw:+.0f} pitch {pitch:+.0f} | yellow: approximate iris gaze',cyan))
        for line in lines[-3:]:
            rows.extend((s,(240,240,240)) for s in textwrap.wrap(line, width=max(35,w//9))[:2])
        for index,(line,color) in enumerate(rows[-9:]):
            cv2.putText(panel,line,(10,22+index*23),0,.48,color,1,cv2.LINE_AA)
        return clean, np.vstack((canvas,panel))

    def close(self):
        self.face.close(); self.detector.close()
