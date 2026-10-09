"""Part 2 continuous face/hand geometry and head movement display."""
from collections import deque
from pathlib import Path
import numpy as np
from conference_ready_vision import Vision
from conference_ready_game import HeadAnswer

ROOT = Path(__file__).resolve().parent
CONNECTIONS = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
               (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),
               (0,17),(17,18),(18,19),(19,20))


class GameVision:
    def __init__(self):
        import cv2
        import mediapipe as mp
        from live_head_gaze_compare import load_model, head_rotation, head_angles, iris_offsets
        if not (ROOT/'models/face_landmarker.task').exists():
            raise FileNotFoundError('Cached models/face_landmarker.task is missing.')
        self.cv2,self.mp = cv2,mp
        self.rotation,self.angles,self.iris = head_rotation,head_angles,iris_offsets
        self.face = load_model('both',1)
        self.gesture = None
        self.timestamp = -1
        self.head_trace=deque(maxlen=900)
        self.head_events=deque(maxlen=30)
        self.motion_detector=HeadAnswer()
        self.next_motion=0.

    def mark_head(self,answer,at):
        label='NOD / YES' if answer else 'HEAD SHAKE / NO'
        if not self.head_events or at-self.head_events[-1][0]>.5:
            self.head_events.append((at,label))

    def update_head_display(self,angles,at):
        self.head_trace.append((at,None if angles is None else tuple(map(float,angles))))
        while self.head_trace and at-self.head_trace[0][0]>8.: self.head_trace.popleft()
        answer=self.motion_detector.update(angles,at)
        if answer is not None and at>=self.next_motion:
            self.mark_head(answer,at);self.next_motion=at+1.2

    def enable_hands(self):
        if self.gesture is not None:
            return
        path = ROOT/'models/gesture_recognizer.task'
        if not path.exists():
            raise FileNotFoundError(f'Cached hand model missing: {path}')
        mp = self.mp
        self.gesture = mp.tasks.vision.GestureRecognizer.create_from_options(
            mp.tasks.vision.GestureRecognizerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,num_hands=2))

    def analyze(self, frame, at, hands_enabled=True):
        mp,cv2 = self.mp,self.cv2
        h,w = frame.shape[:2]
        stamp = max(self.timestamp+1,int(at*1000)); self.timestamp = stamp
        image = mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        result = self.face.detect_for_video(image,stamp)
        row = dict(points=None,box=None,rotation=None,angles=None,uv=None,eyes=[],objects=[],hands=[])
        if result.face_landmarks:
            points = np.array([(p.x*w,p.y*h) for p in result.face_landmarks[0]])
            low,high = points.min(axis=0),points.max(axis=0)
            row.update(points=points,box=(max(0,int(low[0])),max(0,int(low[1])),min(w,int(high[0])+1),min(h,int(high[1])+1)))
            if result.facial_transformation_matrixes:
                r = self.rotation(result.facial_transformation_matrixes[0])
                row.update(rotation=r,angles=self.angles(r))
            row['uv'],row['eyes'] = self.iris(points)
        if hands_enabled and self.gesture is not None:
            result = self.gesture.recognize_for_video(image,stamp)
            for index,points in enumerate(result.hand_landmarks):
                categories = result.gestures[index] if index<len(result.gestures) else []
                best = max(categories,key=lambda c:c.score) if categories else None
                row['hands'].append(dict(label=best.category_name if best else 'None',
                    score=float(best.score) if best else 0.,points=np.array([(p.x*w,p.y*h) for p in points])))
        self.update_head_display(row['angles'],at)
        return row

    def render(self, frame, observation, name, speaker, status, lines, mirror=True):
        clean,canvas = Vision.render(self,frame,observation,name,speaker,status,lines,mirror)
        cv2,w = self.cv2,frame.shape[1]
        def pt(p):
            return int(w-1-p[0] if mirror else p[0]),int(p[1])
        if observation['points'] is not None and observation['rotation'] is not None:
            origin=observation['points'][1]
            r=observation['rotation']
            for axis,label,color in ((0,'X',(70,70,255)),(1,'Y',(80,240,80)),(2,'Z',(255,160,80))):
                tip=origin+np.array([r[0,axis],-r[1,axis]])*65
                cv2.arrowedLine(canvas,pt(origin),pt(tip),color,2)
                cv2.putText(canvas,label,pt(tip),0,.45,color,1)
        for hand in observation['hands']:
            points = hand['points']
            for a,b in CONNECTIONS:
                cv2.line(canvas,pt(points[a]),pt(points[b]),(90,240,90),2)
            for p in points:
                cv2.circle(canvas,pt(p),3,(60,190,255),-1)
            x,y = pt(points[0])
            cv2.putText(canvas,f"{hand['label']} {hand['score']:.0%}",(max(0,x-50),max(20,y+20)),0,.5,(240,240,240),1)
        return clean,np.vstack((canvas,self.head_panel(w)))

    def head_panel(self,w):
        """Eight seconds of measured angles; plotting adds no neural model."""
        cv2=self.cv2;panel=np.zeros((165,w,3),dtype=np.uint8)
        cv2.putText(panel,'Head movement',(10,21),0,.6,(240,240,240),1,cv2.LINE_AA)
        if not self.head_trace: return panel
        now=self.head_trace[-1][0];left,right=105,w-12
        def x(at): return int(left+(at-(now-8.))/8.*(right-left))
        for axis,label,color,center in ((0,'Yaw',(255,190,70),65),(1,'Pitch',(80,220,100),123)):
            cv2.putText(panel,label,(10,center+5),0,.48,color,1)
            cv2.line(panel,(left,center),(right,center),(55,55,55),1)
            previous=None
            for at,angles in self.head_trace:
                if angles is None: previous=None;continue
                point=(x(at),int(center-np.clip(angles[axis],-35,35)/35.*23))
                if previous is not None: cv2.line(panel,previous,point,color,1,cv2.LINE_AA)
                previous=point
        for at,label in self.head_events:
            if now-8.<=at<=now:
                color=(80,240,100) if label.startswith('NOD') else (80,180,255)
                cv2.line(panel,(x(at),35),(x(at),148),color,1)
        latest=self.head_events[-1] if self.head_events else None
        label=latest[1]+' detected' if latest and now-latest[0]<2.5 else 'No nod / shake detected'
        if self.head_trace[-1][1] is None: label='Face not visible'
        cv2.putText(panel,label,(170,21),0,.48,(100,240,220),1,cv2.LINE_AA)
        cv2.putText(panel,'8s ago                                     now | angles +/-35 deg',(105,160),0,.35,(170,170,170),1)
        return panel

    def close(self):
        self.face.close()
        if self.gesture is not None:
            self.gesture.close()
