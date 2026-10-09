"""One Face Landmarker supplies geometry, expressions and ASD face crops."""
from pathlib import Path
import numpy as np
from conference_ready_game_vision import GameVision,CONNECTIONS
from conference_ready_vision import Vision

ROOT=Path(__file__).resolve().parent


class ReflectionVision(GameVision):
    def __init__(self):
        import cv2
        import mediapipe as mp
        from live_head_gaze_compare import head_rotation,head_angles,iris_offsets
        path=ROOT/'models/face_landmarker.task'
        if not path.exists(): raise FileNotFoundError(f'Missing cached model: {path}')
        self.cv2,self.mp=cv2,mp
        self.rotation,self.angles,self.iris=head_rotation,head_angles,iris_offsets
        self.face=mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,num_faces=1,
                output_facial_transformation_matrixes=True,output_face_blendshapes=True))
        self.gesture=None
        self.hand=None
        self.timestamp=-1
        self.last_face=None

    def enable_hands(self):
        if self.hand is not None: return
        mp=self.mp;path=ROOT/'models/hand_landmarker.task'
        if not path.exists(): raise FileNotFoundError(f'Missing cached hand model: {path}')
        self.hand=mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,num_hands=2))

    def analyze(self,frame,at,hands_enabled=True):
        mp,cv2=self.mp,self.cv2
        h,w=frame.shape[:2]
        stamp=max(self.timestamp+1,int(at*1000));self.timestamp=stamp
        image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        result=self.face.detect_for_video(image,stamp)
        row=dict(points=None,box=None,rotation=None,angles=None,uv=None,eyes=[],objects=[],hands=[],scores={},crop=None)
        if result.face_landmarks:
            points=np.array([(p.x*w,p.y*h) for p in result.face_landmarks[0]])
            low,high=points.min(axis=0),points.max(axis=0)
            box=(max(0,int(low[0])),max(0,int(low[1])),min(w,int(high[0])+1),min(h,int(high[1])+1))
            row.update(points=points,box=box)
            self.last_face=(at,box)
            x1,y1,x2,y2=box
            if x2>x1 and y2>y1:
                row['crop']=cv2.resize(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY),(112,112))
            if result.facial_transformation_matrixes:
                r=self.rotation(result.facial_transformation_matrixes[0]);row.update(rotation=r,angles=self.angles(r))
            row['uv'],row['eyes']=self.iris(points)
            if result.face_blendshapes:
                row['scores']={c.category_name:float(c.score) for c in result.face_blendshapes[0]}
        row['motion_box']=row['box']
        if row['motion_box'] is None and self.last_face is not None and at-self.last_face[0]<=.5:
            row['motion_box']=self.last_face[1]  # Brief hand occlusion, not a persistent identity.
        if hands_enabled and self.hand is not None:
            hands=self.hand.detect_for_video(image,stamp)
            row['hands']=[dict(points=np.array([(p.x*w,p.y*h) for p in landmarks])) for landmarks in hands.hand_landmarks]
        return row

    def render(self,frame,observation,name,speaker,status,lines,mirror=True):
        clean,canvas=Vision.render(self,frame,observation,name,speaker,status,lines,mirror)
        cv2,w=self.cv2,frame.shape[1]
        def pt(p): return int(w-1-p[0] if mirror else p[0]),int(p[1])
        for hand in observation['hands']:
            points=hand['points']
            for a,b in CONNECTIONS: cv2.line(canvas,pt(points[a]),pt(points[b]),(90,240,90),2)
            for p in points: cv2.circle(canvas,pt(p),3,(60,190,255),-1)
        if observation['hands'] and observation['box']:
            x1,y1,x2,y2=observation['box'];dx,dy=(x2-x1)*.3,(y2-y1)*.3
            cv2.rectangle(canvas,pt((x1-dx,y1-dy)),pt((x2+dx,y2+dy)),(80,160,240),1)
        return clean,canvas

    def close(self):
        try:
            if self.hand is not None: self.hand.close()
        finally: self.face.close()
