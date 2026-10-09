"""Two tracked people; continuous face geometry and part-specific perception."""
from collections import deque
from pathlib import Path
import textwrap
from time import perf_counter
import numpy as np
from .policies import Seats,object_owner,HeadMotion
from conference_ready_reflection import CueHistory,HandNearFace,HAND_CUE
from .cues import ReflectionCues

ROOT=Path(__file__).resolve().parents[1]


def screen_points(points,width,mirror):
    """Batch the same pixel conversion used by the overlay's scalar pt()."""
    xy=np.array(points,copy=True)
    if mirror: xy[:,0]=width-1-xy[:,0]
    return xy.astype(np.int32)


def head_trace(history,at,left,right,base):
    rows=np.asarray([(t,*angles[:2]) for t,angles in history if t>=at-6.],dtype=float)
    if not len(rows): return []
    x=(left+(rows[:,0]-at+6)/6.*(right-left)).astype(np.int32)
    y=(base-np.clip(rows[:,1:],-35,35)).astype(np.int32)
    return [np.column_stack((x,y[:,axis])) for axis in (0,1)]


class Vision:
    def __init__(self,part,object_model='ssd'):
        import os
        os.environ['YOLO_AUTOINSTALL']='false'
        import cv2,mediapipe as mp,torch
        from ultralytics import YOLO
        from live_head_gaze_compare import head_rotation,head_angles,iris_offsets
        self.cv2,self.mp=cv2,mp
        torch.set_num_threads(2);cv2.setNumThreads(1)
        self.rotation,self.angles,self.iris=head_rotation,head_angles,iris_offsets
        self.part=part;self.seats=Seats();self.face=self.pose=self.gesture=self.objects=None
        self.profile=False;self.timings={}
        self.people=[];self.detect_at=-float('inf');self.stamp=-1
        self.cues={i:(ReflectionCues() if part==3 else CueHistory()) for i in (1,2)}
        self.motion={i:HandNearFace() for i in (1,2)}
        self.head={i:HeadMotion() for i in (1,2)}
        self.head_signal={i:deque(maxlen=300) for i in (1,2)}
        self.head_origin={}
        self.head_event={};self.head_cooldown={1:0.,2:0.}
        self.object_history=deque(maxlen=300)
        def path(name):
            p=ROOT/'models'/name
            if not p.exists(): raise FileNotFoundError(f'Missing cached model: {p}; no automatic downloads.')
            return str(p)
        base=lambda name:mp.tasks.BaseOptions(model_asset_path=path(name))
        video=mp.tasks.vision.RunningMode.VIDEO
        try:
            self.detector=YOLO(path('yolo26n.pt'))
            self.face=mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base('face_landmarker.task'),running_mode=video,num_faces=2,
                output_facial_transformation_matrixes=True,output_face_blendshapes=True))
            if part in (2,3):
                self.pose=mp.tasks.vision.PoseLandmarker.create_from_options(mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=base('pose_landmarker_lite.task'),running_mode=video,num_poses=2,
                    output_segmentation_masks=False))
                if part==3:
                    # Use the EXACT hand detector/landmark assets already inside
                    # our cached gesture task, without its unused gesture classifier.
                    from zipfile import ZipFile
                    with ZipFile(path('gesture_recognizer.task')) as bundle:
                        hand_asset=bundle.read('hand_landmarker.task')
                    self.gesture=mp.tasks.vision.HandLandmarker.create_from_options(mp.tasks.vision.HandLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(model_asset_buffer=hand_asset),running_mode=video,num_hands=4))
                else:
                    self.gesture=mp.tasks.vision.GestureRecognizer.create_from_options(mp.tasks.vision.GestureRecognizerOptions(
                        base_options=base('gesture_recognizer.task'),running_mode=video,num_hands=4))
            if part==1:
                from conference_ready_vision import OBJECT_MODELS
                self.objects=mp.tasks.vision.ObjectDetector.create_from_options(mp.tasks.vision.ObjectDetectorOptions(
                    base_options=base(OBJECT_MODELS[object_model]),running_mode=video,max_results=8,score_threshold=.3,
                    category_allowlist=['bottle','cup','book','cell phone']))
        except BaseException:
            self.close();raise

    def observe(self,frame,at):
        from furhat_interaction.detection import observations
        from people_awareness import associate_faces
        from group_interaction.hand_ownership import body_owners,assign_hands
        cv2,mp=self.cv2,self.mp;h,w=frame.shape[:2]
        timings={};tick=perf_counter() if self.profile else 0.
        if at-self.detect_at>=.2:
            result=self.detector.track(frame,classes=[0],persist=True,tracker='bytetrack.yaml',
                imgsz=320,device='cpu',conf=.35,max_det=2,verbose=False)[0]
            self.people=self.seats.update(observations(result),at,w);self.detect_at=at
        if self.profile:
            end=perf_counter();timings['people_ms']=(end-tick)*1000;tick=end
        self.stamp=max(self.stamp+1,int(at*1000))
        image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        result=self.face.detect_for_video(image,self.stamp)
        if self.profile:
            end=perf_counter();timings['face_ms']=(end-tick)*1000;tick=end
        points=[np.array([(p.x*w,p.y*h) for p in face]) for face in result.face_landmarks]
        association=associate_faces(self.people,[tuple(p[1]) for p in points])
        faces={};crops={};events={}
        for slot,index in association.items():
            p=points[index];lo,hi=p.min(axis=0),p.max(axis=0)
            box=(max(0,int(lo[0])),max(0,int(lo[1])),min(w,int(hi[0])+1),min(h,int(hi[1])+1))
            r=self.rotation(result.facial_transformation_matrixes[index])
            angles=self.angles(r);uv,eyes=self.iris(p)
            scores={c.category_name:float(c.score) for c in result.face_blendshapes[index]}
            if self.cues[slot].baseline is None: self.cues[slot].calibrate(angles,uv,at)
            cue=self.cues[slot].add(angles,uv,scores,at)
            answer=self.head[slot].update(angles,at)
            if answer is not None and at>=self.head_cooldown[slot]:
                self.head_event[slot]=(at,answer);events[slot]=answer;self.head_cooldown[slot]=at+1.2
            self.head_origin.setdefault(slot,angles)
            relative=tuple((angle-origin+180.)%360.-180. for angle,origin in zip(angles,self.head_origin[slot]))
            self.head_signal[slot].append((at,relative))
            faces[slot]=dict(points=p,box=box,rotation=r,angles=angles,uv=uv,eyes=eyes,cues=cue)
            x1,y1,x2,y2=box
            if x2>x1 and y2>y1: crops[slot]=cv2.resize(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY),(112,112))
        for slot in (1,2):
            if slot not in faces: self.head[slot].update(None,at)
        poses=[];hands=[];motion=[]
        if self.profile:
            end=perf_counter();timings['geometry_ms']=(end-tick)*1000;tick=end
        if self.pose:
            poses=self.pose.detect_for_video(image,self.stamp).pose_landmarks
            if self.profile:
                end=perf_counter();timings['body_ms']=(end-tick)*1000;tick=end
            result=(self.gesture.detect_for_video(image,self.stamp) if self.part==3
                    else self.gesture.recognize_for_video(image,self.stamp))
            if self.profile:
                end=perf_counter();timings['hands_ms']=(end-tick)*1000;tick=end
            raw=[]
            for i,p in enumerate(result.hand_landmarks):
                xy=np.array([(v.x*w,v.y*h) for v in p]);cats=result.gestures[i] if self.part==2 else []
                best=max(cats,key=lambda c:c.score) if cats else None
                raw.append(dict(points=xy,wrist=tuple(map(float,xy[0])),label=best.category_name if best else ('Hand' if self.part==3 else 'None'),
                                score=float(best.score) if best else 0.))
            hands=assign_hands(raw,body_owners(poses,self.people,w,h))
            for slot in (1,2):
                f=faces.get(slot)
                if self.motion[slot].update(f['box'] if f else None,[p for p in hands if p['person_id']==slot],at): motion.append(slot)
        objects=[]
        if self.objects:
            result=self.objects.detect_for_video(image,self.stamp)
            for det in result.detections:
                cat=max(det.categories,key=lambda c:c.score);b=det.bounding_box
                box=(b.origin_x,b.origin_y,b.origin_x+b.width,b.origin_y+b.height)
                objects.append(dict(label=cat.category_name,score=float(cat.score),box=box,owner=object_owner(box,self.people)))
            self.object_history.append((at,objects))
        if self.profile:
            timings['ownership_objects_ms']=(perf_counter()-tick)*1000
            self.timings=timings
        return dict(people=self.people,faces=faces,crops=crops,hands=hands,poses=poses,objects=objects,
                    head_events=events,motion=motion,at=at,width=w,height=h)

    def object_for(self,slot,at):
        from collections import Counter
        counts=Counter(o['label'] for t,rows in self.object_history if at-12.<=t<=at
                       for o in rows if o['owner']==slot)
        eligible=[label for label,count in counts.items() if count>=3]
        return next((x for x in ('bottle','cup','book','cell phone') if x in eligible),None)

    def render(self,frame,obs,names,active,focus,status,lines,mirror=True):
        from conference_ready_game_vision import CONNECTIONS
        cv2=self.cv2;h,w=frame.shape[:2]
        clean=cv2.flip(frame,1) if mirror else frame.copy();canvas=cv2.convertScaleAbs(clean,alpha=.22)
        def pt(p): return int(w-1-p[0] if mirror else p[0]),int(p[1])
        for person in obs['people']:
            slot=person['id'];x1,y1,x2,y2=person['box'];color=(80,240,80) if slot==active else (240,200,80)
            cv2.rectangle(canvas,pt((x1,y1)),pt((x2,y2)),color,2)
            cv2.putText(canvas,f"{names.get(slot,'Person '+str(slot))} | ID {slot}"+(' SPEAKING' if slot==active else ''),
                        (min(pt((x1,y1))[0],pt((x2,y1))[0]),max(20,int(y1)-8)),0,.48,color,1)
        for slot,f in obs['faces'].items():
            if self.part==3:
                # A radius-one filled circle is this five-pixel cross. Draw all
                # landmarks together instead of ~956 Python->OpenCV calls/frame.
                xy=screen_points(f['points'],w,mirror)
                for dx,dy in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
                    x,y=xy[:,0]+dx,xy[:,1]+dy
                    valid=(x>=0)&(x<w)&(y>=0)&(y<h)
                    canvas[y[valid],x[valid]]=(200,170,80)
            else:
                for p in f['points']: cv2.circle(canvas,pt(p),1,(200,170,80),-1)
            r=f['rotation'];origin=f['points'][1]
            for axis,color in ((0,(80,80,255)),(1,(80,240,80)),(2,(255,160,80))):
                cv2.arrowedLine(canvas,pt(origin),pt(origin+np.array([r[0,axis],-r[1,axis]])*60),color,2)
            if f['uv'] is not None:
                for iris,horizontal,vertical in f['eyes']:
                    cv2.arrowedLine(canvas,pt(iris),pt(iris+(horizontal*f['uv'][0]+vertical*f['uv'][1])*220),(0,230,255),1)
        for pose in obs['poses']:
            for a,b in ((11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24)):
                if min(pose[a].visibility,pose[b].visibility)>.5:
                    cv2.line(canvas,pt((pose[a].x*w,pose[a].y*h)),pt((pose[b].x*w,pose[b].y*h)),(100,150,100),1)
        for hand in obs['hands']:
            color=(80,240,80) if hand['person_id'] else (60,60,255)
            for a,b in CONNECTIONS: cv2.line(canvas,pt(hand['points'][a]),pt(hand['points'][b]),color,1)
            for p in hand['points']: cv2.circle(canvas,pt(p),2,color,-1)
            owner=names.get(hand['person_id'],'Unassigned')
            cv2.putText(canvas,f"{owner}: {hand['label']}",pt(hand['points'][0]),0,.4,color,1)
        for o in obs['objects']:
            x1,y1,x2,y2=o['box'];cv2.rectangle(canvas,pt((x1,y1)),pt((x2,y2)),(80,190,255),2)
            cv2.putText(canvas,o['label'],pt((x1,y1)),0,.48,(80,190,255),1)
        panel=np.zeros((370 if self.part==3 else 330,w,3),dtype=np.uint8)
        rows=[status,f"Speaking: {names.get(active,'uncertain / none')} | Attention: {names.get(focus,'none')}"]
        for line in lines[-3:]: rows.extend(textwrap.wrap(line,width=max(35,w//8))[:2])
        for i,line in enumerate(rows[:8]): cv2.putText(panel,line,(10,20+i*21),0,.45,(230,230,230),1,cv2.LINE_AA)
        cv2.putText(panel,'Head movement | yaw: blue | pitch: green',(10,197),0,.46,(240,240,240),1)
        for slot in (1,2):
            left=(slot-1)*(w//2)+8;right=left+w//2-16;base=250
            cv2.putText(panel,names.get(slot,f'Person {slot}'),(left,219),0,.45,(240,240,240),1)
            if self.part==3: paths=head_trace(self.head_signal[slot],obs['at'],left,right,base)
            for axis,color in ((0,(255,180,70)),(1,(80,230,100))):
                if self.part==3:
                    if len(paths) and len(paths[axis])>1: cv2.polylines(panel,[paths[axis]],False,color,1)
                    continue
                prior=None
                for t,angles in self.head_signal[slot]:
                    if t<obs['at']-6.: continue
                    p=(int(left+(t-obs['at']+6)/6.*(right-left)),int(base-np.clip(angles[axis],-35,35)))
                    if prior: cv2.line(panel,prior,p,color,1)
                    prior=p
            event=self.head_event.get(slot)
            if event and obs['at']-event[0]<2.5:
                cv2.putText(panel,'NOD / YES' if event[1] else 'HEAD SHAKE / NO',(left,300),0,.45,(80,240,240),1)
            if self.part==3:
                d=self.motion[slot].diagnostics()
                cv2.putText(panel,f"Hands {d['hands_visible']} near {d['near_face']} cue {d['triggered']}",(left,322),0,.37,(200,200,200),1)
                d=self.cues[slot].diagnostics(obs['at'])
                fresh=d['sample_age_s'] is not None and d['sample_age_s']<=.3
                lowered=fresh and any('head lowered' in c or 'shifted downward' in c for c in d['stable_cues'])
                label='LOOKING DOWN' if lowered else 'Face unavailable' if not fresh else 'Posture ready' if d['calibrated'] else 'Calibrating posture'
                cv2.putText(panel,label,(left,341),0,.42,(80,240,240) if lowered else (200,200,200),1)
                cv2.putText(panel,f"Head delta {d['head_down_delta_deg']} | gaze {d['gaze_down_delta']}",(left,359),0,.37,(200,200,200),1)
        return clean,np.vstack((canvas,panel))

    def close(self):
        for model in (self.gesture,self.pose,self.face,self.objects):
            if model is not None: model.close()
