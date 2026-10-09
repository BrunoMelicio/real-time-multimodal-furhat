"""Part 3 context and policies; no camera, microphone or model inference."""
from collections import deque
import json
import re
from pathlib import Path
import numpy as np
from furhat_interaction.dialogue import Ollama, clean_reply
from furhat_interaction.game import remembered_name


def game_memory(root):
    for path in sorted((Path(root)/'text_output/single_person/game').glob('*/session.json'),reverse=True):
        try:
            data=json.loads(path.read_text())
            if len(data.get('rounds',[]))==3:
                return {k:data.get(k) for k in ('name','scores','rounds')}
        except (OSError,ValueError):
            continue
    return dict(name=remembered_name(root),scores=None,rounds=[])


class CueHistory:
    """Calibrated head/iris displacement and sustained expression observations."""
    def __init__(self):
        self.baseline=None
        self.calibration=[]
        self.rows=deque(maxlen=240)
        self.candidate,self.since=None,None

    def calibrate(self, angles, uv, at):
        if angles is None:
            return False
        self.calibration.append((at,float(angles[1]),float(uv[1]) if uv is not None else None))
        self.calibration=self.calibration[-50:]
        if len(self.calibration)>=10 and at-self.calibration[0][0]>=.8:
            iris=[r[2] for r in self.calibration if r[2] is not None]
            self.baseline=(float(np.median([r[1] for r in self.calibration])),float(np.median(iris)) if iris else None)
        return self.baseline is not None

    def add(self, angles, uv, scores, at):
        cues=[]
        if self.baseline is not None and angles is not None:
            delta=(float(angles[1])-self.baseline[0]+180)%360-180
            if delta>10:
                cues.append('head lowered relative to starting posture')
            if uv is not None and self.baseline[1] is not None and float(uv[1])-self.baseline[1]>.045:
                cues.append('iris gaze estimate shifted downward')
        smile=(scores.get('mouthSmileLeft',0)+scores.get('mouthSmileRight',0))/2
        frown=min(scores.get('mouthFrownLeft',0),scores.get('mouthFrownRight',0))
        if smile>=.35:
            cues.append('smiling facial movement')
        elif frown>=.5:
            cues.append('downturned mouth movement')
        candidate=tuple(cues)
        if not self.rows or at-self.rows[-1][0]>.3 or candidate!=self.candidate:
            self.candidate,self.since=candidate,at
        stable=list(candidate) if at-self.since>=.4 else []
        self.rows.append((at,stable))
        return stable

    def context(self,start,end):
        seen=[]
        for at,cues in self.rows:
            if start-1.5<=at<=end+.2:
                for cue in cues:
                    if cue not in seen: seen.append(cue)
        return seen


class BargeEvidence:
    """Two fresh neural matches, not VAD alone, in visual echo-guard mode."""
    def __init__(self,threshold=.65):
        self.threshold=threshold
        self.matches=deque(maxlen=2)

    def reset(self):
        self.matches.clear()

    def update(self,score,at,voiced):
        if not voiced or score<self.threshold:
            self.reset(); return False
        if self.matches and at-self.matches[-1]>.6:
            self.reset()
        self.matches.append(at)
        return len(self.matches)==2 and self.matches[-1]-self.matches[0]>=.1


HAND_CUE='repeated hand motion close to the face (contact and intent unknown)'


class HandNearFace:
    """2D proximity + alternating motion. Does NOT establish impact or intent."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.rows=deque()
        self.fired=False
        self.hands_visible=0
        self.near=False
        self.motion_range=0.
        self.reversals=0
        self.tracked_frames=0
        self.near_frames=0

    def diagnostics(self):
        return dict(hands_visible=self.hands_visible,near_face=self.near,
                    motion_range=round(self.motion_range,3),reversals=self.reversals,
                    tracked_frames=self.tracked_frames,near_frames=self.near_frames,
                    triggered=self.fired)

    def update(self,face_box,hands,at):
        self.hands_visible=len(hands)
        self.near=False
        if hands: self.tracked_frames+=1
        if face_box is None or not hands:
            self.rows.clear();self.motion_range=0.;self.reversals=0;return False
        x1,y1,x2,y2=face_box
        scale=np.array([max(1,x2-x1),max(1,y2-y1)])
        center=np.array([(x1+x2)/2,(y1+y2)/2])
        palms=[np.mean(hand['points'][[0,5,9,13,17]],axis=0) for hand in hands]
        positions=[(p-center)/scale for p in palms]
        position=min(positions,key=lambda p:float(np.linalg.norm(p)))
        # Face-relative coordinates suppress apparent motion caused by head travel.
        near=abs(position[0])<=.8 and abs(position[1])<=.8
        self.near=bool(near)
        if near: self.near_frames+=1
        if self.fired: return False
        if not near:
            self.rows.clear();self.motion_range=0.;self.reversals=0;return False
        if self.rows and (at-self.rows[-1][0]>.25 or np.linalg.norm(position-self.rows[-1][1])>.7):
            self.rows.clear()
        self.rows.append((at,position))
        while self.rows and at-self.rows[0][0]>2.: self.rows.popleft()
        self.motion_range=0.;self.reversals=0
        for axis in (0,1):
            values=[p[axis] for _,p in self.rows]
            anchor,direction,reversals=values[0],0,0
            for value in values[1:]:
                delta=value-anchor
                if abs(delta)>=.07:
                    new=1 if delta>0 else -1
                    reversals+=int(direction!=0 and direction!=new)
                    anchor,direction=value,new
            span=max(values)-min(values)
            self.motion_range=max(self.motion_range,float(span))
            self.reversals=max(self.reversals,reversals)
            if len(self.rows)>=6 and at-self.rows[0][0]>=.4 and span>=.18 and reversals>=2:
                self.fired=True;return True
        return False


class ReflectionLLM(Ollama):
    def respond(self, kind, name, speech, memory, previous, cues):
        goal={
            'first':'Respond to their disappointment about the game in 20-35 words. Acknowledge the actual spoken feeling; losing is okay and learning takes practice. If head/gaze is lowered, explicitly acknowledge looking down without naming sadness or shyness and without ordering them to look up. End with a calm reassurance, not a question.',
            'support':'The person responded during reflection or interrupted encouragement with new or corrected information. Accept the latest answer; the new answer replaces the old one if they corrected it. If they criticize themselves, gently separate losing from their worth or intelligence. If hand_safety_context is supplied, explicitly ask them to stop possible self-hitting using conditional wording, explain it could hurt, and ask them to lower their hand. Do not claim contact or injury occurred. Use at most 40 words, no question.',
            'choice':'Respond warmly to their final acknowledgment in at most 15 words, no further question; do not start another game or assume agreement if their words disagree.'}[kind]
        system=('You are Furhat, a warm robot speaking English to one person after rock, paper, scissors. '
                'Use plain spoken text, at most two sentences. No JSON, stage directions or diagnoses. '
                'Use the supplied facts, spoken words and correction; never invent details. '
                'Visual cues are uncertain observations, not proof of feelings; do not infer sadness, depression or autism. '
                'Offer choices without demanding eye contact. Do not say the conversation is over. '+goal)
        context=dict(person=name,game=memory,earlier_speech=previous,latest_speech=speech,
                     visual_observations=cues)
        if kind=='support' and HAND_CUE in cues:
            context['hand_safety_context']=dict(
                observation=HAND_CUE,contact_confirmed=False,
                required_response="If you're hitting yourself, please stop; it could hurt you. Ask them to lower their hand, then reassure them based on their actual words.")
        result=self.request('/api/chat',dict(model=self.model,stream=False,keep_alive='5m',
            messages=[dict(role='system',content=system),dict(role='user',content=json.dumps(context,ensure_ascii=False))],
            options=dict(temperature=.2,num_ctx=2048,num_predict=120,num_thread=2)))
        return clean_reply(result.get('message',{}).get('content',''))


def reflective_reply(text,cues):
    lowered=any('head lowered' in cue or 'shifted downward' in cue for cue in cues)
    if lowered and not re.search(r'\b(down|lowered)\b',text,re.I):
        return "I noticed you're looking down. "+text
    return text


def acknowledgment_action(speech,cues):
    """This scene asks for acknowledgment, not a yes/no answer or permission."""
    text=speech.lower().replace('’',"'")
    if HAND_CUE in cues or re.search(r'\b(stupid|dumb|useless|idiot)\b',text):
        return 'support'
    if re.search(r"\b(?:please |let's |can we |i want to |i need to )?(?:pause|stop|take a break)\b",text) and not re.search(r"\b(?:don't|do not|no need to) (?:pause|stop|take a break)\b",text):
        return 'pause'
    return 'encourage'


def reassuring_reply(text,cues,speech=''):
    """Prescribed response to an uncertain visual cue; no diagnosis or LLM retry."""
    if HAND_CUE in cues:
        reminder="If you're hitting yourself, please stop; it could hurt you. "
        if re.search(r'\b(stupid|dumb|useless|idiot)\b',speech,re.I):
            return reminder+'Please lower your hand; losing a game does not make you stupid, and we can take this slowly.'
        return reminder+'Please lower your hand for a moment; we can pause and take this at your own pace.'
    return text
