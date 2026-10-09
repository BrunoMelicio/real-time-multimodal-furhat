"""Part 3 visual context: independent cue persistence and explicit feedback."""
from collections import deque
import re
import numpy as np
from conference_ready_reflection import CueHistory


class ReflectionCues(CueHistory):
    def __init__(self):
        super().__init__()
        self.recent=deque(maxlen=60);self.onsets={};self.last=None;self.stable=[]
        self.pitch_delta=self.gaze_delta=None;self.angles=None

    def rebase(self,at):
        """Use the settled posture immediately before Space, not model startup."""
        rows=[r for r in self.recent if at-1.<=r[0]<=at]
        if len(rows)<3 or rows[-1][0]-rows[0][0]<.3: return False
        # Unwrap around a recent reference before taking the pitch median.
        ref=rows[-1][1]
        pitches=[ref+(r[1]-ref+180.)%360.-180. for r in rows]
        gaze=[r[2] for r in rows if r[2] is not None]
        self.baseline=(float(np.median(pitches)),float(np.median(gaze)) if gaze else None)
        self.onsets.clear();self.rows.clear();self.stable=[]
        self.pitch_delta=self.gaze_delta=None
        return True

    def add(self,angles,uv,scores,at):
        if angles is not None:
            self.angles=tuple(float(a) for a in angles)
            self.recent.append((at,float(angles[1]),float(uv[1]) if uv is not None else None))
        raw=[];self.pitch_delta=self.gaze_delta=None
        if self.baseline is not None and angles is not None:
            self.pitch_delta=(float(angles[1])-self.baseline[0]+180.)%360.-180.
            if self.pitch_delta>10.: raw.append('head lowered relative to starting posture')
            if uv is not None and self.baseline[1] is not None:
                self.gaze_delta=float(uv[1])-self.baseline[1]
                if self.gaze_delta>.045: raw.append('iris gaze estimate shifted downward')
        smile=(scores.get('mouthSmileLeft',0)+scores.get('mouthSmileRight',0))/2
        if smile>=.35: raw.append('smiling facial movement')
        elif min(scores.get('mouthFrownLeft',0),scores.get('mouthFrownRight',0))>=.5:
            raw.append('downturned mouth movement')
        if self.last is None or at-self.last>.3: self.onsets.clear()
        self.onsets={cue:self.onsets.get(cue,at) for cue in raw}
        self.stable=[cue for cue in raw if at-self.onsets[cue]>=.4]
        self.last=at;self.rows.append((at,list(self.stable)))
        return list(self.stable)

    def diagnostics(self,at):
        return dict(calibrated=self.baseline is not None,
                    pitch_deg=round(self.angles[1],1) if self.angles else None,
                    pitch_reference_deg=round(self.baseline[0],1) if self.baseline else None,
                    head_down_delta_deg=round(self.pitch_delta,1) if self.pitch_delta is not None else None,
                    gaze_down_delta=round(self.gaze_delta,3) if self.gaze_delta is not None else None,
                    sample_age_s=round(max(0.,at-self.last),2) if self.last is not None else None,
                    stable_cues=list(self.stable))


def acknowledge_downward(text,cues):
    text=text.strip().strip('"“”')
    lowered=any('head lowered' in cue or 'shifted downward' in cue for cue in cues)
    observed=re.search(r'\b(?:look(?:ing|ed)?|gaz(?:e|ing)|head)\b[^.!?]{0,35}\b(?:down(?:ward)?|lowered)\b',text,re.I)
    if lowered and not observed: return 'I noticed you looked down. '+text
    return text
