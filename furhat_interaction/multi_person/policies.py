"""Model-free ownership, identity and attribution policies."""
from collections import Counter
from math import hypot


def center(box):
    return ((box[0]+box[2])/2,(box[1]+box[3])/2)


class Seats:
    """Short-range geometry continuity, not biometric re-identification.

    Do not transfer a known name to a distant new track. Crossing/occluded
    participants can remain unknown; filming assumes two separated seats.
    """
    def __init__(self):
        self.records={}

    def update(self,people,at,width):
        used=set();output=[]
        for item in sorted(people,key=lambda p:center(p['box'])[0]):
            x,y=center(item['box']);costs=[]
            for slot,old in self.records.items():
                if slot in used or at-old['last']>3.: continue
                ox,oy=center(old['box'])
                cost=hypot(x-ox,y-oy)/width
                if cost<.22: costs.append((cost,slot))
            costs.sort()
            slot=None
            if costs and (len(costs)==1 or costs[1][0]-costs[0][0]>.06): slot=costs[0][1]
            if slot is None and not costs and len(self.records)<2:
                slot=next(i for i in (1,2) if i not in self.records)
                self.records[slot]=dict(first=at)
            if slot is None: continue
            old=self.records[slot]
            if at-old.get('last',at)>.5: old['first']=at
            old.update(box=list(item['box']),last=at,track_id=item.get('id'))
            used.add(slot);output.append(dict(item,id=slot,track_id=item.get('id')))
        return output

    def stable(self,slot,now):
        row=self.records.get(slot)
        return bool(row and now-row['last']<.4 and now-row['first']>=.6)


def speaker_choice(scores,threshold=.65,margin=.12):
    ranked=sorted(scores.items(),key=lambda pair:pair[1],reverse=True)
    if not ranked or ranked[0][1]<threshold: return None
    if len(ranked)>1 and ranked[0][1]-ranked[1][1]<margin: return None
    return int(ranked[0][0])


def freeze_speaker(votes):
    """Freeze evidence collected DURING speech, never choose after decoding."""
    counts=Counter(v for v in votes if v is not None)
    if not counts: return None
    identity,count=counts.most_common(1)[0]
    return identity if count>=2 and count/sum(counts.values())>=.65 else None


def object_owner(box,people):
    """Unique spatial association only; a box does not prove holding."""
    x,y=center(box)
    owners=[p['id'] for p in people if p['box'][0]<=x<=p['box'][2] and p['box'][1]<=y<=p['box'][3]]
    if owners: return owners[0] if len(owners)==1 else None
    # A carried object may extend beyond the torso's person detection box.
    candidates=[]
    for p in people:
        x1,y1,x2,y2=p['box'];width=max(1,x2-x1)
        distance=hypot(max(x1-x,0,x-x2),max(y1-y,0,y-y2))/width
        if distance<=.2: candidates.append((distance,p['id']))
    candidates.sort()
    if candidates and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>.15): return candidates[0][1]
    return None


class HeadMotion:
    """Unwrapped angles, short median smoothing and dominant-axis evidence."""
    def __init__(self): self.reset()

    def reset(self):
        self.samples=[];self.previous=None;self.unwrapped=None;self.recent=[]
        self.metrics=dict(yaw_span=0.,pitch_span=0.,yaw_reversals=0,pitch_reversals=0)

    def update(self,angles,at):
        import numpy as np
        if angles is None:
            if self.samples and at-self.samples[-1][0]>.25: self.reset()
            return None
        if self.samples and at-self.samples[-1][0]>.25: self.reset()
        raw=np.array(angles[:2],dtype=float)
        if self.previous is None: self.unwrapped=raw.copy()
        else: self.unwrapped+=((raw-self.previous+180.)%360.)-180.
        self.previous=raw;self.recent.append(self.unwrapped.copy());self.recent=self.recent[-3:]
        smooth=np.median(self.recent,axis=0)
        self.samples.append((at,smooth));self.samples=[row for row in self.samples if at-row[0]<=2.]
        spans=[];reversals=[]
        for axis,step in ((0,5.),(1,4.)):
            values=[p[axis] for _,p in self.samples];anchor=values[0];direction=0;turns=0
            for value in values[1:]:
                if abs(value-anchor)>=step:
                    new=1 if value>anchor else -1
                    turns+=int(direction!=0 and new!=direction);anchor=value;direction=new
            spans.append(float(max(values)-min(values)));reversals.append(turns)
        self.metrics=dict(yaw_span=round(spans[0],1),pitch_span=round(spans[1],1),
                          yaw_reversals=reversals[0],pitch_reversals=reversals[1])
        if len(self.samples)<6 or at-self.samples[0][0]<.5: return None
        answer=None
        if spans[0]>=18. and spans[0]>=1.4*spans[1] and reversals[0]>=2: answer=False
        elif spans[1]>=13. and spans[1]>=1.4*spans[0] and reversals[1]>=1: answer=True
        if answer is not None:
            # Retain metrics for the event log, but require a fresh motion next.
            metrics=self.metrics;self.reset();self.metrics=metrics
        return answer


def bind_saved_names(people,saved,width):
    """Resolve saved seats geometrically on Space, explicitly rejecting ties."""
    if len(people)!=2 or len(saved)!=2: return None
    import itertools
    options=[]
    for rows in itertools.permutations(saved):
        costs=[abs(center(p['box'])[0]/width-r['x']) for p,r in zip(people,rows)]
        if max(costs)<=.25: options.append((sum(costs),rows))
    options.sort(key=lambda item:item[0])
    if not options or (len(options)>1 and options[1][0]-options[0][0]<.12): return None
    return {p['id']:r['name'] for p,r in zip(people,options[0][1])}
