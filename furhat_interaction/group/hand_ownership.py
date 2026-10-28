"""Geometry only: person box -> body skeleton -> wrist -> classified hand."""
from math import hypot, isfinite


def point_xy(point, width, height, confidence=.5):
    values = (point.x, point.y, point.visibility, point.presence)
    if any(value is None or not isfinite(float(value)) for value in values):
        return None
    if min(point.visibility, point.presence) < confidence or not 0 <= point.x <= 1 or not 0 <= point.y <= 1:
        return None
    return float(point.x*width), float(point.y*height)


def body_owners(poses, people, width, height):
    """Require both shoulder midpoint and nose to have one clear box owner.

    Overlapping upper bodies and duplicate skeletons are rejected, rather than
    optimistically selecting whichever person happens to appear first.
    """
    output = {}
    for index, pose in enumerate(poses):
        if len(pose) < 33:
            continue
        joints = {i:point_xy(pose[i], width, height) for i in (0,11,12,15,16)}
        if any(joints[i] is None for i in (0,11,12)):
            continue
        shoulder = tuple((joints[11][i]+joints[12][i])/2 for i in (0,1))
        candidates = []
        for person in people:
            if person['id'] is None:
                continue
            x1,y1,x2,y2 = map(float,person['box'])
            w,h = x2-x1,y2-y1
            if w <= 0 or h <= 0:
                continue
            if all(x1-.03*w <= x <= x2+.03*w and y1-.06*h <= y <= y1+.65*h
                   for x,y in (joints[0],shoulder)):
                candidates.append(int(person['id']))
        if len(candidates) != 1:
            continue
        identity = candidates[0]
        scale = hypot(joints[11][0]-joints[12][0],joints[11][1]-joints[12][1])
        if scale < 25:
            continue
        wrists = {i:joints[i] for i in (15,16) if joints[i] is not None}
        output[index] = dict(person_id=identity,wrists=wrists,radius=max(25.,min(100.,scale*.65)))
    duplicated = {body['person_id'] for body in output.values()
                  if sum(other['person_id']==body['person_id'] for other in output.values()) > 1}
    return {index:body for index,body in output.items() if body['person_id'] not in duplicated}


def spatial_owner(wrist, people, margin=.25):
    """Use a clear person region when a pose wrist is temporarily unavailable.

    This uses the observed roster, never the configured participant capacity.
    Hands outside everyone's reach, and overlapping/tied regions, stay unknown.
    """
    x,y=wrist
    candidates=[]
    for person in people:
        identity=person.get('id')
        x1,y1,x2,y2=map(float,person['box'])
        width,height=x2-x1,y2-y1
        if identity is None or width<=0 or height<=0: continue
        dx=max(x1-x,0.,x-x2)/width
        dy=max(y1-y,0.,y-y2)/height
        if dx<=.2 and dy<=.15:
            candidates.append((hypot(dx,dy),identity))
    candidates.sort()
    if not candidates or (len(candidates)>1 and candidates[1][0]-candidates[0][0]<margin):
        return None
    return candidates[0][1]


def assign_hands(hands, bodies, margin=.25, *, people=()):
    """Prefer pose wrists, then clear person regions; reject competing owners.

    Handedness means left/right anatomy, never a player ID. A single body wrist
    cannot own two detections. Each output retains a diagnostic reason.
    """
    output = []
    for hand in hands:
        item = dict(hand,person_id=None,reason='No reliable body wrist nearby')
        x,y = hand['wrist']
        costs = []
        if not all(isfinite(float(value)) for value in (x,y)):
            output.append(item); continue
        for pose_index,body in bodies.items():
            possible = [(hypot(x-wx,y-wy)/body['radius'],joint,(wx,wy))
                        for joint,(wx,wy) in body['wrists'].items()]
            if possible:
                distance,joint,wrist = min(possible)
                costs.append((distance,body['person_id'],pose_index,joint,wrist))
        costs.sort()
        if costs and costs[0][0] <= 1.:
            if len(costs) > 1 and costs[1][0]-costs[0][0] < margin:
                item['reason'] = 'Ambiguous: wrists of both players are too close'
            else:
                distance,identity,pose_index,joint,wrist = costs[0]
                item.update(person_id=identity,pose_index=pose_index,body_joint=joint,
                            body_wrist=wrist,distance=float(distance),reason='Wrist match')
        else:
            identity=spatial_owner((x,y),people,margin)
            if identity is not None:
                item.update(person_id=identity,reason='Clear person region; pose wrist unavailable')
        output.append(item)
    used = {}
    for item in output:
        if item['person_id'] is not None and 'body_joint' in item:
            key = item['person_id'],item['body_joint']
            used.setdefault(key,[]).append(item)
    for matches in used.values():
        if len(matches) > 1:
            for item in matches:
                item.update(person_id=None,reason='Duplicate detections for one body wrist')
    return output
