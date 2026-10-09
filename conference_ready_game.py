"""Small Part 2 policies. No hardware or neural inference at import."""
from collections import deque
from pathlib import Path
import json
import re

MOVES = {'Closed_Fist':'rock', 'Open_Palm':'paper', 'Victory':'scissors'}
BEATS = {'rock':'scissors', 'paper':'rock', 'scissors':'paper'}
ROBOT_MOVES = ('paper','scissors','rock')  # Rehearsed, committed before seeing the hand.


def remembered_name(root):
    folder = Path(root)/'text_output/conference_ready_1'
    for path in sorted(folder.glob('*/session.json'),reverse=True):
        try:
            name = json.loads(path.read_text()).get('name')
            if isinstance(name,str) and name.strip():
                return name.strip()
        except (OSError,ValueError):
            continue
    return None


def yes_no(text):
    text = text.lower().replace('’',"'")
    if re.search(r"\b(no|nope|not|don't|cannot|can't|never)\b",text):
        return False
    if re.search(r"\b(yes|yeah|yep|sure|okay|ok|absolutely|please|know|agree)\b|let's",text):
        return True
    return None


class HeadAnswer:
    """Large alternating yaw/pitch excursions within two seconds; not gaze/emotion."""
    def __init__(self):
        self.samples = deque()
        self.last = None

    def reset(self):
        self.samples.clear(); self.last = None

    def update(self, angles, now):
        if angles is None:
            if self.last is not None and now-self.last > .25:
                self.reset()
            return None
        if self.last is not None and now-self.last > .25:
            self.reset()
        self.last = now
        self.samples.append((now,float(angles[0]),float(angles[1])))
        while self.samples and now-self.samples[0][0] > 2.:
            self.samples.popleft()
        if len(self.samples)<6 or now-self.samples[0][0]<.35:
            return None
        # Anchor advances only after meaningful movement, suppressing small jitter.
        strengths = []
        for axis, step, span in ((1,5.,18.),(2,4.,13.)):
            values = [s[axis] for s in self.samples]
            anchor, direction, reversals = values[0],0,0
            for value in values[1:]:
                delta = value-anchor
                if abs(delta)>=step:
                    new = 1 if delta>0 else -1
                    reversals += int(direction!=0 and direction!=new)
                    direction, anchor = new,value
            strengths.append((max(values)-min(values))/span if reversals>=(2 if axis==1 else 1) else 0.)
        if max(strengths)<1:
            return None
        answer = False if strengths[0]>=strengths[1] else True
        self.reset()
        return answer


class Match:
    def __init__(self):
        self.results = []
        self.candidate = self.since = self.last = None
        self.samples = 0
        self.release_since = None

    @property
    def robot_move(self):
        return ROBOT_MOVES[len(self.results)] if len(self.results)<3 else None

    def reset_hold(self):
        self.candidate = self.since = self.last = None
        self.samples = 0

    def released(self, hands, now):
        if hands:
            self.release_since = None
            return False
        if self.release_since is None:
            self.release_since = now
        return now-self.release_since>=.35

    def observe(self, hands, now):
        if len(self.results)>=3:
            return None
        if self.last is not None and now-self.last>.3:
            self.reset_hold()
        self.last = now
        if len(hands)!=1 or hands[0]['label'] not in MOVES or hands[0]['score']<.6:
            self.candidate = self.since = None; self.samples = 0
            return None
        move = MOVES[hands[0]['label']]
        if move!=self.candidate:
            self.candidate,self.since,self.samples = move,now,1
        else:
            self.samples += 1
        if now-self.since<.35 or self.samples<3:
            return None
        robot = self.robot_move
        winner = 'draw' if move==robot else 'person' if BEATS[move]==robot else 'furhat'
        result = dict(round=len(self.results)+1,person=move,furhat=robot,winner=winner)
        self.results.append(result)
        self.reset_hold(); self.release_since = None
        return result

    def scores(self):
        return {who:sum(r['winner']==who for r in self.results) for who in ('person','furhat','draw')}
