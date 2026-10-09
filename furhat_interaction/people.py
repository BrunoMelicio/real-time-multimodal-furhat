"""Small v33 policies; no camera, models, robot or external dependencies."""
from itertools import product
from math import isfinite
from queue import Queue, Empty, Full
from threading import Event, Thread


class CountWindow:
    """Announce a count only after continuously agreeing fresh observations.

    Uses counts rather than IDs: a tracker ID switch alone is not an arrival.
    Long gaps in inference reset the window instead of counting as evidence.
    """
    def __init__(self, arrival=1.5, departure=3., max_gap=1.):
        self.arrival, self.departure, self.max_gap = arrival, departure, max_gap
        self.confirmed = self.candidate = None
        self.since = self.last_sample = None

    @property
    def required(self):
        return (self.departure if self.confirmed is not None and self.candidate < self.confirmed
                else self.arrival)

    def update(self, count, now):
        count = min(2, max(0, int(count)))
        gap = self.last_sample is not None and now-self.last_sample > self.max_gap
        self.last_sample = now
        if gap or count != self.candidate:
            self.candidate, self.since = count, now
        if count == self.confirmed or now-self.since < self.required:
            return None
        previous, self.confirmed = self.confirmed, count
        return previous, count


def count_message(previous, count):
    if previous is None:
        return ('I do not see anyone yet.', 'I can see one person.', 'I can see two people.')[count]
    if count == 0:
        return 'Everyone is out of view now. Where did you go?'
    if count > previous:
        return 'Hello, I see one person now.' if count == 1 else 'Oh, there are two of you now.'
    return 'I see one person now. The other person is out of view.'


def select_people(rows):
    # The model is already filtered to class 0; defend the downstream boundary too.
    return sorted((row for row in rows if row['class'] == 0 and row['label'] == 'person'),
                  key=lambda row: row['confidence'], reverse=True)[:2]


def associate_faces(people, noses):
    """At most two faces: exhaustive one-to-one assignment avoids greedy swaps.

    A face must fall in the upper part of a person box. Prefer the face nearest
    that box's expected head position. Overlap/occlusion can still be ambiguous.
    """
    tracks = [row for row in people if row['id'] is not None]
    costs = {}
    for p, row in enumerate(tracks):
        x1, y1, x2, y2 = row['box']
        w, h = x2-x1, y2-y1
        if w <= 0 or h <= 0:
            continue
        for f, (x, y) in enumerate(noses):
            if x1-.03*w <= x <= x2+.03*w and y1-.05*h <= y <= y1+.60*h:
                costs[p, f] = abs(x-(x1+x2)/2)/w + abs(y-(y1+.18*h))/h
    best, best_score = {}, (1, float('inf'))
    for assignment in product(range(-1, len(noses)), repeat=len(tracks)):
        used = [f for f in assignment if f >= 0]
        if len(set(used)) != len(used) or any((p,f) not in costs for p,f in enumerate(assignment) if f >= 0):
            continue
        score = (-len(used), sum(costs[p,f] for p,f in enumerate(assignment) if f >= 0))
        if score < best_score:
            best_score = score
            best = {tracks[p]['id']: f for p,f in enumerate(assignment) if f >= 0}
    return best


class EyeEstimate:
    """Per-ID iris baseline and smoothed displacement, not physical gaze angles."""
    def __init__(self):
        self.baseline = self.since = self.last_sample = None
        self.samples = []
        self.current = [0., 0.]

    def update(self, uv, now):
        gap = self.last_sample is not None and now-self.last_sample > .5
        self.last_sample = now
        if uv is None or not all(isfinite(float(v)) for v in uv):
            self.samples.clear()
            self.since = None
            return None
        if gap:
            self.samples.clear()
            self.since = None
            self.current = [0., 0.]
        if self.baseline is None:
            if self.since is None:
                self.since = now
            self.samples.append(tuple(map(float, uv)))
            if now-self.since < 1.:
                return None
            self.baseline = tuple(sum(v[i] for v in self.samples)/len(self.samples) for i in (0,1))
            self.samples.clear()
        for i in (0,1):
            target = float(uv[i])-self.baseline[i]
            self.current[i] += .3*(target-self.current[i])
        return tuple(self.current)


def gaze_label(value):
    if value is None:
        return 'unknown / calibrating'
    x,y = value
    horizontal = 'right' if x > .035 else 'left' if x < -.035 else ''
    vertical = 'down' if y > .025 else 'up' if y < -.025 else ''
    return '/'.join(part for part in (horizontal,vertical) if part) or 'centre'


class CountSpeaker:
    """One speech worker, one pending update; camera never waits for TTS."""
    def __init__(self, robot, on_speech=None):
        self.robot = robot
        self.on_speech = on_speech
        self.pending = Queue(maxsize=1)
        self.stopping = Event()
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, change):
        try:
            self.pending.put_nowait(change)
        except Full:
            try:
                self.pending.get_nowait()
            except Empty:
                pass
            self.pending.put_nowait(change)

    def run(self):
        while not self.stopping.is_set():
            try:
                change = self.pending.get(timeout=.1)
            except Empty:
                continue
            if self.stopping.is_set():
                break
            message = count_message(*change)
            print(f'[COUNT SPEECH] {message}', flush=True)
            try:
                if self.on_speech is not None:
                    self.on_speech(True)
                self.robot.request_speak_text(message, wait=True, abort=False)
            except Exception as exc:
                print(f'[SPEECH ERROR] {type(exc).__name__}: {exc}; camera continues.',flush=True)
            finally:
                if self.on_speech is not None:
                    self.on_speech(False)

    def close(self):
        self.stopping.set()
        try:
            self.robot.request_speak_stop()
        except Exception as exc:
            print(f'[SPEECH STOP] {exc}',flush=True)
        self.thread.join(timeout=2)
