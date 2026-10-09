"""Recent displayed FPS and optional aggregate timings; no file output."""
from collections import deque


class FrameStats:
    def __init__(self,profile=False):
        self.profile=profile;self.frames=deque();self.fps=0.
        self.since=None;self.count=0;self.total={};self.peaks={}

    def record(self,at,timings):
        self.frames.append(at)
        while len(self.frames)>2 and self.frames[0]<at-2.: self.frames.popleft()
        if len(self.frames)>1:
            self.fps=(len(self.frames)-1)/max(.001,at-self.frames[0])
        if not self.profile: return None
        if self.since is None: self.since=at
        self.count+=1
        for key,value in timings.items():
            self.total[key]=self.total.get(key,0.)+value
            self.peaks[key]=max(self.peaks.get(key,0.),value)
        if at-self.since<3.: return None
        result={key:round(value/self.count,1) for key,value in self.total.items()}
        result.update(fps=round(self.fps,1),people_peak_ms=round(self.peaks.get('people_ms',0.),1))
        self.since=at;self.count=0;self.total.clear();self.peaks.clear()
        return result
