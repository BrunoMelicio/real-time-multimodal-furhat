"""Three valid rounds; simultaneous stable moves; no models or robot imports."""
MOVES = {'Closed_Fist':'rock','Open_Palm':'paper','Victory':'scissors'}
BEATS = {'rock':'scissors','paper':'rock','scissors':'paper'}


class RefereeGame:
    def __init__(self,hold=.4,confidence=.6,answer_seconds=6.):
        self.hold,self.confidence,self.answer_seconds = hold,confidence,answer_seconds
        self.players,self.state,self.round = (), 'waiting', 1
        self.scores,self.draws,self.results = {},0,[]
        self.roster_candidate,self.roster_since = None,None
        self.release_since,self.missing_since = None,None
        self.candidate,self.since,self.samples,self.last_sample = None,None,0,None
        self.deadline,self.armed_at = None,None
        self.last_result,self.detail = None,'Waiting for two stable person IDs'

    def clear_pair(self):
        self.candidate,self.since,self.samples = None,None,0

    def observe(self,ids,hands,now,fresh=True):
        if not fresh:
            return None
        if self.last_sample is not None and now-self.last_sample > .55:
            self.clear_pair(); self.roster_since = self.release_since = None
        self.last_sample = now
        ids = tuple(ids)
        if self.state == 'waiting':
            if len(ids) != 2 or len(set(ids)) != 2:
                self.roster_candidate,self.roster_since = None,None
                return None
            if ids != self.roster_candidate or self.roster_since is None:
                self.roster_candidate,self.roster_since = ids,now
            if now-self.roster_since >= 1.:
                self.players,self.scores = ids,{identity:0 for identity in ids}
                self.state = 'instructions'
                return 'instructions'
            return None
        if self.state in ('finished','identity_changed'):
            return None
        if len(ids) >= 2 and set(ids) != set(self.players):
            self.state = 'identity_changed'
            self.clear_pair()
            self.detail = 'Person IDs changed: press R to restart safely'
            return 'identity_changed'
        if set(ids) != set(self.players):
            self.clear_pair(); self.release_since = None
            self.missing_since = now if self.missing_since is None else self.missing_since
            return None
        self.missing_since = None
        if self.state in ('release','paused'):
            playing_hands = [hand for hand in hands if hand['label'] in MOVES and hand['score'] >= self.confidence]
            if playing_hands:
                self.release_since = None
                self.detail = 'Lower playing hands before the next countdown'
            else:
                if self.release_since is None:
                    self.release_since = now
                if now-self.release_since >= .45:
                    self.state,self.release_since = 'countdown',None
                    self.detail = 'Wait for the countdown to finish'
                    return 'countdown'
            return None
        if self.state != 'playing' or now < self.armed_at:
            return None
        if now >= self.deadline:
            return self.tick(now)
        selected = {identity:[hand for hand in hands if hand['person_id']==identity] for identity in self.players}
        if any(hand['person_id'] is None for hand in hands):
            self.detail = 'Hand ownership uncertain: separate your wrists / arms'
            self.clear_pair(); return None
        if any(len(selected[identity]) != 1 for identity in self.players):
            self.detail = 'Each player must show exactly one visible hand'
            self.clear_pair(); return None
        pair = []
        for identity in self.players:
            hand = selected[identity][0]
            if hand['label'] not in MOVES or hand['score'] < self.confidence:
                self.detail = 'Show a clear fist, open palm or V sign'
                self.clear_pair(); return None
            pair.append((identity,MOVES[hand['label']]))
        pair = tuple(pair)
        if pair != self.candidate or self.since is None:
            self.candidate,self.since,self.samples = pair,now,1
        else:
            self.samples += 1
        self.detail = f'Hold both moves together: {max(0.,self.hold-(now-self.since)):.1f}s'
        if now-self.since < self.hold or self.samples < 3:
            return None
        first,second = pair
        winner = None if first[1]==second[1] else first[0] if BEATS[first[1]]==second[1] else second[0]
        if winner is None:
            self.draws += 1
        else:
            self.scores[winner] += 1
        self.last_result = dict(round=self.round,status='valid',moves=dict(pair),winner_id=winner)
        self.results.append(self.last_result)
        self.state = 'feedback'
        self.clear_pair()
        return 'result'

    def instructions_done(self):
        if self.state == 'instructions':
            self.state = 'release'

    def arm(self,now):
        if self.state != 'countdown':
            return
        self.state,self.armed_at,self.deadline = 'playing',now,now+self.answer_seconds
        self.last_sample = None
        self.clear_pair()
        self.detail = 'Show one hand each and hold together'

    def tick(self,now):
        if self.state == 'playing' and self.missing_since is not None and now-self.missing_since >= .8:
            self.state = 'paused'
            self.detail = 'Waiting for the original two players; no points awarded'
            self.clear_pair()
            return 'paused'
        if self.state == 'playing' and now >= self.deadline:
            self.state = 'feedback'
            self.last_result = dict(round=self.round,status='retry',moves={},winner_id=None)
            self.results.append(self.last_result)
            self.clear_pair()
            return 'retry'
        return None

    def feedback_done(self):
        if self.state != 'feedback':
            return None
        if self.last_result['status']=='valid':
            if self.round == 3:
                self.state = 'finished'
                return 'finished'
            self.round += 1
        self.state,self.release_since = 'release',None
        return None

    def summary(self):
        return dict(state=self.state,players=list(self.players),scores=self.scores,draws=self.draws,
                    completed_rounds=sum(item['status']=='valid' for item in self.results),attempts=self.results)
