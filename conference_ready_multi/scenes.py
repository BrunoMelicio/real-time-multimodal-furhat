"""Explicit scene sequencing; perception and speech keep their real evidence."""
from conference_ready_support import extract_name
from conference_ready_game import yes_no
from conference_ready_reflection import HAND_CUE,acknowledgment_action
from group_interaction.referee_game import RefereeGame


class Scene:
    def __init__(self,part,names=None,game=None,first_name=None):
        self.part=part;self.names=dict(names or {});self.memory=dict(game or {})
        self.state='armed';self.focus=None;self.first=None;self.second=None
        self.rule_answers={};self.rule_sources={};self.game=RefereeGame();self.done=False
        self.winner=self.loser=None;self.hand_seen=set();self.history=[]
        self.first_name=first_name

    def label(self,slot): return self.names.get(slot,f'Person {slot}')

    def say(self,text,next_state,target=None,**extra):
        self.state='busy'
        if target is not None: self.focus=target
        if self.part==2 and extra.get('segments'):
            self.focus=extra['segments'][0]['look_at']
        return dict(text=text,next=next_state,target=self.focus,**extra)

    def start(self,obs):
        ids=[p['id'] for p in obs['people']]
        if self.part==1:
            self.state='wait_first';return None
        self.first,self.second=sorted(ids,key=lambda i:next(p['box'][0] for p in obs['people'] if p['id']==i))
        preferred=next((i for i in ids if self.names.get(i)==self.first_name),None)
        if preferred is not None: self.first,self.second=preferred,next(i for i in ids if i!=preferred)
        if self.part==2:
            return self.say(f'{self.label(self.first)}, what do you like doing for fun?','hobby1',self.first)
        winner_name=self.memory.get('winner')
        self.winner=next((i for i,n in self.names.items() if n==winner_name),None)
        if self.winner is None:
            self.state='armed';raise ValueError('Part 3 requires a completed game with one overall winner; replay Part 2 if tied.')
        self.loser=next(i for i in ids if i!=self.winner)
        return self.say(f'{self.label(self.winner)}, how was the game for you?','winner_answer',self.winner)

    def observe(self,obs,now,vision,can_listen):
        ids=[p['id'] for p in obs['people']]
        if self.part==1:
            if self.state=='wait_object' and can_listen:
                label=vision.object_for(self.second,now)
                if label: return self.object_question(self.second,label)
            if self.state=='wait_first':
                stable=[i for i in ids if vision.seats.stable(i,now)]
                if stable:
                    self.first=stable[0]
                    return self.say('Hello! What is your name?','name1',self.first)
            if self.state=='wait_new':
                stable=[i for i in ids if i!=self.first and vision.seats.stable(i,now)]
                if stable:
                    self.second=stable[0]
                    return self.say('Oh, someone has joined us. Welcome! What is your name?','name2',self.second)
        if self.part==2:
            if self.state=='rules' and can_listen:
                for slot,answer in obs['head_events'].items():
                    self.rule_answers[slot]=answer;self.rule_sources[slot]='nod' if answer else 'head shake'
                if len(self.rule_answers)==2: return self.explain()
            if self.state in ('release','playing'):
                event=self.game.observe(ids,obs['hands'],now)
                if event=='countdown':
                    return self.say(f'Round {self.game.round}. Get ready. Rock, paper, scissors!','playing')
                if event=='result': return self.result()
                if event=='identity_changed':
                    return self.say('I lost track of the players. Please restart this scene from your original seats.','complete')
                if event in ('retry','paused') or self.game.tick(now) in ('retry','paused'):
                    if self.game.state=='paused': self.game.state='release'
                    return self.say('Let us retry that round. Keep your arms apart and show one clear hand each.','release')
        return None

    def stage(self,state,now):
        self.state=state
        if self.part==2 and state in ('release','playing','complete'): self.focus=None
        if state=='release':
            self.game.state='release';self.game.release_since=None
        if state=='playing': self.game.arm(now)
        if state=='complete': self.done=True

    def explain(self):
        segments=[]
        for slot,answer in sorted(self.rule_answers.items(),key=lambda item:not item[1]):
            if answer:
                text=self.label(slot)+' knows the rules.'
            else:
                observed='I saw your head shake; ' if self.rule_sources.get(slot)=='head shake' else ''
                text=self.label(slot)+', '+observed+'no problem, I will explain.'
            segments.append(dict(text=text,look_at=slot))
        text=('You play against each other, and I am the judge. A fist is rock, an open palm is paper, '
               'and a V sign is scissors.')
        segments.append(dict(text=text,look_at=None))
        segments.append(dict(text='Rock beats scissors, scissors beats paper, and paper beats rock. '
               'Three rounds; show one hand each after the countdown, then lower your hands.',look_at=None))
        self.game.players=(self.first,self.second);self.game.scores={self.first:0,self.second:0}
        self.game.state='instructions'
        return self.say(' '.join(s['text'] for s in segments),'release',segments=segments)

    def result(self):
        r=self.game.last_result
        moves=r['moves'];winner=r['winner_id']
        text=f'{self.label(self.first)} showed {moves[self.first]}. {self.label(self.second)} showed {moves[self.second]}.'
        segments=[dict(text=text,look_at=None),dict(text='A draw.' if winner is None else f'{self.label(winner)} wins this round.',look_at=winner)]
        finished=self.game.feedback_done()=='finished'
        if finished:
            a,b=self.game.scores[self.first],self.game.scores[self.second]
            winner=None if a==b else self.first if a>b else self.second
            self.memory=dict(winner=self.names.get(winner),scores={self.label(i):s for i,s in self.game.scores.items()},
                             rounds=self.game.results)
            segments.append(dict(text='The game is a draw.' if winner is None else f'{self.label(winner)} wins the game!',look_at=winner))
            return self.say(' '.join(s['text'] for s in segments),'complete',winner,segments=segments,
                            after_text='Thank you both for playing.',wink=winner is not None)
        return self.say(' '.join(s['text'] for s in segments),'release',winner,segments=segments)

    def expected(self):
        return {'name1':self.first,'day':self.first,'name2':self.second,'drink':self.second,
                'hobby1':self.first,'hobby2':self.second,'winner_answer':self.winner,
                'loser_answer':self.loser,'ack':self.loser,'correction':self.loser,'final_ack':self.loser}.get(self.state)

    def object_question(self,slot,label,greeting=''):
        questions={'bottle':"I noticed you're holding a bottle. What are you drinking?",
                   'cup':"I noticed you're holding a cup. What are you drinking?",
                   'book':"I noticed you're holding a book. What is it about?",
                   'cell phone':"I noticed you're holding a phone. What do you use it for?"}
        return self.say(greeting+questions.get(label,f"I noticed you're holding a {label}. What do you use it for?"),'drink',slot)

    def transcript(self,slot,text,cues,objects):
        if self.state=='rules':
            answer=yes_no(text)
            if answer is not None:
                self.rule_answers[slot]=answer;self.rule_sources[slot]='speech'
            if len(self.rule_answers)==2: return self.explain()
            missing=next((i for i in (self.first,self.second) if i not in self.rule_answers),None)
            if missing is not None: self.focus=missing
            return None
        expected=self.second if self.state=='wait_object' else self.focus if self.state in ('correction','final_ack') else self.expected()
        if expected is None or slot!=expected: return None
        self.history.append(dict(person=self.label(slot),speech=text,visual_context=cues))
        state=self.state
        if state in ('name1','name2'):
            name=extract_name(text)
            if not name: return self.say('Sorry, what is your name? Please say: my name is, then your name.',state,slot)
            self.names[slot]=name
            if state=='name1': return self.say(f'Nice to meet you, {name}. How has your day been?','day',slot)
            if objects: return self.object_question(slot,objects,f'Nice to meet you, {name}. ')
            return self.say(f'Nice to meet you, {name}. Could you show me the object clearly?','wait_object',slot)
        if state=='wait_object':
            if objects: return self.object_question(slot,objects)
            return None
        if state=='day': return self.say(None,'wait_new',slot,kind='day',speech=text,cues=cues)
        if state=='drink': return self.say(f'Thanks for sharing, {self.label(slot)}. It is lovely to have you both here.','complete',slot)
        if state=='hobby1': return self.say(f'What about you, {self.label(self.second)}? What do you enjoy doing for fun?','hobby2',self.second)
        if state=='hobby2':
            return self.say('Let us play rock, paper, scissors against each other, and I will be the judge. Do you both know the rules?',
                            'rules',self.first)
        if state=='winner_answer':
            return self.say(f'Thanks, {self.label(slot)}. {self.label(self.loser)}, how did the game feel for you?','loser_answer',self.loser)
        if state=='loser_answer': return self.say(None,'ack',slot,kind='first',speech=text,cues=cues)
        if state=='ack':
            action=acknowledgment_action(text,cues)
            if action=='pause': return self.say('Of course. We can pause here.','complete',slot)
            if action=='support': return self.say(None,'final_ack',slot,kind='support',speech=text,cues=cues)
            return self.say('The more we practise, the more familiar the game can become. We can try slowly, without keeping score, '
                            'and focus on learning together rather than winning every round.','correction',slot,interruptible=True)
        if state=='correction': return self.say(None,'final_ack',slot,kind='support',speech=text,cues=cues)
        if state=='final_ack': return self.say(None,'complete',slot,kind='choice',speech=text,cues=cues)
        return None
