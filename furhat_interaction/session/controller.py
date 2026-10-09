"""Session state, semantic decisions and game transitions; no hardware ownership."""
import json
import random
from furhat_interaction.dialogue import clean_reply, extract_name
from furhat_interaction.game import Match, yes_no
from furhat_interaction.group.referee_game import RefereeGame
from furhat_interaction.multi_person.scenes import Scene
from furhat_interaction.reflection import HAND_CUE


class RandomMatch(Match):
    """Commit a random robot move before each hand capture."""
    def __init__(self):
        super().__init__()
        self.moves = [random.choice(('rock', 'paper', 'scissors')) for _ in range(3)]

    @property
    def robot_move(self):
        return self.moves[len(self.results)] if len(self.results) < 3 else None


class Session(Scene):
    def __init__(self, capacity=2):
        super().__init__(0)
        self.capacity = capacity
        self.visible = []
        self.single = None
        self.rules_players = ()
        self.last_speaker = None

    def start(self, obs):
        self.visible = [p['id'] for p in obs['people']]
        self.state = 'chat'
        known = [self.label(i) for i in self.visible if i in self.names]
        if known:
            return self.say('Welcome, ' + ' and '.join(known) + '! What would you like to talk about?', 'chat', self.visible[0])
        return None

    def expected(self):
        return self.focus if self.state == 'name' else None

    def stage(self, state, now):
        if state == 'after_game':
            self.part = 3
            self.state = 'reflect'
            return
        if state in ('chat', 'reflect'):
            self.part = 3 if state == 'reflect' else 0
        self.state = state
        if state == 'release':
            self.focus = None
            if self.single:
                self.single.release_since = None
            else:
                self.game.state = 'release'
                self.game.release_since = None
        elif state == 'playing':
            self.focus = None
            if not self.single:
                self.game.arm(now)
        self.done = False

    def observe(self, obs, now, vision, can_listen):
        self.visible = [p['id'] for p in obs['people']]
        if self.state in ('chat', 'reflect') and can_listen:
            unknown = [i for i in self.visible if i not in self.names and vision.seats.stable(i, now)]
            if unknown:
                slot = unknown[0]
                return self.say('Hello, welcome! What is your name?', 'name', slot)
        if self.part != 2 or self.state == 'busy':
            return None
        if self.state == 'rules' and can_listen:
            for slot, answer in obs['head_events'].items():
                if slot in self.rules_players:
                    self.rule_answers[slot] = answer
                    self.rule_sources[slot] = 'nod' if answer else 'head shake'
            if all(i in self.rule_answers for i in self.rules_players):
                return self.explain()
        if self.state not in ('release', 'playing') or not can_listen:
            return None
        if self.single:
            slot = self.rules_players[0]
            if slot not in self.visible:
                return None
            hands = [h for h in obs['hands'] if h['person_id'] == slot]
            if self.state == 'release' and self.single.released(hands, now):
                return self.say('Get ready. Rock, paper, scissors!', 'playing', target=slot)
            if self.state == 'playing':
                result = self.single.observe(hands, now)
                if result:
                    return self.single_result(slot, result)
            return None
        event = self.game.observe(self.visible, obs['hands'], now)
        if event == 'countdown':
            return self.say(f'Round {self.game.round}. Get ready. Rock, paper, scissors!', 'playing')
        if event == 'result':
            action = super().result()
            if action['next'] == 'complete':
                winner_name = self.memory.get('winner')
                self.winner = next((i for i, n in self.names.items() if n == winner_name), None)
                self.loser = next((i for i in self.rules_players if i != self.winner), self.first)
                target = self.loser if self.winner is not None else self.first
                action['next'] = 'after_game'
                action['after_text'] = f'Thank you for playing. {self.label(target)}, how did the game feel for you?'
                action['target'] = target
                action.setdefault('segments', []).append(dict(text=f'{self.label(target)}, let us reflect on the game.', look_at=target))
                self.focus = target
            return action
        if event == 'identity_changed':
            self.part = 0
            return self.say('I lost track of the players. Let us pause the game.', 'chat')
        if event in ('retry', 'paused') or self.game.tick(now) in ('retry', 'paused'):
            return self.say('Let us retry. Show one clear hand each, then lower your hands between rounds.', 'release')
        return None

    def begin_game(self, slot):
        players = [i for i in self.visible if i in self.names]
        if slot not in players:
            return self.say('Please stay in view so we can play together.', 'chat', slot)
        self.part = 2
        self.rule_answers = {}
        self.rule_sources = {}
        self.rules_players = tuple(players)
        self.first = slot
        self.second = next((i for i in players if i != slot), None)
        self.single = RandomMatch() if len(players) == 1 else None
        self.game = RefereeGame()
        if self.single:
            text = 'Let us play rock, paper, scissors together. Do you know the rules?'
        else:
            text = 'You can play rock, paper, scissors against each other. I will be the judge. Do you both know the rules?'
        return self.say(text, 'rules', slot)

    def explain(self):
        if not self.single:
            if all(self.rule_answers.values()):
                self.game.players = (self.first, self.second)
                self.game.scores = {i: 0 for i in self.game.players}
                return self.say('Great, three rounds. Lower your hands; let us begin.', 'release')
            return super().explain()
        slot = self.rules_players[0]
        text = 'Great, three rounds. Let us begin.' if self.rule_answers[slot] else (
            'No problem. A fist is rock, an open palm is paper, and a V sign is scissors. '
            'Rock beats scissors, scissors beats paper, and paper beats rock. '
            'Show one hand after the countdown and lower it between rounds. We will play three rounds.')
        return self.say(text, 'release', slot)

    def single_result(self, slot, result):
        outcome = {'furhat': 'I win this round!', 'person': 'You win this round!', 'draw': 'A draw!'}[result['winner']]
        text = f"You showed {result['person']}; I chose {result['furhat']}. {outcome}"
        finished = len(self.single.results) == 3
        after_text = None
        if finished:
            scores = self.single.scores()
            winner = 'Furhat' if scores['furhat'] > scores['person'] else self.label(slot) if scores['person'] > scores['furhat'] else None
            self.memory = dict(winner=winner, scores=scores, rounds=list(self.single.results))
            self.loser = slot
            text += ' The game is a draw.' if winner is None else f' {winner} wins the game!'
            after_text = f'Thank you for playing. {self.label(slot)}, how did the game feel for you?'
        return self.say(text, 'after_game' if finished else 'release', slot,
                        wink=result['winner'] == 'furhat', after_text=after_text)

    def transcript(self, slot, text, cues, objects):
        if self.state == 'name':
            if slot != self.focus:
                return None
            name = extract_name(text)
            if not name:
                return self.say('What should I call you? Please say my name is, followed by your name.', 'name', slot)
            self.names[slot] = name
            obj = f' I noticed a {objects}. What do you use it for?' if objects else ' How has your day been?'
            if objects in ('bottle', 'cup'):
                obj = f' I noticed your {objects}. What are you drinking?'
            return self.say(f'Nice to meet you, {name}.' + obj, 'chat', slot)
        if slot not in self.names:
            return self.say('Hello! What is your name?', 'name', slot)
        if self.state == 'rules':
            answer = yes_no(text)
            if answer is not None and slot in self.rules_players:
                self.rule_answers[slot] = answer
                self.rule_sources[slot] = 'speech'
                if all(i in self.rule_answers for i in self.rules_players):
                    return self.explain()
                self.focus = next(i for i in self.rules_players if i not in self.rule_answers)
                return None
        self.history.append(dict(person=self.label(slot), speech=text, visual_context=list(cues)))
        self.history = self.history[-12:]
        self.last_speaker = slot
        if self.state == 'correction':
            self.part = 3
        next_state = 'reflect' if self.state == 'correction' else self.state
        return self.say(None, next_state, slot, kind='conversation', speech=text, cues=cues,
                        objects=objects, interruptible=self.part != 2, phase=next_state)

    def apply_decision(self, action, slot):
        if action == 'start_game' and self.part != 2:
            return self.begin_game(slot)
        if action == 'stop_game' and self.part == 2:
            self.part = 0
            self.single = None
            return self.say('Of course. We can stop playing and talk instead.', 'chat', slot)
        return None


def decide(llm, job):
    """The LLM chooses game intent semantically; Python validates the action."""
    phase = job.get('phase', 'chat')
    context = dict(person=job['name'], speech=job.get('speech', ''), visual_cues=job.get('cues', []),
                   detected_object=job.get('objects'), game=job.get('game', {}),
                   phase=phase, conversation=job.get('history', [])[-6:])
    system = (
        'You are Furhat, a friendly robot in a live conversation. Reply to the named person in English. '
        'Use at most two short sentences and 35 words. Be curious and ask relevant questions. '
        'Visual cues are uncertain observations, not diagnoses. Acknowledge a detected object when relevant; do not invent objects. '
        'After a game, ask about their experience and respond kindly to disappointment. '
        'Return JSON with action and reply. action must be chat, start_game, or stop_game. '
        'Choose start_game only when the person requests or accepts playing; rock paper scissors is the only game. '
        'Choose stop_game when they want to stop an active game. Otherwise choose chat. '
        'Do not interpret an unrelated no or disagreement as ending the conversation. '
        'Never claim you have started a game in reply; Python handles game transitions.')
    if HAND_CUE in context['visual_cues']:
        system += " If near-face repetitive hand motion is observed, say: if you're hitting yourself, please stop and lower your hand. Contact is unconfirmed."
    data = llm.request('/api/chat', dict(model=llm.model, stream=False, keep_alive='5m', format='json',
        messages=[dict(role='system', content=system), dict(role='user', content=json.dumps(context, ensure_ascii=False))],
        options=dict(temperature=.2, num_ctx=2048, num_predict=140, num_thread=2)))
    raw = data.get('message', {}).get('content', '')
    parsed = json.loads(raw)
    action = parsed.get('action')
    if action not in ('chat', 'start_game', 'stop_game'):
        raise ValueError('Invalid conversation action')
    if action == 'start_game' and phase in ('rules', 'playing', 'release'):
        action = 'chat'
    if action == 'stop_game' and phase not in ('rules', 'playing', 'release'):
        action = 'chat'
    text = clean_reply(parsed.get('reply', ''))
    if not text:
        raise ValueError('Empty conversation reply')
    return action, text
