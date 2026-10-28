"""Session state, semantic decisions and game transitions; no hardware ownership."""
import json
import random
import re
from furhat_interaction.dialogue import clean_reply, extract_name
from furhat_interaction.game import Match, yes_no
from furhat_interaction.group.referee_game import RefereeGame
from furhat_interaction.multi_person.scenes import Scene
from furhat_interaction.reflection import HAND_CUE


def acceptance(text):
    answer = yes_no(text)
    if re.fullmatch(r'why not[.!?]*', text.strip(), re.I): return True
    if answer is not False and re.search(r"\b(let us|let's|go ahead|sounds good|sounds fun|i'm in)\b", text, re.I):
        return True
    return answer


def play_request(text):
    """Distinguish an instruction to play from talking about games/hobbies."""
    text = text.lower().replace('’', "'")
    if re.search(r"\b(no|not|never|don't|cannot|can't)\b", text): return False
    intent = re.search(r"\b(?:want(?: us)? to|would like to|let's|let us|can we|could we|shall we|(?:we|i) can|ready to)\s+(?:play|start(?: playing)?|begin)\b", text)
    target = re.search(r'\bgames?\b|\brock\b.*\bpaper\b.*\bscissors\b|\bplay[.!?]*$', text)
    return bool(intent and target)


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
        self.game_offer = None
        self.game_acceptance = None
        self.chat_turns = 0
        self.last_game_offer_turn = 0
        self.head_answers = {}
        self.question_started = None
        self.rules_prompted = set()

    def say(self, text, next_state, target=None, **extra):
        # A response belongs to the current robot turn, never a later question.
        self.head_answers.clear()
        self.question_started = None
        return super().say(text, next_state, target, **extra)

    def robot_speaking(self, at):
        """Accept gestures made during this prompt or its listening pause."""
        self.question_started = at
        self.head_answers.clear()

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
        if self.question_started is None:
            self.question_started = now
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
        at = obs.get('at', now)
        for slot, answer in obs['head_events'].items():
            if (slot in self.visible and self.question_started is not None
                    and at >= self.question_started):
                self.head_answers[slot] = (at, answer)
        self.head_answers = {slot: row for slot, row in self.head_answers.items()
                             if slot in self.visible and 0 <= now-row[0] <= 3.}
        if self.state == 'busy' or not can_listen:
            return None
        for slot, (_, answer) in list(self.head_answers.items()):
            self.head_answers.pop(slot, None)
            # Reuse speech handling, including invitation and rules state.
            # Other conversational questions go to the LLM with the gesture.
            question = next((row['speech'] for row in reversed(self.history)
                             if row['person'] == 'Furhat'), '')
            addressed = self.focus in (None, slot)
            if (self.game_offer == slot or (self.state == 'rules' and slot in self.rules_players)
                    or (self.state in ('chat', 'reflect') and addressed and '?' in question)):
                cue = 'participant nodded yes' if answer else 'participant shook their head no'
                action = self.transcript(slot, 'Yes.' if answer else 'No.', [cue], None)
                if action is not None:
                    action['response_source'] = 'nod' if answer else 'head shake'
                    action['response_person'] = slot
                    return action
        if self.state in ('chat', 'reflect') and can_listen:
            unknown = [i for i in self.visible if i not in self.names and vision.seats.stable(i, now)]
            if unknown:
                slot = unknown[0]
                return self.say('Hello, welcome! What is your name?', 'name', slot)
        if self.part != 2 or self.state == 'busy':
            return None
        if self.state == 'rules' and self.rule_answers:
            missing = next((i for i in self.rules_players if i not in self.rule_answers), None)
            if missing is not None and missing not in self.rules_prompted:
                return self.rules_prompt(missing)
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
        self.game_offer = self.game_acceptance = None
        players = [i for i in self.visible if i in self.names]
        if slot not in players:
            return self.say('Please stay in view so we can play together.', 'chat', slot)
        self.part = 2
        self.rule_answers = {}
        self.rule_sources = {}
        self.rules_prompted.clear()
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
            if all(self.rule_answers.get(i) is True for i in self.rules_players):
                self.game.players = (self.first, self.second)
                self.game.scores = {i: 0 for i in self.game.players}
                return self.say('Great, three rounds. Lower your hands; let us begin.', 'release')
            return super().explain()
        slot = self.rules_players[0]
        text = 'Great, three rounds. Let us begin.' if self.rule_answers.get(slot, False) else (
            'No problem. A fist is rock, an open palm is paper, and a V sign is scissors. '
            'Rock beats scissors, scissors beats paper, and paper beats rock. '
            'Show one hand after the countdown and lower it between rounds. We will play three rounds.')
        return self.say(text, 'release', slot)

    def rules_prompt(self, slot=None):
        slot = slot or next((i for i in self.rules_players if i not in self.rule_answers), self.first)
        self.rules_prompted.add(slot)
        return self.say(f'{self.label(slot)}, do you know the rules? I can explain before we start.', 'rules', slot)

    def rules_reply(self, slot, text, cues=()):
        """Speech and head answers share one setup transition, including group replies."""
        for person, (_, answer) in self.head_answers.items():
            if person in self.rules_players:
                self.rule_answers[person] = answer
                self.rule_sources[person] = 'nod' if answer else 'head shake'
        self.head_answers.clear()
        answer = acceptance(text)
        source = ('head shake' if 'participant shook their head no' in cues else
                  'nod' if 'participant nodded yes' in cues else 'speech')
        collective = slot is None or bool(re.search(r'\b(we|both|all of us)\b', text, re.I))
        if answer is not None:
            people = self.rules_players if collective else (slot,)
            for person in people:
                if person in self.rules_players:
                    self.rule_answers[person] = answer
                    self.rule_sources[person] = 'group speech' if collective else source
        # One person needing instructions is enough to explain to everyone.
        if False in self.rule_answers.values() or all(i in self.rule_answers for i in self.rules_players):
            return self.explain()
        if re.search(r'\b(start|begin|ready)\b', text, re.I) and answer is not False:
            # Readiness can bypass the knowledge question; give the short rules
            # instead of pretending the missing players already answered.
            return self.explain()
        if answer is None:
            return self.rules_prompt()
        self.focus = next(i for i in self.rules_players if i not in self.rule_answers)
        return None

    def unattributed(self, text):
        """Group game controls do not require guessing who spoke.

        Names, personal turns and per-person ownership still require identity.
        The invitation owner is used as an attention target, not as speaker ID.
        """
        if self.part == 2 and re.search(r'\b(stop|quit|cancel|pause)\b|\bno more\b', text, re.I):
            return self.apply_decision('stop_game', None)
        if self.state == 'rules':
            return self.rules_reply(None, text)
        if self.game_offer is not None and acceptance(text) is not None:
            self.history.append(dict(person='Unattributed participant', speech=text, visual_context=[]))
            self.history = self.history[-12:]
            target = self.game_offer
            self.game_offer = self.game_acceptance = None
            if acceptance(text): return self.begin_game(target)
            return self.say('Of course. We can keep chatting instead.', 'chat', target)
        return None

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
        # A spoken turn supersedes a buffered head answer from the same turn.
        if self.state != 'rules': self.head_answers.clear()
        # Consume an invitation on the next attributed utterance, even a name turn.
        invitation = self.game_offer == slot
        self.game_offer = None
        self.game_acceptance = slot if invitation else None
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
        if self.part == 2 and re.search(r'\b(stop|quit|cancel|pause)\b|\bno more\b',text,re.I):
            return self.apply_decision('stop_game',slot)
        if self.state == 'rules':
            return self.rules_reply(slot, text, cues)
        if self.part != 2 and play_request(text):
            self.history.append(dict(person=self.label(slot),speech=text,visual_context=list(cues)))
            self.history = self.history[-12:]
            return self.begin_game(slot)
        if invitation:
            answer = acceptance(text)
            if answer is True:
                self.history.append(dict(person=self.label(slot),speech=text,visual_context=list(cues)))
                self.history = self.history[-12:]
                return self.begin_game(slot)
            if answer is False:
                self.game_acceptance = None
        self.history.append(dict(person=self.label(slot), speech=text, visual_context=list(cues)))
        self.history = self.history[-12:]
        self.last_speaker = slot
        if self.part != 2: self.chat_turns += 1
        if self.state == 'correction':
            self.part = 3
        next_state = 'reflect' if self.state == 'correction' else self.state
        game_topic = bool(re.search(r'\b(play|playing|games?|fun)\b',text,re.I))
        declined = bool(re.search(r"\b(no thanks|nope|not now|not today|rather not|(?:don't|don’t|do not) (?:want|play|like))\b",text,re.I))
        offer = (self.part != 2 and not invitation and not declined and
                 (game_topic or self.chat_turns-self.last_game_offer_turn >= 4))
        if offer:
            self.game_offer = slot
            self.last_game_offer_turn = self.chat_turns
        return self.say(None, next_state, slot, kind='conversation', speech=text, cues=cues,
                        objects=objects, interruptible=True, phase=next_state,
                        game_invitation=invitation,offer_game=offer)

    def apply_decision(self, action, slot):
        if action == 'offer_game' and self.part != 2 and slot in self.visible and slot in self.names:
            self.game_offer = slot
            self.game_acceptance = None
            self.last_game_offer_turn = self.chat_turns
            return self.say('Would you like to play a few rounds of rock, paper, scissors?', 'chat', slot)
        if action == 'start_game' and self.part != 2 and self.game_acceptance == slot:
            return self.begin_game(slot)
        if action == 'stop_game' and self.part == 2:
            self.part = 0
            self.single = None
            self.game_offer = self.game_acceptance = None
            return self.say('Of course. We can stop playing and talk instead.', 'chat', slot)
        return None

    def interrupted(self, next_state, slot, source, now):
        """Resume the interrupted conversation/question, keeping game memory."""
        self.stage(next_state,now)
        for row in reversed(self.history):
            if row.get('person')=='Furhat':
                row['interrupted']=True;break
        if slot is not None: self.focus=slot
        if self.part==2 and next_state=='playing':
            if self.single:
                self.single.reset_hold()
            else:
                # Rearm after the interruption has been answered, giving both
                # players a full response window rather than expiring mid-speech.
                self.game.state='countdown';self.game.clear_pair();self.game.last_sample=None
        if source=='hand motion' or self.part==3:
            self.state='correction';self.part=3
            if slot is not None: self.loser=slot


def decide(llm, job):
    """One free-form reply; simple session rules handle game entry and exit."""
    context = dict(person=job['name'],visual_cues=job.get('cues',[]),
                   detected_object=job.get('objects'),game_results=job.get('game',{}),
                   session_phase=job.get('phase','chat'))
    system = (
        'You are Furhat, a friendly robot in a natural English conversation. '
        'Respond to what the person just said and follow their topic. Keep replies conversational, '
        'usually one to three short sentences. You can be curious, playful and use your own words. '
        'Visual context is uncertain; do not invent objects, ownership or feelings. '
        'Rock, paper, scissors is the available interactive game. The session controller runs its setup, '
        'countdowns, hand capture and scoring. Do not invent countdowns, moves or scores, '
        'or claim another game has started. '
        'Speak as the listener, not as the person. Return only the words to say aloud, without JSON or role labels. '
        'Context: '+json.dumps(context,ensure_ascii=False))
    if job.get('offer_game'):
        system += (' After acknowledging their answer, smoothly invite them to play a few rounds of rock, paper, scissors. '
                   'Connect the invitation to what they said if possible. End by asking whether they would like to play; '
                   'make that the only question in this reply and wait for their answer before starting.')
    elif job.get('game_invitation'):
        system += ' They did not accept the game invitation. Follow what they said and keep chatting without asking again now.'
    else:
        system += ' Stay in conversation; the session will tell you when it is time to suggest playing.'
    if HAND_CUE in context['visual_cues']:
        system += " If near-face repetitive hand motion is relevant, ask them to lower their hand without claiming contact."
    last_response=next((row for row in reversed(job.get('history',[])) if row.get('person')=='Furhat'),{})
    if last_response.get('interrupted'):
        system += ' Your last reply was interrupted. Answer the new turn and revise your answer as needed; do not simply finish the old reply.'
    messages = [dict(role='system',content=system)]
    history = job.get('history',[])[-8:]
    if history and history[-1].get('person') == job['name'] and history[-1].get('speech') == job.get('speech',''):
        history = history[:-1]
    for row in history:
        messages.append(dict(role='assistant' if row.get('person')=='Furhat' else 'user',content=row['speech']))
    messages.append(dict(role='user',content=job.get('speech','')))
    return 'chat',clean_reply(llm.chat_text(messages),max_sentences=3)
