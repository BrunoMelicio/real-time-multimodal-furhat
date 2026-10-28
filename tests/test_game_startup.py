"""Game setup failures reproduced from actual interaction transcripts."""
from contextlib import ExitStack
from queue import Queue
from threading import Event
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

from furhat_interaction.session.controller import Session


def obs(ids=(1,2),hands=()):
    return dict(people=[dict(id=i) for i in ids],hands=list(hands),head_events={})


class GameStartup(unittest.TestCase):
    def session(self,capacity=2,ids=(1,2)):
        scene=Session(capacity);scene.names={i:f'Player {i}' for i in ids}
        scene.visible=list(ids);scene.stage('chat',0.)
        return scene

    def rules(self,scene):
        action=scene.begin_game(1);scene.stage(action['next'],0.)

    def test_collective_rules_reply_reaches_real_countdown_and_scoring(self):
        scene=self.session()
        offer=scene.apply_decision('offer_game',1);scene.stage(offer['next'],0.)
        accepted=scene.transcript(1,'Yes, we would like to play.',[],None)
        self.assertEqual(accepted['next'],'rules');scene.stage(accepted['next'],0.)
        action=scene.transcript(1,'Yes, we know the rules.',[],None)
        self.assertIsNotNone(action)
        self.assertEqual(action['next'],'release')
        scene.stage(action['next'],1.)
        self.assertIsNone(scene.observe(obs(),1.,NS(),True))
        countdown=scene.observe(obs(),1.5,NS(),True)
        self.assertEqual(countdown['next'],'playing')
        scene.stage(countdown['next'],2.)
        self.assertEqual(scene.game.state,'playing')
        hands=[dict(person_id=i,label=label,score=.95) for i,label in ((1,'Closed_Fist'),(2,'Victory'))]
        result=None
        for t in (2.,2.15,2.3,2.45):
            result=scene.observe(obs(hands=hands),t,NS(),True) or result
        self.assertIsNotNone(result)
        self.assertEqual(scene.game.scores,{1:1,2:0})

    def test_unrelated_speech_during_rules_cannot_enter_free_form_gameplay(self):
        scene=self.session();self.rules(scene)
        for text in ("I'm so sorry.","That's me. Sorry, it is some new ship."):
            action=scene.transcript(1,text,[],None)
            self.assertIsNotNone(action)
            self.assertIsNotNone(action['text'])
            self.assertEqual(action['next'],'rules')
            scene.stage(action['next'],1.)

    def test_request_to_proceed_during_rules_reaches_the_controller(self):
        for capacity,ids in ((1,(1,)),(2,(1,)),(2,(1,2))):
            scene=self.session(capacity,ids);self.rules(scene)
            action=scene.transcript(1,'We can start playing the game.',[],None)
            self.assertEqual(action['next'],'release')
            self.assertIn('fist',action['text'])

    def test_short_unattributed_acceptance_does_not_need_an_invented_identity(self):
        for capacity,ids in ((1,(1,)),(2,(1,)),(2,(1,2))):
            for text in ('Yes.','Sounds good.'):
                scene=self.session(capacity,ids)
                offer=scene.apply_decision('offer_game',1);scene.stage(offer['next'],0.)
                action=scene.unattributed(text)
                self.assertEqual(action['next'],'rules')
                self.assertEqual(scene.rules_players,ids)
                self.assertEqual(scene.history[-1]['person'],'Unattributed participant')

    def test_unattributed_rules_answer_applies_to_the_group_question(self):
        for text in ('Yes.','Yes, we know the rules.','No, please explain.'):
            scene=self.session();self.rules(scene)
            action=scene.unattributed(text)
            self.assertEqual(action['next'],'release')

    def test_unattributed_general_speech_cannot_supply_a_name_or_start_uninvited(self):
        scene=self.session()
        self.assertIsNone(scene.unattributed('Yes.'))
        self.assertEqual(scene.part,0)
        scene.state='name';scene.focus=1;scene.names={}
        self.assertIsNone(scene.unattributed('My name is William.'))
        self.assertEqual(scene.names,{})

    def test_individual_answer_prompts_the_remaining_player(self):
        scene=self.session();self.rules(scene)
        self.assertIsNone(scene.transcript(1,'Yes, I know them.',[],None))
        prompt=scene.observe(obs(),1.,NS(),True)
        self.assertEqual(prompt['next'],'rules')
        self.assertEqual(prompt['target'],2)
        self.assertIn('Player 2',prompt['text'])
        self.assertEqual(scene.rule_answers,{1:True})

    def test_explicit_play_request_starts_setup_but_game_mentions_do_not(self):
        for text in ('I want us to play a game of rock, paper and scissors.',
                     'We want to play.',"Let's play rock-paper-scissors.",'We can try, we can play.'):
            scene=self.session()
            self.assertEqual(scene.transcript(1,text,[],None)['next'],'rules')
        for text in ('I like games.','I do not want to play games.','I want to play guitar.'):
            scene=self.session()
            self.assertEqual(scene.transcript(1,text,[],None)['next'],'chat')
            self.assertEqual(scene.part,0)

    def test_buffered_nods_are_not_erased_by_a_delayed_unrelated_transcript(self):
        scene=self.session();self.rules(scene)
        scene.observe(dict(obs(),head_events={1:True,2:True},at=.5),.5,NS(),False)
        action=scene.transcript(2,"That's me. Sorry, it is some new ship.",[],None)
        self.assertEqual(action['next'],'release')
        self.assertEqual(scene.rule_sources,{1:'nod',2:'nod'})

    def test_negative_answer_explains_without_waiting_for_everyone(self):
        scene=self.session();self.rules(scene)
        action=scene.transcript(1,"No, I don't know the rules.",[],None)
        self.assertEqual(action['next'],'release')
        self.assertIn('fist',action['text'])
        self.assertNotIn(2,scene.rule_answers)

    def test_pending_individual_answer_is_not_reprompted_every_frame(self):
        scene=self.session();self.rules(scene)
        scene.transcript(1,'Yes.',[],None)
        prompt=scene.observe(obs(),1.,NS(),True);scene.stage(prompt['next'],2.)
        self.assertIsNone(scene.observe(obs(),2.1,NS(),True))

    def test_decline_after_an_invitation_keeps_the_conversation(self):
        scene=self.session();offer=scene.apply_decision('offer_game',1);scene.stage(offer['next'],0.)
        action=scene.unattributed('No thanks.')
        self.assertEqual(action['next'],'chat')
        self.assertEqual(scene.part,0)
        self.assertIsNone(scene.game_offer)


class RuntimeGroupResponses(unittest.TestCase):
    def test_anonymous_game_answers_pass_through_the_actual_runtime(self):
        from furhat_interaction.session.runtime import run
        for capacity,ids in ((1,(1,)),(2,(1,)),(2,(1,2))):
            for pending in ('invitation','rules'):
                with self.subTest(capacity=capacity,ids=ids,pending=pending),ExitStack() as stack:
                    class Channel(Queue):
                        def cancel_join_thread(self): pass
                        def close(self): pass
                    context=Mock();context.Event.side_effect=Event
                    context.Value.side_effect=lambda kind,value:NS(value=value)
                    channels=[Channel(),Channel(),Channel()];context.Queue.side_effect=channels
                    context.Process.return_value.exitcode=None
                    context.Process.return_value.is_alive.return_value=False
                    scene=Session(capacity);scene.names={i:f'Player {i}' for i in ids};scene.visible=list(ids)
                    action=scene.begin_game(1) if pending=='rules' else scene.apply_decision('offer_game',1)
                    scene.stage(action['next'],0.)
                    channels[2].put(dict(type='transcript',epoch=0,person=None,turn=1,text='Yes.',
                                         start=1.,end=1.2,final=True,decode_ms=10.))
                    vision=Mock();vision.cv2.waitKey.return_value=27
                    vision.head={i:NS(metrics={}) for i in (1,2)}
                    camera=Mock();camera.error=None;camera.snapshot.return_value=None
                    stack.enter_context(patch('furhat_interaction.session.runtime.mp.get_context',return_value=context))
                    stack.enter_context(patch('furhat_interaction.session.runtime.load_environment'))
                    stack.enter_context(patch('furhat_interaction.session.runtime.create_llm'))
                    stack.enter_context(patch('furhat_interaction.session.runtime.Session',return_value=scene))
                    threads=stack.enter_context(patch('furhat_interaction.session.runtime.Thread'))
                    stack.enter_context(patch('furhat_interaction.multi_person.vision.Vision',return_value=vision))
                    stack.enter_context(patch('furhat_interaction.vision.Camera',return_value=camera))
                    stack.enter_context(patch('furhat_interaction.group.robot.Robot'))
                    printed=stack.enter_context(patch('builtins.print'))
                    self.assertEqual(run(capacity,[]),0,str(printed.call_args_list))
                    queued=threads.call_args.kwargs['args'][2].get_nowait()
                    self.assertEqual(queued['next'],'release' if pending=='rules' else 'rules')
                    self.assertNotIn('repeat',queued['text'])
                    self.assertTrue(any('[GROUP_RESPONSE]' in str(call) for call in printed.call_args_list))


if __name__=='__main__': unittest.main()
