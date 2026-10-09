"""Continuous-session transitions and model decisions, without hardware/inference."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from furhat_interaction.session.controller import Session, decide


def observation(ids=(1, 2), hands=(), head=None):
    return dict(people=[dict(id=i,box=(i*100,0,i*100+80,200)) for i in ids],
                hands=list(hands),head_events=head or {})


class ContinuousSession(unittest.TestCase):
    def session(self, ids=(1,2)):
        s=Session(len(ids));s.names={i:('Alice' if i==1 else 'Bob') for i in ids}
        s.start(observation(ids));s.stage('chat',0)
        return s

    def test_unknown_newcomer_waits_for_current_reply(self):
        s=self.session((1,));s.capacity=2
        vision=SimpleNamespace(seats=SimpleNamespace(stable=lambda *_:True))
        s.state='busy'
        self.assertIsNone(s.observe(observation(),1,vision,True))
        s.stage('chat',2)
        action=s.observe(observation(),2,vision,True)
        self.assertEqual(action['target'],2)
        self.assertEqual(action['next'],'name')

    def test_name_and_object_become_conversation_context(self):
        s=Session();s.state='name';s.focus=2
        action=s.transcript(2,'My name is Bob',[],'bottle')
        self.assertEqual(s.names[2],'Bob')
        self.assertIn('bottle',action['text'])
        self.assertIn('drinking',action['text'])
        s.stage(action['next'],1)
        action=s.transcript(2,'It is water.',['smiling'],'bottle')
        self.assertEqual(action['kind'],'conversation')
        self.assertEqual(action['objects'],'bottle')
        self.assertEqual(action['cues'],['smiling'])

    def test_either_named_person_can_talk(self):
        s=self.session()
        for slot in (2,1):
            action=s.transcript(slot,'How are you?',[],None)
            self.assertEqual(action['target'],slot)
            s.stage(action['next'],1)
        self.assertFalse(s.done)

    def test_semantic_decision_starts_and_stops_game(self):
        s=self.session()
        action=s.apply_decision('start_game',2)
        self.assertIn('judge',action['text'])
        self.assertEqual(s.part,2)
        self.assertEqual(s.rules_players,(1,2))
        s.stage(action['next'],1)
        action=s.apply_decision('stop_game',2)
        self.assertEqual(action['next'],'chat')
        s.stage(action['next'],2)
        self.assertEqual(s.part,0)
        self.assertFalse(s.done)

    def test_head_shake_and_speech_answers_combine(self):
        s=self.session();action=s.begin_game(1);s.stage(action['next'],0)
        self.assertIsNone(s.transcript(1,'Yes',[],None))
        vision=SimpleNamespace()
        action=s.observe(observation(head={2:False}),1,vision,True)
        self.assertIn('head shake',action['text'])
        self.assertEqual(action['next'],'release')

    def test_multi_game_winner_carries_into_reflection(self):
        s=self.session();s.begin_game(1)
        s.game.players=(1,2);s.game.scores={1:0,2:2};s.game.round=3
        s.game.state='playing'
        s.game.last_result=dict(status='valid',moves={1:'rock',2:'paper'},winner_id=2)
        def result(*_):
            s.game.state='feedback';return 'result'
        with patch.object(s.game,'observe',side_effect=result):
            s.state='playing'
            action=s.observe(observation(),2,SimpleNamespace(),True)
        self.assertEqual(action['next'],'after_game')
        self.assertEqual(s.memory['winner'],'Bob')
        self.assertEqual(action['target'],1)
        s.stage(action['next'],3)
        self.assertEqual(s.state,'reflect')
        self.assertFalse(s.done)
        response=s.transcript(1,'It could have been better.',['head lowered relative to starting posture'],None)
        self.assertEqual(response['kind'],'conversation')
        self.assertTrue(response['interruptible'])

    def test_tied_game_still_enters_reflection(self):
        s=self.session();s.begin_game(1);s.game.players=(1,2);s.game.scores={1:1,2:1};s.game.round=3
        s.game.last_result=dict(status='valid',moves={1:'rock',2:'rock'},winner_id=None)
        def result(*_): s.game.state='feedback';return 'result'
        with patch.object(s.game,'observe',side_effect=result):
            s.state='playing';action=s.observe(observation(),2,SimpleNamespace(),True)
        self.assertEqual(action['next'],'after_game')
        self.assertIsNone(s.memory['winner'])

    def test_single_game_commits_moves_and_reflects_without_winner_argument(self):
        s=self.session((1,));s.begin_game(1)
        s.single.results=[dict(winner='person'),dict(winner='furhat'),dict(winner='draw')]
        action=s.single_result(1,dict(person='rock',furhat='rock',winner='draw'))
        self.assertEqual(action['next'],'after_game')
        self.assertIsNone(s.memory['winner'])
        self.assertIn('how did the game feel',action['after_text'])

    def test_second_game_has_fresh_rules_and_scores(self):
        s=self.session();s.begin_game(1);s.rule_answers={1:True,2:False};s.part=3
        s.begin_game(2)
        self.assertEqual(s.rule_answers,{})
        self.assertEqual(s.game.results,[])
        self.assertEqual(s.first,2)

    def test_no_does_not_end_the_session(self):
        s=self.session();action=s.transcript(1,'No, I disagree.',[],None)
        self.assertEqual(action['next'],'chat')
        self.assertFalse(s.done)


class ModelDecisions(unittest.TestCase):
    def job(self,phase='chat'):
        return dict(name='Alice',speech='Could we play something?',phase=phase,
                    game={},history=[],cues=[],objects=None)

    def llm(self,action,reply='That sounds fun.'):
        return SimpleNamespace(model='fake',request=Mock(return_value=dict(message=dict(content=json.dumps(dict(action=action,reply=reply))))))

    def test_llm_controls_intent_without_keyword_matching(self):
        llm=self.llm('start_game')
        self.assertEqual(decide(llm,self.job())[0],'start_game')
        payload=llm.request.call_args.args[1]
        self.assertEqual(payload['format'],'json')

    def test_invalid_output_fails_closed(self):
        with self.assertRaises(ValueError): decide(self.llm('execute_command'),self.job())
        with self.assertRaises(ValueError): decide(self.llm('chat',''),self.job())

    def test_stop_outside_game_and_start_inside_game_are_ignored(self):
        self.assertEqual(decide(self.llm('stop_game'),self.job())[0],'chat')
        self.assertEqual(decide(self.llm('start_game'),self.job('playing'))[0],'chat')


class ContinuousRuntime(unittest.TestCase):
    def test_esc_releases_owners_and_writes_no_files(self):
        from contextlib import ExitStack
        from pathlib import Path
        from queue import Queue
        from threading import Event
        from furhat_interaction.session.runtime import run
        class Channel(Queue):
            def cancel_join_thread(self): pass
            def close(self): pass
        context=Mock();context.Event.side_effect=Event
        context.Value.side_effect=lambda kind,value:SimpleNamespace(value=value)
        context.Queue.side_effect=Channel
        context.Process.return_value.exitcode=None
        context.Process.return_value.is_alive.return_value=False
        vision=Mock();vision.cv2.waitKey.return_value=27
        camera=Mock();camera.error=None;camera.snapshot.return_value=None
        with ExitStack() as stack:
            stack.enter_context(patch('furhat_interaction.session.runtime.mp.get_context',return_value=context))
            stack.enter_context(patch('furhat_interaction.session.runtime.Ollama'))
            factory=stack.enter_context(patch('furhat_interaction.multi_person.vision.Vision',return_value=vision))
            stack.enter_context(patch('furhat_interaction.vision.Camera',return_value=camera))
            robot=stack.enter_context(patch('furhat_interaction.group.robot.Robot'))
            stack.enter_context(patch('builtins.print'))
            mkdir=stack.enter_context(patch.object(Path,'mkdir'))
            write=stack.enter_context(patch.object(Path,'write_text'))
            opened=stack.enter_context(patch.object(Path,'open'))
            self.assertEqual(run(1,[]),0)
            factory.assert_called_once_with(3,'lite0',continuous=True,max_people=1)
            vision.close.assert_called_once();camera.close.assert_called_once()
            robot.return_value.close.assert_called_once()
            mkdir.assert_not_called();write.assert_not_called();opened.assert_not_called()

    def test_game_decision_reaches_controller_before_robot_speech(self):
        from queue import Queue
        from threading import Event
        from furhat_interaction.session.runtime import dialogue
        stop=Event();jobs=Queue();robot=Mock();events=[]
        class Sink:
            def put(self,event):
                events.append(event)
                if event['type']=='decision': stop.set()
        jobs.put(dict(text=None,epoch=7,target=1,kind='conversation',name='Alice',next='chat'))
        value=lambda v:SimpleNamespace(value=v)
        with patch('furhat_interaction.session.runtime.decide',return_value=('start_game','Ready!')):
            dialogue(robot,Mock(),jobs,Sink(),stop,value(0),value(7),value(0.),value(0.))
        self.assertEqual(events,[dict(type='decision',epoch=7,action='start_game',person=1)])
        robot.say.assert_not_called();robot.client.async_client.request_speak_text.assert_not_called()


if __name__=='__main__': unittest.main()
