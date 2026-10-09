"""Offline Part 2 checks: synthetic head/hand data, no camera or inference."""
import json
from pathlib import Path
import tempfile
import unittest
from furhat_interaction.game import Match, HeadAnswer, yes_no, remembered_name


class GamePolicies(unittest.TestCase):
    def test_planned_loss_three_rounds(self):
        match=Match()
        for number,label in enumerate(('Closed_Fist','Open_Palm','Victory')):
            before=match.robot_move
            result=None
            for at in (0.,.1,.2,.4):
                result=match.observe([dict(label=label,score=.9)],number+at) or result
                if result: break
            self.assertEqual(result['winner'],'furhat')
            self.assertEqual(result['furhat'],before)
        self.assertEqual(match.scores(),dict(person=0,furhat=3,draw=0))
        self.assertIsNone(match.observe([dict(label='Closed_Fist',score=.9)],10))

    def test_different_move_is_scored_honestly(self):
        match=Match()
        for at in (0.,.1,.2,.4):
            result=match.observe([dict(label='Victory',score=.9)],at)
            if result: break
        self.assertEqual(result['winner'],'person')  # Scissors beats precommitted paper.

    def test_uncertain_or_two_hands_never_score(self):
        for hands in ([],[dict(label='Closed_Fist',score=.3)],
                      [dict(label='Closed_Fist',score=.9)]*2):
            match=Match()
            for at in (0.,.1,.2,.4):
                self.assertIsNone(match.observe(hands,at))
            self.assertFalse(match.results)

    def test_detection_gap_restarts_hold(self):
        match=Match(); hand=[dict(label='Closed_Fist',score=.9)]
        match.observe(hand,0); match.observe(hand,.1)
        self.assertIsNone(match.observe(hand,1))

    def test_hand_release(self):
        match=Match()
        self.assertFalse(match.released([],0))
        self.assertTrue(match.released([],.4))
        self.assertFalse(match.released([{}],.5))
        self.assertFalse(match.released([],.6))

    def test_shake(self):
        policy=HeadAnswer(); answers=[]
        for i,yaw in enumerate((0,6,12,6,0,-6,-12,-6,0)):
            result=policy.update((yaw,0,0),i*.08)
            if result is not None: answers.append(result)
        self.assertEqual(answers,[False])

    def test_nod(self):
        policy=HeadAnswer(); answers=[]
        for i,pitch in enumerate((0,4,8,16,12,8,4,0)):
            result=policy.update((0,pitch,0),i*.08)
            if result is not None: answers.append(result)
        self.assertEqual(answers,[True])

    def test_jitter_and_long_gap_are_not_answers(self):
        policy=HeadAnswer()
        for i in range(25):
            self.assertIsNone(policy.update(((-1)**i,(-1)**i,0),i*.05))
        self.assertIsNone(policy.update((20,0,0),5))

    def test_yes_no(self):
        self.assertIs(yes_no('Yes, please.'),True)
        self.assertIs(yes_no("I don't know how to play."),False)
        self.assertIs(yes_no('Not really'),False)
        self.assertIsNone(yes_no('What do you mean?'))

    def test_name_from_latest_part1(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for run,name in (('20260101','Bruno'),('20260102','Llian')):
                folder=root/'text_output/single_person/introduction'/run
                folder.mkdir(parents=True)
                (folder/'session.json').write_text(json.dumps(dict(name=name)))
            self.assertEqual(remembered_name(root),'Llian')

    def test_wink_follows_completed_speech(self):
        from furhat_interaction.single_person.game import speech_worker
        from threading import Event
        from queue import Queue
        from types import SimpleNamespace
        stop=Event(); calls=[]
        class Robot:
            client=SimpleNamespace(async_client=SimpleNamespace(request_gesture_start=lambda *a,**kw:'wink'))
            def say(self,text): calls.append(('speech_finished',text))
            def call(self,request): calls.append(('gesture',request))
        class Events:
            def put(self,event):
                if event['type']=='stage': stop.set()
        jobs=Queue(); jobs.put(dict(text='I won the game. Yay!',next='complete',wink=True,after_text='Thank you for playing.'))
        mute=SimpleNamespace(value=0.)
        speech_worker(Robot(),jobs,Events(),stop,mute)
        self.assertEqual(calls,[('speech_finished','I won the game. Yay!'),('gesture','wink'),
                                ('speech_finished','Thank you for playing.')])
        self.assertEqual(mute.value,float('inf'))


if __name__=='__main__':
    unittest.main()
