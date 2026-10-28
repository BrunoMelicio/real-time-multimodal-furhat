"""Timed head responses through the shared conversation/game controller."""
import math
from types import SimpleNamespace as NS
import unittest

from furhat_interaction.multi_person.policies import HeadMotion
from furhat_interaction.session.controller import Session


def observation(ids,at,head=None):
    return dict(people=[dict(id=i,box=(i*100,0,i*100+80,200)) for i in ids],
                hands=[],head_events=head or {},at=at)


VISION=NS(seats=NS(stable=lambda *_:True))


class HeadResponses(unittest.TestCase):
    def session(self,capacity,ids):
        scene=Session(capacity);scene.names={i:f'Player {i}' for i in ids}
        scene.start(observation(ids,0.));scene.stage('chat',0.)
        return scene

    def offer(self,scene,slot):
        action=scene.apply_decision('offer_game',slot)
        scene.robot_speaking(1.)
        scene.history.append(dict(person='Furhat',speech=action['text']))
        return action

    def test_invitation_nod_and_shake_use_same_path_as_spoken_answers(self):
        for capacity,ids in ((1,(1,)),(2,(1,)),(2,(2,)),(2,(1,2))):
            for slot in ids:
                for answer in (True,False):
                    with self.subTest(capacity=capacity,ids=ids,slot=slot,answer=answer):
                        gesture=self.session(capacity,ids);spoken=self.session(capacity,ids)
                        for scene in (gesture,spoken):
                            self.offer(scene,slot);scene.stage('chat',2.)
                        by_head=gesture.observe(observation(ids,2.2,{slot:answer}),2.2,VISION,True)
                        by_voice=spoken.transcript(slot,'Yes.' if answer else 'No.',[],None)
                        self.assertEqual(by_head['next'],by_voice['next'])
                        self.assertEqual(gesture.part,spoken.part)
                        self.assertEqual(gesture.rules_players,spoken.rules_players)
                        self.assertEqual(gesture.game_offer,spoken.game_offer)
                        self.assertEqual(by_head['response_source'],'nod' if answer else 'head shake')
                        self.assertEqual(by_head['next'],'rules' if answer else 'chat')

    def test_response_during_prompt_or_listening_pause_is_not_lost(self):
        for capacity,ids in ((1,(1,)),(2,(1,)),(2,(1,2))):
            for response_at in (1.8,2.1):
                scene=self.session(capacity,ids);self.offer(scene,1)
                if response_at>2.: scene.stage('chat',2.5)
                self.assertIsNone(scene.observe(observation(ids,response_at,{1:True}),response_at,VISION,False))
                self.assertEqual(scene.part,0)
                scene.stage('chat',2.5)
                action=scene.observe(observation(ids,2.6),2.6,VISION,True)
                self.assertEqual(action['next'],'rules')
                self.assertIsNone(scene.observe(observation(ids,2.7),2.7,VISION,False))

    def test_two_players_can_answer_rules_together_or_on_separate_frames(self):
        for together in (True,False):
            scene=self.session(2,(1,2));scene.begin_game(1);scene.robot_speaking(1.)
            scene.stage('rules',2.)
            if together:
                action=scene.observe(observation((1,2),2.1,{1:True,2:False}),2.1,VISION,True)
            else:
                prompt=scene.observe(observation((1,2),2.1,{1:True}),2.1,VISION,True)
                self.assertEqual(prompt['target'],2)
                scene.stage(prompt['next'],2.1)
                action=scene.observe(observation((1,2),2.2,{2:False}),2.2,VISION,True)
            self.assertEqual(scene.rule_answers,{1:True,2:False})
            self.assertEqual(scene.rule_sources,{1:'nod',2:'head shake'})
            self.assertEqual(action['next'],'release')

    def test_spoken_turn_supersedes_a_buffered_gesture(self):
        scene=self.session(2,(1,2));self.offer(scene,1)
        scene.observe(observation((1,2),1.9,{1:True}),1.9,VISION,False)
        scene.stage('chat',2.)
        action=scene.transcript(1,'No thanks.',[],None)
        scene.stage(action['next'],2.2)
        self.assertIsNone(scene.observe(observation((1,2),2.3),2.3,VISION,True))
        self.assertEqual(scene.part,0)

    def test_stale_or_departed_responders_cannot_answer_a_later_question(self):
        for stale in (True,False):
            scene=self.session(2,(1,2));self.offer(scene,1)
            scene.observe(observation((1,2),1.5,{1:True}),1.5,VISION,False)
            scene.stage('chat',2.)
            ids=(1,2) if stale else (2,)
            now=5. if stale else 2.1
            self.assertIsNone(scene.observe(observation(ids,now),now,VISION,True))
            self.assertEqual(scene.part,0)

    def test_new_robot_turn_discards_a_previous_questions_gesture(self):
        scene=self.session(1,(1,));self.offer(scene,1)
        scene.observe(observation((1,),1.5,{1:True}),1.5,VISION,False)
        scene.say('Would you prefer to keep talking?', 'chat',1)
        scene.robot_speaking(2.);scene.game_offer=None;scene.stage('chat',3.)
        self.assertIsNone(scene.observe(observation((1,),3.1),3.1,VISION,True))

    def test_a_bystanders_nod_does_not_accept_an_addressed_invitation(self):
        scene=self.session(2,(1,2));self.offer(scene,1);scene.stage('chat',2.)
        self.assertIsNone(scene.observe(observation((1,2),2.1,{2:True}),2.1,VISION,True))
        self.assertEqual(scene.game_offer,1)
        self.assertEqual(scene.part,0)

    def test_conversational_head_answer_reaches_llm_with_visual_context(self):
        for capacity,ids in ((1,(1,)),(2,(1,)),(2,(1,2))):
            scene=self.session(capacity,ids);scene.focus=1
            scene.history.append(dict(person='Furhat',speech='Did you enjoy your day? I would love to hear.'))
            action=scene.observe(observation(ids,.5,{1:True}),.5,VISION,True)
            self.assertEqual(action['speech'],'Yes.')
            self.assertIn('participant nodded yes',action['cues'])
            self.assertEqual(action['kind'],'conversation')

    def test_incidental_nod_without_a_question_does_not_create_a_turn(self):
        scene=self.session(2,(1,2))
        scene.history.append(dict(person='Furhat',speech='That sounds interesting.'))
        self.assertIsNone(scene.observe(observation((1,2),.5,{1:True}),.5,VISION,True))

    def test_newcomer_does_not_erase_the_present_players_response(self):
        scene=self.session(2,(1,));self.offer(scene,1)
        scene.stage('chat',2.)
        action=scene.observe(observation((1,2),2.1,{1:True}),2.1,VISION,True)
        self.assertEqual(action['next'],'rules')
        self.assertEqual(scene.rules_players,(1,))

    def test_real_head_motion_detection_drives_invitation_at_multiple_frame_rates(self):
        for fps in (4,6,10,20,30):
            for answer in (True,False):
                with self.subTest(fps=fps,answer=answer):
                    scene=self.session(2,(1,));self.offer(scene,1);scene.stage('chat',1.)
                    detector=HeadMotion();action=None
                    for frame in range(int(fps*2)+1):
                        t=frame/fps
                        angle=22*math.sin(2*math.pi*t/1.6)
                        event=detector.update((0,angle,0) if answer else (angle,0,0),1.+t)
                        action=scene.observe(observation((1,),1.+t,{1:event} if event is not None else {}),1.+t,VISION,True)
                        if action: break
                    self.assertIsNotNone(action)
                    self.assertEqual(action['next'],'rules' if answer else 'chat')

    def test_head_detector_does_not_bridge_missing_faces_or_accept_jitter(self):
        detector=HeadMotion()
        for frame in range(30):
            self.assertIsNone(detector.update((frame%2,frame%3,0),frame*.1))
        detector.update(None,4.)
        self.assertIsNone(detector.update((0,22,0),4.1))
        self.assertIsNone(detector.update((0,float('nan'),0),4.2))
        self.assertIsNone(detector.update((0,-22,0),4.3))


if __name__=='__main__': unittest.main()
