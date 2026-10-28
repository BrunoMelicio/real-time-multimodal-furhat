"""Ownership across observed rosters, missing pose data and game scoring."""
from collections import deque
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
import numpy as np

from furhat_interaction.group.hand_ownership import assign_hands,body_owners
from furhat_interaction.group.referee_game import RefereeGame
from furhat_interaction.game import Match
from furhat_interaction.multi_person.vision import Vision


PEOPLE=[dict(id=1,box=(0,0,300,480)),dict(id=2,box=(340,0,640,480))]


def pose(center):
    points=[NS(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
    for index,x,y in ((0,center,.12),(11,center-.1,.28),(12,center+.1,.28),
                       (15,center-.12,.6),(16,center+.12,.6)):
        points[index]=NS(x=x,y=y,visibility=.95,presence=.95)
    return points


def hand(x,y,label='Victory'):
    return dict(wrist=(x,y),points=np.tile([x,y],(21,1)),label=label,score=.95)


class HandOwnership(unittest.TestCase):
    def test_single_person_owns_hands_even_without_pose_wrists(self):
        hands=assign_hands([hand(100,280)],{},people=PEOPLE[:1])
        self.assertEqual(hands[0]['person_id'],1)
        self.assertIn('region',hands[0]['reason'])

    def test_single_person_keeps_two_hands_so_game_rejects_ambiguous_signs(self):
        hands=assign_hands([hand(100,280),hand(230,280)],{},people=PEOPLE[:1])
        self.assertEqual([h['person_id'] for h in hands],[1,1])
        game=Match()
        for at in (0.,.1,.2,.4): self.assertIsNone(game.observe(hands,at))
        self.assertFalse(game.results)

    def test_invalid_hand_coordinates_are_not_assigned(self):
        self.assertIsNone(assign_hands([hand(float('nan'),50)],{},people=PEOPLE[:1])[0]['person_id'])

    def test_two_person_ownership_uses_geometry_not_detection_order(self):
        bodies=body_owners([pose(.77),pose(.23)],PEOPLE,640,480)
        hands=assign_hands([hand(.35*640,.6*480),hand(.65*640,.6*480)],bodies)
        self.assertEqual([h['person_id'] for h in hands],[1,2])

    def test_two_person_mode_does_not_guess_when_wrists_are_missing(self):
        self.assertIsNone(assign_hands([hand(100,280)],{})[0]['person_id'])

    def test_missing_pose_wrists_use_clear_regions_for_both_people(self):
        hands=assign_hands([hand(120,280),hand(480,280)],{},people=PEOPLE)
        self.assertEqual([h['person_id'] for h in hands],[1,2])

    def test_roster_changes_do_not_change_remaining_hand_ownership(self):
        # Detection order, participant count and missing skeletons can all vary.
        for people in (PEOPLE[:1],PEOPLE,PEOPLE[::-1],PEOPLE[1:],[],PEOPLE[:1]):
            for bodies in ({},body_owners([pose(.23),pose(.77)],people,640,480)):
                hands=assign_hands([hand(120,288),hand(480,288)],bodies,people=people)
                expected=[i if any(p['id']==i for p in people) else None for i in (1,2)]
                self.assertEqual([h['person_id'] for h in hands],expected)

    def test_region_fallback_rejects_overlaps_and_unrelated_distant_hands(self):
        overlapping=[dict(id=1,box=(0,0,400,480)),dict(id=2,box=(200,0,640,480))]
        self.assertIsNone(assign_hands([hand(300,280)],{},people=overlapping)[0]['person_id'])
        self.assertIsNone(assign_hands([hand(600,280)],{},people=PEOPLE[:1])[0]['person_id'])

    def test_clear_wrist_evidence_takes_priority_over_person_region(self):
        bodies={0:dict(person_id=1,wrists={15:(480,280)},radius=80)}
        self.assertEqual(assign_hands([hand(480,280)],bodies,people=PEOPLE)[0]['person_id'],1)

    def test_crossed_wrists_are_rejected_for_two_people(self):
        bodies={0:dict(person_id=1,wrists={15:(320,240)},radius=80),
                1:dict(person_id=2,wrists={16:(322,240)},radius=80)}
        self.assertIsNone(assign_hands([hand(321,240)],bodies,people=PEOPLE)[0]['person_id'])

    def test_duplicate_detections_for_one_group_wrist_are_rejected(self):
        bodies={0:dict(person_id=1,wrists={15:(100,280)},radius=80)}
        self.assertEqual([h['person_id'] for h in assign_hands([hand(100,280),hand(102,281)],bodies)],[None,None])

    def test_overlapping_body_boxes_do_not_produce_a_guessed_owner(self):
        people=[dict(id=1,box=(0,0,640,480)),dict(id=2,box=(0,0,640,480))]
        self.assertFalse(body_owners([pose(.5)],people,640,480))

    def test_two_person_game_scores_three_rounds_from_assigned_hands(self):
        bodies=body_owners([pose(.23),pose(.77)],PEOPLE,640,480)
        game=RefereeGame();game.players=(1,2);game.scores={1:0,2:0};game.state='release'
        for number,(left,right) in enumerate((('Victory','Open_Palm'),('Closed_Fist','Open_Palm'),('Victory','Victory'))):
            start=number*3.
            game.observe((1,2),[],start)
            self.assertEqual(game.observe((1,2),[],start+.5),'countdown')
            game.arm(start+.6)
            hands=assign_hands([hand(.35*640,288,left),hand(.65*640,288,right)],bodies)
            result=None
            for dt in (0.,.15,.3,.45): result=game.observe((1,2),hands,start+.6+dt) or result
            self.assertEqual(result,'result')
            game.feedback_done()
        self.assertEqual(game.state,'finished')
        self.assertEqual(game.scores,{1:1,2:1})
        self.assertEqual(game.draws,1)


class VisionHandIntegration(unittest.TestCase):
    def view(self,capacity,people):
        view=Vision.__new__(Vision)
        view.max_people=capacity;view.profile=False;view.part=3;view.classify_gestures=True
        view.people=people;view.detect_at=0.;view.stamp=-1
        view.cv2=NS(COLOR_BGR2RGB=1,cvtColor=lambda frame,_:frame)
        view.mp=NS(Image=lambda **kw:kw,ImageFormat=NS(SRGB=1))
        view.face=NS(detect_for_video=lambda *_:NS(face_landmarks=[]))
        view.head={i:Mock() for i in (1,2)}
        view.motion={i:Mock(update=Mock(return_value=False)) for i in (1,2)}
        view.pose=NS(detect_for_video=lambda *_:NS(pose_landmarks=[]))
        points=[NS(x=.2,y=.5) for _ in range(21)]
        result=NS(hand_landmarks=[points],gestures=[[NS(category_name='Victory',score=.95)]])
        view.gesture=NS(recognize_for_video=lambda *_:result);view.objects=None
        return view

    def test_live_vision_assigns_single_person_hands_without_a_pose(self):
        view=self.view(1,PEOPLE[:1])
        obs=view.observe(np.zeros((480,640,3),dtype=np.uint8),0.)
        self.assertEqual(obs['hands'][0]['person_id'],1)

    def test_live_group_vision_assigns_the_person_present_before_the_joiner(self):
        view=self.view(2,PEOPLE[:1])
        obs=view.observe(np.zeros((480,640,3),dtype=np.uint8),0.)
        self.assertEqual(obs['hands'][0]['person_id'],1)

    def test_same_vision_pipeline_handles_a_joiner_and_departure(self):
        for capacity in (1,2):
            view=self.view(capacity,PEOPLE[:1]);frame=np.zeros((480,640,3),dtype=np.uint8)
            rosters=(PEOPLE[:1],PEOPLE,PEOPLE[1:],PEOPLE[:1]) if capacity==2 else (PEOPLE[:1],[],PEOPLE[:1])
            for index,people in enumerate(rosters):
                view.people=people
                obs=view.observe(frame,index*.04)
                self.assertEqual(obs['hands'][0]['person_id'],1 if PEOPLE[0] in people else None)

    def test_no_visible_person_leaves_the_hand_unassigned(self):
        view=self.view(1,[])
        obs=view.observe(np.zeros((480,640,3),dtype=np.uint8),0.)
        self.assertIsNone(obs['hands'][0]['person_id'])

    def test_unnamed_assigned_hand_is_labelled_person_in_the_preview(self):
        import cv2
        view=self.view(1,PEOPLE[:1]);frame=np.zeros((480,640,3),dtype=np.uint8)
        obs=view.observe(frame,0.)
        view.cv2=cv2;view.part=1;view.head_signal={1:deque(),2:deque()};view.head_event={}
        with patch.object(cv2,'putText',wraps=cv2.putText) as text:
            view.render(frame,obs,{},None,None,'Preview',[])
        labels=[call.args[1] for call in text.call_args_list]
        self.assertIn('Person 1: Victory',labels)
        self.assertNotIn('Unassigned: Victory',labels)


if __name__=='__main__': unittest.main()
