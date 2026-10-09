"""Offline Part 3 cue/response and interruption sequencing checks."""
import asyncio
from concurrent.futures import Future
from queue import Queue,Empty
from threading import Event
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock,Mock,patch
from conference_ready_multi.cues import ReflectionCues,acknowledge_downward
from conference_ready_multi.runtime import dialogue
from conference_ready_multi.scenes import Scene


LOW='head lowered relative to starting posture'


class MultiReflection(unittest.TestCase):
    def test_sustained_head_down_survives_flickering_iris_and_expression(self):
        cues=ReflectionCues();cues.baseline=(0.,0.)
        for i in range(15):
            seen=cues.add((0.,15.,0.),(0.,.06 if i%2 else .02),
                          {'mouthSmileLeft':.8,'mouthSmileRight':.8} if i%2 else {},i*.08)
        self.assertIn(LOW,seen)
        self.assertIn(LOW,cues.context(0.,1.2))
        self.assertNotIn('iris gaze estimate shifted downward',seen)

    def test_space_rebases_from_settled_posture_instead_of_startup(self):
        cues=ReflectionCues();cues.baseline=(25.,.1)
        for i in range(20): cues.add((0.,0.,0.),(0.,0.),{},i*.08)
        self.assertTrue(cues.rebase(1.6));self.assertEqual(cues.baseline,(0.,0.))
        for i in range(10): seen=cues.add((0.,15.,0.),(0.,0.),{},1.7+i*.08)
        self.assertIn(LOW,seen)

    def test_rebase_and_delta_preserve_pitch_wrap(self):
        cues=ReflectionCues()
        for i in range(15): cues.add((0.,179 if i%2 else -179,0.),None,{},i*.08)
        self.assertTrue(cues.rebase(1.2))
        for i in range(10): seen=cues.add((0.,-165.,0.),None,{},1.3+i*.08)
        self.assertIn(LOW,seen)
        self.assertLess(cues.diagnostics(2.)['head_down_delta_deg'],20.)

    def test_no_baseline_or_brief_motion_does_not_claim_looking_down(self):
        cues=ReflectionCues()
        for i in range(10): self.assertEqual(cues.add((0.,30.,0.),None,{},i*.08),[])
        cues.baseline=(0.,0.)
        self.assertEqual(cues.add((0.,15.,0.),None,{},1.),[])
        self.assertEqual(cues.add((0.,15.,0.),None,{},1.5),[])

    def test_generic_feeling_down_is_not_visual_acknowledgment(self):
        text='I am sorry you feel down. Learning takes practice.'
        self.assertTrue(acknowledge_downward(text,[LOW]).startswith('I noticed you looked down.'))
        self.assertEqual(acknowledge_downward(text,[]),text)
        explicit='I noticed you looked down. Losing is okay.'
        self.assertEqual(acknowledge_downward(explicit,[LOW]),explicit)

    def run_dialogue(self,job,interrupt=False):
        jobs=Queue();jobs.put(dict(job,epoch=1));events=Queue();stop=Event()
        mode=NS(value=0);client=NS(request_speak_text=AsyncMock())
        robot=NS(client=NS(async_client=client,_loop=None))
        pending=Future()
        def schedule(coro,loop):
            if interrupt:
                coro.close();return pending
            future=Future();future.set_result(asyncio.run(coro));return future
        def next_job(timeout):
            if not jobs.empty(): return Queue.get(jobs)
            stop.set();raise Empty
        with patch.object(jobs,'get',side_effect=next_job),patch(
                'conference_ready_multi.runtime.asyncio.run_coroutine_threadsafe',side_effect=schedule),patch(
                'conference_ready_multi.runtime.reply',return_value='Sorry you feel down. Losing is okay.'):
            if interrupt:
                def wait(timeout): mode.value=3;return False
                with patch.object(stop,'wait',side_effect=wait):
                    dialogue(robot,None,jobs,events,stop,mode,NS(value=1),NS(value=0.),NS(value=0.))
            else: dialogue(robot,None,jobs,events,stop,mode,NS(value=1),NS(value=0.),NS(value=0.))
        return list(events.queue),client,pending

    def test_actual_speech_gets_visual_acknowledgment_when_observed(self):
        events,client,_=self.run_dialogue(dict(text=None,kind='first',cues=[LOW],next='ack'))
        self.assertIn('I noticed you looked down.',client.request_speak_text.await_args.args[0])
        self.assertEqual(events[-1]['stage'],'ack')

    def test_encouragement_is_armed_and_actual_speech_wait_cancels(self):
        scene=Scene(3,{1:'Bruno',2:'Maria'});scene.loser=1;scene.focus=1;scene.state='ack'
        job=scene.transcript(1,'Okay.',[],None)
        self.assertTrue(job['interruptible'])
        events,_,pending=self.run_dialogue(job,True)
        self.assertTrue(pending.cancelled())
        self.assertTrue(next(e for e in events if e['type']=='robot')['interruptible'])
        self.assertFalse(any(e['type']=='stage' for e in events))


if __name__=='__main__': unittest.main()
