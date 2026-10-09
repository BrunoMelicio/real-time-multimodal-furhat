"""Offline Part 3 checks with synthetic cues and mocked audio/SDK; no inference."""
from concurrent.futures import Future
import json
from pathlib import Path
from queue import Queue
import tempfile
from threading import Event
from time import perf_counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from furhat_interaction.reflection import CueHistory,BargeEvidence,game_memory,ReflectionLLM,HandNearFace,HAND_CUE,reassuring_reply,reflective_reply,acknowledgment_action


class ReflectionPolicies(unittest.TestCase):
    def test_hand_tracking_runs_by_default_even_without_a_face(self):
        from furhat_interaction.reflection_vision import ReflectionVision
        vision=ReflectionVision.__new__(ReflectionVision)
        vision.timestamp=-1;vision.last_face=None
        vision.cv2=SimpleNamespace(COLOR_BGR2RGB=1,cvtColor=lambda frame,_:frame)
        vision.mp=SimpleNamespace(Image=lambda **kw:kw,ImageFormat=SimpleNamespace(SRGB=1))
        vision.face=SimpleNamespace(detect_for_video=lambda *_:SimpleNamespace(face_landmarks=[]))
        from unittest.mock import Mock
        detect=Mock(return_value=SimpleNamespace(hand_landmarks=[]))
        vision.hand=SimpleNamespace(detect_for_video=detect)
        vision.analyze(np.zeros((100,100,3),dtype=np.uint8),1.)
        detect.assert_called_once()

    def test_motion_diagnostics_explain_missing_or_insufficient_evidence(self):
        policy=HandNearFace();box=(100,100,200,200)
        policy.update(box,[],0.)
        self.assertEqual(policy.diagnostics()['hands_visible'],0)
        for i in range(10): policy.update(box,self.hand(150,150),i*.08)
        diagnostic=policy.diagnostics()
        self.assertTrue(diagnostic['near_face'])
        self.assertEqual(diagnostic['reversals'],0)
        self.assertFalse(diagnostic['triggered'])
        self.assertEqual(diagnostic['near_frames'],10)

    def test_acknowledgment_does_not_require_yes_or_loop_on_continue(self):
        for text in ('Okay.', 'Continue.', 'actually', 'I understand.', 'No, I am not stupid.', "Don't stop."):
            # Negative self-talk is separately routed to reassurance.
            expected='support' if 'stupid' in text else 'encourage'
            self.assertEqual(acknowledgment_action(text,[]),expected)

    def test_acknowledgment_routes_safety_and_explicit_pause(self):
        self.assertEqual(acknowledgment_action("I'm stupid. I'm stupid.",[]),'support')
        self.assertEqual(acknowledgment_action('Okay.',[HAND_CUE]),'support')
        for text in ('Please stop.', 'Can we pause?', "Let's take a break."):
            self.assertEqual(acknowledgment_action(text,[]),'pause')

    def test_barge_requires_two_fresh_high_matches(self):
        gate=BargeEvidence(.65)
        self.assertFalse(gate.update(.9,1,True))
        self.assertTrue(gate.update(.8,1.2,True))
        self.assertFalse(gate.update(.2,1.4,True))
        self.assertFalse(gate.update(.9,1.6,True))
        self.assertFalse(gate.update(.9,3,True))
        self.assertFalse(gate.update(.9,3.2,False))

    def test_cues_are_sustained_observations_not_emotions(self):
        cues=CueHistory()
        for i in range(20): cues.calibrate((0,0,0),(0,0),i*.05)
        for i in range(12): seen=cues.add((0,15,0),(0,.08),{},1+i*.05)
        self.assertIn('head lowered relative to starting posture',seen)
        self.assertIn('iris gaze estimate shifted downward',seen)
        context=cues.context(1,2)
        self.assertFalse(any(word in ' '.join(context) for word in ('sad','depress','frustrated')))
        self.assertEqual(cues.context(10,11),[])

    def test_neutral_head_pitch_offset_is_calibrated_out(self):
        cues=CueHistory()
        for i in range(20): cues.calibrate((0,22,0),(0,.1),i*.05)
        for i in range(12): seen=cues.add((0,22,0),(0,.1),{},1+i*.05)
        self.assertEqual(seen,[])

    def test_hand_cue_reply_is_gentle_and_does_not_claim_impact(self):
        result=reassuring_reply('An unrelated reply.',[HAND_CUE],'I am so stupid.')
        self.assertIn('lower your hand',result)
        self.assertIn('does not make you stupid',result)
        self.assertIn("If you're hitting yourself, please stop; it could hurt you.",result)
        self.assertNotIn('?',result)
        self.assertNotIn('stupid',reassuring_reply('Okay.',[HAND_CUE],''))
        self.assertEqual(reassuring_reply('An ordinary reply.',[],''),'An ordinary reply.')

    def test_head_acknowledgment_only_if_observed(self):
        text='It is okay to lose.'
        self.assertEqual(reflective_reply(text,[]),text)
        self.assertIn('looking down',reflective_reply(text,['head lowered relative to starting posture']))

    @staticmethod
    def hand(x,y):
        return [dict(points=np.tile([x,y],(21,1)))]

    def test_repeated_motion_near_face_fires_once(self):
        policy=HandNearFace();box=(100,100,200,200);events=[]
        for i,x in enumerate((180,170,160,150,140,150,160,170,180,170,160,150)):
            if policy.update(box,self.hand(x,150),i*.08): events.append(i)
        self.assertEqual(len(events),1)

    def test_stationary_touch_single_pass_and_distant_motion_do_not_fire(self):
        for positions in ([150]*20,list(range(120,181,4)),[350,330,310,330,350,330,310]*2):
            policy=HandNearFace()
            for i,x in enumerate(positions):
                self.assertFalse(policy.update((100,100,200,200),self.hand(x,150),i*.08))

    def test_head_translation_does_not_fake_hand_motion(self):
        policy=HandNearFace()
        for i,offset in enumerate((0,10,20,10,0,10,20,10,0)):
            self.assertFalse(policy.update((100+offset,100,200+offset,200),self.hand(150+offset,150),i*.08))

    def test_latest_completed_game_only(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'text_output/single_person/game'
            for date,rounds in (('20260101',[{}, {}, {}]),('20260102',[])):
                run=folder/date;run.mkdir(parents=True)
                (run/'session.json').write_text(json.dumps(dict(name='Bruno',rounds=rounds,scores=dict(person=1,furhat=2))))
            memory=game_memory(temp)
            self.assertEqual(memory['scores']['furhat'],2)
            self.assertEqual(len(memory['rounds']),3)

    def test_prompt_includes_correction_and_game(self):
        model=ReflectionLLM('test','http://unused');payloads=[]
        def request(route,payload):
            payloads.append(payload)
            return dict(message=dict(content='Losing does not make you stupid. Would you like a break?'))
        model.request=request
        model.respond('support','Bruno','Actually I was upset.',dict(scores=dict(person=1,furhat=2)),
                      [dict(role='person',text='I was fine.')],['head lowered relative to starting posture'])
        payload=payloads[0]
        self.assertIn('new answer replaces the old',payload['messages'][0]['content'])
        self.assertIn('Actually I was upset.',payload['messages'][1]['content'])
        self.assertIn('head lowered',payload['messages'][1]['content'])
        self.assertNotIn('hand_safety_context',json.loads(payload['messages'][1]['content']))
        model.respond('support','Bruno','I am so stupid.',{},[],[HAND_CUE])
        context=json.loads(payloads[-1]['messages'][1]['content'])
        safety=context['hand_safety_context']
        self.assertFalse(safety['contact_confirmed'])
        self.assertIn("If you're hitting yourself, please stop",safety['required_response'])
        self.assertIn('conditional wording',payloads[-1]['messages'][0]['content'])

    def run_speech(self,barge):
        from furhat_interaction.single_person.reflection import speech_worker
        stop,interrupted=Event(),Event()
        mode,epoch=SimpleNamespace(value=0),SimpleNamespace(value=1)
        after,started=SimpleNamespace(value=0.),SimpleNamespace(value=0.)
        future=Future()
        if not barge: future.set_result(None)
        if barge:
            original_done=future.done
            def done():
                mode.value=3
                return original_done()
            future.done=done
        received=[]
        class Events:
            def put(self,event):
                received.append(event)
                if event['type'] in ('stage','interrupted','error'): stop.set()
        robot=SimpleNamespace(client=SimpleNamespace(_loop=None,async_client=SimpleNamespace(
            request_speak_text=lambda *a,**kw:'fake-speech-coroutine')))
        jobs=Queue();jobs.put(dict(epoch=1,text='Let us talk about the game.',next='correction',interruptible=True))
        with patch('furhat_interaction.single_person.reflection.asyncio.run_coroutine_threadsafe',return_value=future):
            speech_worker(robot,None,jobs,Events(),stop,interrupted,mode,epoch,after,started)
        return received,future,mode

    def test_interruption_cancels_wait_and_does_not_advance(self):
        events,future,mode=self.run_speech(True)
        self.assertTrue(future.cancelled())
        self.assertTrue(any(e['type']=='interrupted' for e in events))
        self.assertFalse(any(e['type']=='stage' for e in events))
        self.assertEqual(mode.value,3)

    def test_normal_finish_opens_listening(self):
        events,future,mode=self.run_speech(False)
        self.assertTrue(any(e['type']=='stage' and e['stage']=='correction' for e in events))
        self.assertEqual(mode.value,1)

    def run_audio_capture(self,forced=False):
        from furhat_interaction.reflection_audio import audio_worker
        stop=Event();events=Queue();jobs=Queue()
        mode,epoch=SimpleNamespace(value=2),SimpleNamespace(value=7)
        class VAD:
            def __init__(self,*args): self.count=0
            def reset_states(self): self.count=0
            def is_speech(self,*args):
                self.count+=1
                return self.count<=50
        class Stream:
            def __init__(self,**kwargs): self.callback=kwargs['callback']
            def __enter__(self):
                timing=SimpleNamespace(currentTime=0,inputBufferAdcTime=0)
                for i in range(75):
                    self.callback(np.full((512,1),i/1000,dtype='float32'),512,timing,False)
                return self
            def __exit__(self,*args): pass
        class Jobs:
            def put(self,job,**kwargs):
                jobs.put(job)
                if job['final']: stop.set()
            def put_nowait(self,job): jobs.put(job)
        fake_sd=SimpleNamespace(InputStream=Stream)
        with patch.dict('sys.modules',{'sounddevice':fake_sd}), \
             patch('furhat_interaction.microphone.select_microphone',return_value=(0,dict(name='fake mic'))), \
             patch('furhat_interaction.audio.vad.SileroVAD',VAD):
            audio_worker('fake','vad',.6,.65,stop,mode,epoch,SimpleNamespace(value=0.),
                         SimpleNamespace(value=perf_counter()-2),Queue(),Jobs(),events,
                         SimpleNamespace(value=7) if forced else None)
        all_events=[]
        while not events.empty(): all_events.append(events.get_nowait())
        self.assertTrue(any(e['type']=='barge' for e in all_events))
        self.assertFalse(any(e['type']=='error' for e in all_events))
        final=jobs.get_nowait()
        self.assertTrue(final['final'])
        self.assertEqual(final['epoch'],7)
        self.assertGreater(len(final['audio']),50*512)
        np.testing.assert_allclose(final['audio'][:512],0.)  # Initial speech onset survived.
        return all_events

    def test_barge_audio_is_retained_for_transcription(self):
        self.run_audio_capture()

    def test_visual_interrupt_retains_audio_without_waiting_for_asd(self):
        events=self.run_audio_capture(True)
        self.assertTrue(any(e['type']=='barge' and e['guard']=='hand_motion' for e in events))


if __name__=='__main__': unittest.main()
