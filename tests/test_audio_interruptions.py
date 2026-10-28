"""Exercise the continuous audio worker with timed audio/video, no real devices."""
from queue import Queue,Empty
from threading import Event
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
import numpy as np

from furhat_interaction.multi_person.audio import capture


class AudioInterruptions(unittest.TestCase):
    def run_capture(self,capacity=1,guard='visual',scores=None,voiced_frames=50,
                    face_ids=None,mode_value=2,forced_id=0):
        stop=Event();clock=NS(value=10.);mode=NS(value=mode_value)
        events=Queue();jobs=Queue()
        ids=tuple(range(1,capacity+1)) if face_ids is None else face_ids
        score=scores or (lambda at:np.array([[.9]*len(ids)]))

        class Inbox:
            def __init__(self): self.index=0
            def get(self,timeout):
                if self.index==80:
                    stop.set();raise Empty
                self.index+=1;clock.value=10+self.index*.032
                return (clock.value,np.full(512,self.index/1000,dtype=np.float32),''),0

        class Crops:
            def __init__(self): self.index=0
            def get_nowait(self):
                at=9.3+self.index*.04
                if at>clock.value: raise Empty
                self.index+=1
                return at,{i:np.full((112,112),i,dtype=np.uint8) for i in ids}

        class VAD:
            def __init__(self,*args): self.frames=0
            def reset_states(self): pass
            def is_speech(self,*args):
                self.frames+=1
                return self.frames<=voiced_frames

        class Jobs:
            def put(self,job,**kwargs):
                jobs.put(job)
                if job['final']: stop.set()
            def put_nowait(self,job): jobs.put(job)

        stream=Mock();stream.__enter__=Mock(return_value=stream);stream.__exit__=Mock(return_value=False)
        asd=Mock();asd.score.side_effect=lambda *args:score(clock.value)
        with patch('sounddevice.InputStream',return_value=stream), \
             patch('furhat_interaction.microphone.select_microphone',return_value=(0,{'name':'test mic'})), \
             patch('furhat_interaction.audio.vad.SileroVAD',VAD), \
             patch('furhat_interaction.active_speaker.ActiveSpeakerModel',return_value=asd) as model, \
             patch('furhat_interaction.multi_person.audio.AudioInbox',Inbox), \
             patch('furhat_interaction.multi_person.audio.perf_counter',side_effect=lambda:clock.value), \
             patch('pathlib.Path.exists',return_value=True):
            capture('test',stop,mode,NS(value=7),NS(value=0.),NS(value=9.),NS(value=forced_id),
                    Crops(),Jobs(),events,capacity,guard)
        output=list(events.queue)
        self.assertFalse([e for e in output if e['type']=='error'],output)
        return output,list(jobs.queue),model,asd

    def check_utterance(self,events,jobs,person):
        barges=[e for e in events if e['type']=='barge']
        self.assertEqual(len(barges),1)
        self.assertEqual(barges[0]['person'],person)
        final=[j for j in jobs if j['final']]
        self.assertEqual(len(final),1)
        self.assertEqual(final[0]['person'],person)
        self.assertEqual(final[0]['epoch'],7)
        # The interrupted utterance keeps its opening sounds, not only its tail.
        np.testing.assert_allclose(final[0]['audio'][:512],.001)
        self.assertGreaterEqual(len(final[0]['audio']),50*512)
        return barges[0]

    def test_single_person_vad_interrupts_and_retains_speech_onset(self):
        events,jobs,model,_=self.run_capture(guard='vad')
        barge=self.check_utterance(events,jobs,1)
        self.assertEqual(barge['source'],'VAD')
        self.assertLess(barge['audio_at']-10.,.25)
        model.assert_not_called()

    def test_single_person_default_guard_interrupts_with_visible_speech(self):
        events,jobs,model,asd=self.run_capture()
        self.assertEqual(self.check_utterance(events,jobs,1)['source'],'Light-ASD')
        model.assert_called_once();self.assertGreaterEqual(asd.score.call_count,2)

    def test_robot_voice_without_matching_lip_activity_does_not_interrupt(self):
        events,jobs,_,_=self.run_capture(scores=lambda _:np.array([[.1]]))
        self.assertFalse(any(e['type']=='barge' for e in events))
        self.assertFalse(jobs)

    def test_short_noise_burst_does_not_trigger_vad_interruption(self):
        events,jobs,_,_=self.run_capture(guard='vad',voiced_frames=5)
        self.assertFalse(any(e['type']=='barge' for e in events))
        self.assertFalse(jobs)

    def test_two_person_interruption_identifies_the_second_speaker(self):
        events,jobs,_,_=self.run_capture(capacity=2,scores=lambda _:np.array([[.1,.9]]))
        self.check_utterance(events,jobs,2)

    def test_two_person_vad_stops_furhat_before_speaker_matching_finishes(self):
        events,jobs,_,_=self.run_capture(capacity=2,guard='vad',scores=lambda _:np.array([[.1,.9]]))
        barge=[e for e in events if e['type']=='barge'][0]
        self.assertEqual(barge['source'],'VAD')
        self.assertLess(barge['audio_at']-10.,.25)
        self.assertIsNone(barge['person'])
        self.assertEqual([j['person'] for j in jobs if j['final']],[2])

    def test_ambiguous_two_person_speech_does_not_stop_the_robot(self):
        events,jobs,_,_=self.run_capture(capacity=2,scores=lambda _:np.array([[.9,.88]]))
        self.assertFalse(any(e['type']=='barge' for e in events))
        self.assertFalse(jobs)

    def test_two_person_listening_keeps_correct_transcript_owner(self):
        events,jobs,_,_=self.run_capture(capacity=2,mode_value=1,scores=lambda _:np.array([[.9,.1]]))
        self.assertFalse(any(e['type']=='barge' for e in events))
        self.assertEqual([j['person'] for j in jobs if j['final']],[1])

    def test_group_capacity_with_one_present_participant_uses_the_same_listening_path(self):
        for capacity,ids in ((1,(1,)),(2,(1,)),(2,(2,))):
            events,jobs,_,asd=self.run_capture(capacity=capacity,face_ids=ids,mode_value=1,
                                               scores=lambda _:np.array([[.1]]))
            self.assertFalse(any(e['type']=='barge' for e in events))
            self.assertEqual([j['person'] for j in jobs if j['final']],[ids[0]])
            asd.score.assert_not_called()

    def test_solo_participant_in_group_capacity_still_needs_visual_evidence_to_interrupt(self):
        events,jobs,_,_=self.run_capture(capacity=2,face_ids=(1,),scores=lambda _:np.array([[.1]]))
        self.assertFalse(any(e['type']=='barge' for e in events))
        self.assertFalse(jobs)

    def test_missing_face_evidence_cannot_assign_a_group_interruption(self):
        events,jobs,_,_=self.run_capture(capacity=2,face_ids=())
        self.assertFalse(any(e['type']=='barge' for e in events))
        self.assertFalse(jobs)

    def test_visual_hand_interruption_does_not_wait_for_audio_match(self):
        events,jobs,_,_=self.run_capture(capacity=2,forced_id=2,scores=lambda _:np.array([[.1,.1]]))
        self.assertEqual(self.check_utterance(events,jobs,2)['source'],'hand motion')


if __name__=='__main__': unittest.main()
