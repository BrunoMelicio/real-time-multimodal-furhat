"""Offline scene/ownership regression checks; no hardware or neural inference."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from furhat_interaction.multi_person.policies import Seats,speaker_choice,freeze_speaker,object_owner,bind_saved_names,HeadMotion
from furhat_interaction.multi_person.scenes import Scene
from furhat_interaction.multi_person.runtime import reply,run,attention_target,dialogue
from furhat_interaction.multi_person.audio import AudioInbox,monitoring,capture
from furhat_interaction.reflection import HAND_CUE


def people():
    return [dict(id=1,box=[20,20,260,470]),dict(id=2,box=[370,20,630,470])]


def observation():
    return dict(people=people(),hands=[],head_events={})


class MultiPolicies(unittest.TestCase):
    def test_rules_look_at_each_named_person_then_neutral(self):
        scene=Scene(2,{1:'Bruno',2:'Maria'});scene.first=1;scene.second=2
        scene.rule_answers={2:False,1:True};scene.rule_sources={2:'head shake',1:'speech'}
        job=scene.explain()
        self.assertEqual([s['look_at'] for s in job['segments']],[1,2,None,None])
        self.assertIn('Bruno knows',job['segments'][0]['text'])
        self.assertIn('Maria, I saw your head shake',job['segments'][1]['text'])
        scene.stage('release',1.)
        self.assertIsNone(scene.focus)
        self.assertIsNone(attention_target(2,'release',1,2))
        self.assertIsNone(attention_target(2,'playing',2,1))
        self.assertEqual(attention_target(2,'rules',2,1),2)

    def test_round_result_looks_at_winner_then_resets_forward(self):
        scene=Scene(2,{1:'Bruno',2:'Maria'});scene.first=1;scene.second=2
        scene.game.scores={1:0,2:1};scene.game.last_result={'moves':{1:'rock',2:'paper'},'winner_id':2}
        scene.game.feedback_done=Mock(return_value='release')
        job=scene.result()
        self.assertEqual([s['look_at'] for s in job['segments']],[None,2])
        self.assertIsNone(scene.focus)
        self.assertEqual(job['segments'][1]['text'],'Maria wins this round.')
        scene.stage('release',2.)
        self.assertIsNone(scene.focus)

    def test_pitch_wrap_during_shake_cannot_be_a_nod(self):
        import numpy as np
        detector=HeadMotion();events=[]
        for i in range(45):
            yaw=24*np.sin(i/44*4*np.pi)
            pitch=179+2*np.sin(i/44*4*np.pi)
            pitch=(pitch+180)%360-180
            answer=detector.update((yaw,pitch,0),i*.05)
            if answer is not None: events.append(answer)
        self.assertIn(False,events)
        self.assertNotIn(True,events)

    def test_wrapped_pitch_jitter_is_not_head_movement(self):
        detector=HeadMotion()
        for i in range(30): self.assertIsNone(detector.update((0,179 if i%2 else -179,0),i*.05))

    def test_real_nod_survives_pitch_wrap(self):
        import numpy as np
        detector=HeadMotion();events=[]
        for i in range(45):
            pitch=(179+16*np.sin(i/44*4*np.pi)+180)%360-180
            answer=detector.update((1,pitch,0),i*.05)
            if answer is not None: events.append(answer)
        self.assertIn(True,events);self.assertNotIn(False,events)

    def test_incidental_pitch_during_larger_yaw_is_not_nod(self):
        import numpy as np
        detector=HeadMotion();events=[]
        for i in range(45):
            phase=i/44*4*np.pi
            answer=detector.update((25*np.sin(phase),9*np.sin(phase),0),i*.05)
            if answer is not None: events.append(answer)
        self.assertIn(False,events);self.assertNotIn(True,events)

    def test_edge_object_still_associates_with_unique_nearby_person(self):
        self.assertEqual(object_owner([260,200,300,250],people()),1)

    def test_missing_object_waits_for_detection_instead_of_generic_question(self):
        scene=Scene(1);scene.first=1;scene.second=2;scene.state='name2'
        job=scene.transcript(2,'My name is Maria.',[],None)
        self.assertEqual(job['next'],'wait_object')
        self.assertNotIn('bring with you today',job['text'])
        scene.stage('wait_object',1.)
        vision=SimpleNamespace(object_for=lambda *_:'bottle')
        job=scene.observe(observation(),2.,vision,True)
        self.assertIn("you're holding a bottle",job['text']);self.assertIn('drinking',job['text'])

    def test_seat_survives_short_tracker_id_change(self):
        seats=Seats();a=[dict(id=100,box=[20,20,260,470])]
        self.assertEqual(seats.update(a,0.,640)[0]['id'],1)
        a[0]['id']=900
        self.assertEqual(seats.update(a,.2,640)[0]['id'],1)

    def test_distant_track_does_not_inherit_known_name(self):
        seats=Seats();seats.update(people(),0.,640)
        self.assertEqual(seats.update([dict(id=9,box=[285,0,355,80])],.2,640),[])

    def test_no_identity_recovery_after_long_absence(self):
        seats=Seats();seats.update(people(),0.,640)
        self.assertEqual(seats.update(people(),5.,640),[])

    def test_ambiguous_speaker_is_unknown(self):
        self.assertIsNone(speaker_choice({1:.9,2:.87}))
        self.assertEqual(speaker_choice({1:.9,2:.2}),1)

    def test_speaker_frozen_from_turn_evidence(self):
        votes=[1,1,1,2]
        frozen=freeze_speaker(votes)
        votes.extend([2]*10)
        self.assertEqual(frozen,1)
        self.assertIsNone(freeze_speaker([1,2]))
        self.assertIsNone(freeze_speaker([1]))

    def test_object_unique_box_association(self):
        self.assertEqual(object_owner([400,200,450,250],people()),2)
        self.assertIsNone(object_owner([280,200,350,250],people()))
        self.assertIsNone(object_owner([100,200,150,250],people()+[dict(id=3,box=[0,0,300,480])]))

    def test_saved_names_bind_by_camera_geometry_not_track_id(self):
        saved=[dict(name='Bruno',x=.22),dict(name='Llian',x=.78)]
        p=people();p[0]['id']=2;p[1]['id']=1
        self.assertEqual(bind_saved_names(p,saved,640),{2:'Bruno',1:'Llian'})

    def test_newcomer_waits_for_first_reply_to_finish(self):
        scene=Scene(1);scene.first=1;scene.names={1:'Bruno'};scene.state='day'
        job=scene.transcript(1,'My day was busy.',[],None)
        self.assertEqual(job['kind'],'day')
        vision=SimpleNamespace(seats=SimpleNamespace(stable=lambda *_:True))
        self.assertIsNone(scene.observe(observation(),2.,vision,True))
        scene.stage(job['next'],3.)
        welcome=scene.observe(observation(),3.,vision,True)
        self.assertIn('joined us',welcome['text'])
        self.assertEqual(welcome['target'],2)

    def test_newcomer_name_then_object_question(self):
        scene=Scene(1);scene.first=1;scene.second=2;scene.state='name2'
        job=scene.transcript(2,'My name is Llian.',[],'bottle')
        self.assertEqual(scene.names[2],'Llian')
        self.assertIn('bottle',job['text']);self.assertIn('drinking',job['text'])

    def test_original_person_first_even_in_right_seat(self):
        scene=Scene(2,{1:'Llian',2:'Bruno'},first_name='Bruno')
        job=scene.start(observation())
        self.assertEqual(job['target'],2)

    def test_rules_wait_for_both_answers(self):
        scene=Scene(2,{1:'Bruno',2:'Llian'});scene.start(observation());scene.state='rules'
        self.assertIsNone(scene.transcript(1,'Yes.',[],None))
        self.assertEqual(scene.rule_answers,{1:True})
        obs=observation();obs['head_events']={2:False}
        job=scene.observe(obs,2.,None,True)
        self.assertIn('Bruno knows',job['text']);self.assertIn('Llian',job['text'])
        self.assertIn('V sign',job['text'])

    def test_rules_reprompt_does_not_erase_first_answer(self):
        scene=Scene(2);scene.rule_answers={1:True}
        scene.stage('rules',1.)
        self.assertEqual(scene.rule_answers,{1:True})

    def test_wrong_speaker_cannot_take_anothers_name(self):
        scene=Scene(1);scene.first=1;scene.state='name1'
        self.assertIsNone(scene.transcript(2,'My name is Llian.',[],None))
        self.assertEqual(scene.names,{})

    def test_three_real_rounds_second_person_wins(self):
        scene=Scene(2,{1:'Bruno',2:'Llian'});scene.first=1;scene.second=2
        scene.rule_answers={1:True,2:False};scene.explain()
        pairs=[('Victory','Open_Palm'),('Open_Palm','Victory'),('Victory','Closed_Fist')]
        for n,(a,b) in enumerate(pairs):
            now=n*3.;scene.stage('release',now)
            game=scene.game
            game.observe([1,2],[],now)
            self.assertEqual(game.observe([1,2],[],now+.5),'countdown')
            scene.stage('playing',now+.6)
            hands=[dict(person_id=1,label=a,score=.95),dict(person_id=2,label=b,score=.95)]
            event=None
            for t in (now+.7,now+.9,now+1.15): event=game.observe([1,2],hands,t) or event
            self.assertEqual(event,'result');job=scene.result()
        self.assertEqual(scene.memory['winner'],'Llian')
        self.assertEqual(scene.memory['scores'],{'Bruno':1,'Llian':2})
        self.assertEqual(job['next'],'complete')

    def test_part3_uses_winner_then_other_person(self):
        scene=Scene(3,{1:'Bruno',2:'Llian'},dict(winner='Llian'))
        job=scene.start(observation());self.assertEqual(job['target'],2)
        scene.stage('winner_answer',1.)
        job=scene.transcript(2,'It was great.',[],None)
        self.assertEqual(job['target'],1);self.assertEqual(job['next'],'loser_answer')

    def test_part3_cue_is_not_sent_as_another_persons_cue(self):
        scene=Scene(3,{1:'Bruno',2:'Llian'},dict(winner='Llian'));scene.start(observation())
        scene.state='correction';scene.focus=1
        self.assertIsNone(scene.transcript(2,'I am fine.',[HAND_CUE],None))
        job=scene.transcript(1,'I am stupid.',[HAND_CUE],None)
        self.assertEqual(job['target'],1);self.assertIn(HAND_CUE,job['cues'])

    def test_llm_knows_it_was_referee_and_receives_safety_context(self):
        llm=SimpleNamespace(model='test',request=Mock(return_value=dict(message=dict(content='Please lower your hand.'))))
        reply(llm,dict(kind='support',name='Bruno',speech='I am stupid.',game=dict(winner='Llian'),history=[],cues=[HAND_CUE]))
        payload=llm.request.call_args.args[1]
        self.assertIn('You did not play',payload['messages'][0]['content'])
        context=json.loads(payload['messages'][1]['content'])
        self.assertFalse(context['hand_safety_context']['contact_confirmed'])

    def test_day_answer_uses_plain_conversation_before_game(self):
        llm=SimpleNamespace(model='test',request=Mock(return_value=dict(message=dict(content="I'm glad to hear that!"))))
        result=reply(llm,dict(kind='day',name='Bruno',speech="It's been quite good actually.",
                             game={},history=[dict(speech='My name is Bruno.')],cues=[]))
        payload=llm.request.call_args.args[1]
        messages=payload['messages']
        self.assertEqual(messages[-1],dict(role='user',content="It's been quite good actually."))
        self.assertNotIn('after their game',messages[0]['content'])
        self.assertNotIn('referee',messages[0]['content'])
        self.assertIn('Do not repeat',messages[0]['content'])
        self.assertEqual(result,"I'm glad to hear that!")
        llm.request.assert_called_once()

    def test_day_only_removes_generated_outer_quotes(self):
        llm=SimpleNamespace(model='test',request=Mock(return_value=dict(message=dict(content='"That sounds like a good day!"'))))
        self.assertEqual(reply(llm,dict(kind='day',name='Bruno',speech='Good.')),'That sounds like a good day!')


class MultiExit(unittest.TestCase):
    def test_unread_camera_queue_does_not_hang_interpreter_exit(self):
        import subprocess,sys
        code='''
import multiprocessing as mp
from furhat_interaction.multi_person.runtime import release_queues
q=mp.get_context('spawn').Queue(24)
for i in range(16): q.put_nowait(b'x'*65536)
release_queues(q)
print('queue released',flush=True)
'''
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('queue released',result.stdout)

    def test_esc_exit_writes_no_run_files(self):
        from contextlib import ExitStack
        from pathlib import Path
        from queue import Queue
        from threading import Event
        # All hardware/network/model owners are replaced. Exercise the actual
        # startup -> Esc -> finally path, including the real dialogue thread.
        class Channel(Queue):
            def cancel_join_thread(self): pass
            def close(self): pass
        context=Mock()
        context.Event.side_effect=Event
        context.Value.side_effect=lambda kind,value:SimpleNamespace(value=value)
        context.Queue.side_effect=Channel
        context.Process.return_value.exitcode=None
        context.Process.return_value.is_alive.return_value=False
        vision=Mock();vision.cv2.waitKey.return_value=27
        camera=Mock();camera.error=None;camera.snapshot.return_value=None
        with ExitStack() as stack:
            stack.enter_context(patch('furhat_interaction.multi_person.runtime.mp.get_context',return_value=context))
            stack.enter_context(patch('furhat_interaction.multi_person.runtime.Ollama'))
            stack.enter_context(patch('furhat_interaction.multi_person.vision.Vision',return_value=vision))
            stack.enter_context(patch('furhat_interaction.vision.Camera',return_value=camera))
            stack.enter_context(patch('furhat_interaction.group.robot.Robot'))
            stack.enter_context(patch('builtins.print'))
            mkdir=stack.enter_context(patch.object(Path,'mkdir'))
            write=stack.enter_context(patch.object(Path,'write_text'))
            opened=stack.enter_context(patch.object(Path,'open'))
            self.assertEqual(run(1,[]),0)
            mkdir.assert_not_called();write.assert_not_called();opened.assert_not_called()


class MultiAudio(unittest.TestCase):
    def test_overloaded_inbox_keeps_latest_and_reports_lost_packets(self):
        inbox=AudioInbox(2)
        for packet in range(4): inbox.put(packet)
        packet,lost=inbox.get()
        self.assertEqual(packet,3);self.assertEqual(lost,3)
        inbox.put(4)
        self.assertEqual(inbox.get(),(4,0))

    def test_listening_gate_preserves_interruptions(self):
        self.assertFalse(monitoring(0,100,0))
        self.assertFalse(monitoring(1,100,101))
        self.assertTrue(monitoring(1,100,99))
        self.assertTrue(monitoring(2,100,101))
        self.assertTrue(monitoring(3,100,101))

    def test_muted_robot_speech_runs_no_vad_or_asd(self):
        import numpy as np
        from pathlib import Path
        from queue import Queue
        stop=Mock();stop.is_set.side_effect=[False,True]
        events=Queue();crops=Queue();jobs=Queue()
        state=lambda v:SimpleNamespace(value=v)
        def stream(**kwargs):
            context=Mock()
            def enter():
                kwargs['callback'](np.zeros((512,1),dtype='float32'),512,
                    SimpleNamespace(currentTime=10.,inputBufferAdcTime=9.968),None)
            context.__enter__=Mock(side_effect=enter);context.__exit__=Mock(return_value=False)
            return context
        with patch('sounddevice.InputStream',side_effect=stream), patch(
                'furhat_interaction.microphone.select_microphone',return_value=(0,{'name':'fake'})), patch(
                'furhat_interaction.audio.vad.SileroVAD') as vad, patch(
                'furhat_interaction.active_speaker.ActiveSpeakerModel') as asd, patch.object(Path,'exists',return_value=True):
            capture('fake',stop,state(0),state(1),state(0),state(0),state(0),crops,jobs,events)
        vad.return_value.is_speech.assert_not_called()
        asd.return_value.score.assert_not_called()
        output=list(events.queue)
        self.assertTrue(any(row['type']=='ready' for row in output),output)
        self.assertFalse(any(row['type']=='error' for row in output),output)


class MultiAttention(unittest.TestCase):
    def test_rules_attention_precedes_each_original_sdk_speech_call(self):
        import asyncio
        from concurrent.futures import Future
        from queue import Queue,Empty
        from threading import Event
        from unittest.mock import AsyncMock
        scene=Scene(2,{1:'Bruno',2:'Maria'});scene.first=1;scene.second=2
        scene.rule_answers={1:True,2:False};scene.rule_sources={2:'head shake'}
        job=scene.explain();job['epoch']=1
        jobs=Queue();jobs.put(job);events=Queue();stop=Event()
        client=SimpleNamespace(request_speak_text=AsyncMock())
        robot=SimpleNamespace(client=SimpleNamespace(async_client=client,_loop=None))
        def schedule(coro,loop):
            future=Future();future.set_result(asyncio.run(coro));return future
        def next_job(timeout):
            if not jobs.empty(): return Queue.get(jobs)
            stop.set();raise Empty
        state=lambda v:SimpleNamespace(value=v)
        with patch.object(jobs,'get',side_effect=next_job),patch(
                'furhat_interaction.multi_person.runtime.asyncio.run_coroutine_threadsafe',side_effect=schedule):
            dialogue(robot,None,jobs,events,stop,state(0),state(1),state(0.),state(0.))
        output=list(events.queue)
        self.assertEqual([e['target'] for e in output if e['type']=='attention'],[1,2,None,None])
        for i,e in enumerate(output):
            if e['type']=='attention': self.assertEqual(output[i+1]['type'],'robot')
        self.assertEqual(client.request_speak_text.await_count,4)
        for call in client.request_speak_text.await_args_list:
            self.assertEqual(call.kwargs,dict(wait=True,abort=True))
        self.assertEqual(output[-1]['type'],'stage');self.assertEqual(output[-1]['stage'],'release')


if __name__=='__main__': unittest.main()
if __name__=='__main__': unittest.main()
