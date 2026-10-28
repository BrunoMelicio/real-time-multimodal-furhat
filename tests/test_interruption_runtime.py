"""Furhat stop/listen/update integration for one and two participants."""
import asyncio
from concurrent.futures import Future
from contextlib import ExitStack
from queue import Queue,Empty
from threading import Event
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock,Mock,patch

from furhat_interaction.session.controller import Session,decide
from furhat_interaction.session.runtime import dialogue,run


class SpeechCancellation(unittest.TestCase):
    def run_dialogue(self,after_text=None):
        stop=Event();mode=NS(value=0);jobs=Queue();events=Queue();pending=Future()
        job=dict(text='Original reply.',next='chat',epoch=7,target=1)
        if after_text: job['after_text']=after_text
        jobs.put(job)
        client=NS(request_speak_text=AsyncMock())
        robot=NS(client=NS(async_client=client,_loop=None),say=Mock())
        calls=0
        def schedule(coro,loop):
            nonlocal calls
            calls+=1
            if after_text and calls==1:
                done=Future();done.set_result(asyncio.run(coro));return done
            coro.close();return pending
        def wait(timeout): mode.value=3;return False
        def next_job(timeout):
            if jobs.empty(): stop.set();raise Empty
            return Queue.get(jobs)
        with patch.object(jobs,'get',side_effect=next_job),patch.object(stop,'wait',side_effect=wait), \
             patch('furhat_interaction.session.runtime.asyncio.run_coroutine_threadsafe',side_effect=schedule):
            dialogue(robot,None,jobs,events,stop,mode,NS(value=7),NS(value=0.),NS(value=0.))
        return pending,list(events.queue),robot,calls

    def test_interrupted_reply_cancels_wait_and_never_advances_the_scene(self):
        pending,events,_,_=self.run_dialogue()
        self.assertTrue(pending.cancelled())
        self.assertFalse(any(e['type'] in ('stage','error') for e in events))
        self.assertTrue(next(e for e in events if e['type']=='robot')['interruptible'])

    def test_post_game_followup_speech_is_interruptible_too(self):
        pending,events,robot,calls=self.run_dialogue('How did the game feel?')
        self.assertTrue(pending.cancelled());self.assertEqual(calls,2)
        robot.say.assert_not_called()
        self.assertFalse(any(e['type']=='stage' for e in events))


class StopListenUpdate(unittest.TestCase):
    def test_one_and_two_people_stop_furhat_and_submit_the_new_turn(self):
        for capacity in (1,2):
            with self.subTest(capacity=capacity),ExitStack() as stack:
                class Channel(Queue):
                    def cancel_join_thread(self): pass
                    def close(self): pass
                context=Mock();context.Event.side_effect=Event
                context.Value.side_effect=lambda kind,value:NS(value=value)
                channels=[Channel(),Channel(),Channel()];context.Queue.side_effect=channels
                context.Process.return_value.exitcode=None
                context.Process.return_value.is_alive.return_value=False
                scene=Session(capacity);scene.names={1:'Alice',2:'Bob'};scene.state='busy'
                speaker=capacity
                for event in (
                    dict(type='robot',epoch=0,text='We can meet on Tuesday.',next='chat',interruptible=True),
                    dict(type='barge',epoch=0,person=None,source='VAD'),
                    dict(type='transcript',epoch=0,person=speaker,turn=1,text='Actually, I meant Wednesday.',
                         start=1.,end=2.,final=True,decode_ms=10.)):
                    channels[2].put(event)
                vision=Mock();vision.cv2.waitKey.return_value=27
                vision.head={i:Mock() for i in (1,2)}
                vision.cues={i:NS(context=lambda *_:[]) for i in (1,2)}
                vision.object_for.return_value=None
                camera=Mock();camera.error=None;camera.snapshot.return_value=None
                robot=Mock();robot.client.async_client.request_speak_stop=AsyncMock()
                def schedule(coro,loop):
                    done=Future();done.set_result(asyncio.run(coro));return done
                stack.enter_context(patch('furhat_interaction.session.runtime.mp.get_context',return_value=context))
                stack.enter_context(patch('furhat_interaction.session.runtime.load_environment'))
                stack.enter_context(patch('furhat_interaction.session.runtime.create_llm'))
                stack.enter_context(patch('furhat_interaction.session.runtime.Session',return_value=scene))
                threads=stack.enter_context(patch('furhat_interaction.session.runtime.Thread'))
                stack.enter_context(patch('furhat_interaction.multi_person.vision.Vision',return_value=vision))
                stack.enter_context(patch('furhat_interaction.vision.Camera',return_value=camera))
                stack.enter_context(patch('furhat_interaction.group.robot.Robot',return_value=robot))
                stack.enter_context(patch('furhat_interaction.session.runtime.asyncio.run_coroutine_threadsafe',side_effect=schedule))
                stack.enter_context(patch('builtins.print'))
                self.assertEqual(run(capacity,['--barge-in=vad']),0)
                robot.client.async_client.request_speak_stop.assert_awaited_once()
                queued=threads.call_args.kwargs['args'][2].get_nowait()
                self.assertEqual(queued['speech'],'Actually, I meant Wednesday.')
                self.assertEqual(queued['target'],speaker)
                self.assertEqual(queued['phase'],'chat')
                self.assertTrue(next(r for r in queued['history'] if r['person']=='Furhat')['interrupted'])
                llm=NS(chat_text=Mock(return_value='Wednesday works. What time suits you?'))
                self.assertIn('Wednesday',decide(llm,queued)[1])
                self.assertIn('Your last reply was interrupted',llm.chat_text.call_args.args[0][0]['content'])

    def test_game_interruption_rearms_the_same_round_after_the_new_reply(self):
        scene=Session(2);scene.names={1:'Alice',2:'Bob'};scene.visible=[1,2]
        scene.begin_game(1);scene.game.players=(1,2);scene.game.scores={1:0,2:0}
        scene.game.state='countdown';scene.stage('playing',1.)
        scene.interrupted('playing',2,'VAD',2.)
        self.assertEqual(scene.game.state,'countdown')
        self.assertIsNone(scene.game.tick(20.))
        scene.stage('playing',30.)
        self.assertEqual(scene.game.deadline,36.)
        self.assertEqual(scene.game.round,1)
        self.assertEqual(scene.game.scores,{1:0,2:0})


class HeadResponseRuntime(unittest.TestCase):
    def test_nod_in_post_speech_pause_reaches_game_in_each_roster(self):
        from tests.test_session_responses import observation
        for capacity,ids in ((1,(1,)),(2,(1,)),(2,(1,2))):
            with self.subTest(capacity=capacity,ids=ids),ExitStack() as stack:
                clock=NS(value=10.);frames=NS(value=0);values=[]
                class Channel(Queue):
                    def cancel_join_thread(self): pass
                    def close(self): pass
                class Events(Channel):
                    injected=False
                    def get_nowait(self):
                        if frames.value==1 and not self.injected:
                            self.injected=True
                            values[0].value=1;values[2].value=12.
                            self.put(dict(type='robot',epoch=1,text='Would you like to play?',
                                          next='chat',interruptible=True,at=10.5))
                            self.put(dict(type='stage',epoch=1,stage='chat',at=12.))
                        return super().get_nowait()
                events=Events()
                for worker in ('audio','asr'): events.put(dict(type='ready',worker=worker))
                context=Mock();context.Event.side_effect=Event
                def value(kind,initial):
                    item=NS(value=initial);values.append(item);return item
                context.Value.side_effect=value;context.Queue.side_effect=[Channel(),Channel(),events]
                context.Process.return_value.exitcode=None
                context.Process.return_value.is_alive.return_value=False
                scene=Session(capacity);scene.names={i:f'Player {i}' for i in ids}
                def invite(obs):
                    scene.visible=list(ids)
                    return scene.apply_decision('offer_game',1)
                scene.start=invite
                vision=Mock();vision.head={i:Mock(metrics={}) for i in (1,2)}
                vision.cues={i:Mock() for i in (1,2)}
                for cue in vision.cues.values():
                    cue.diagnostics.return_value={};cue.rebase.return_value=True
                vision.seats.stable.return_value=True;vision.render.return_value=(None,None)
                def observe(frame,at):
                    obs=observation(ids,at,{1:True} if frames.value==2 else {})
                    return dict(obs,crops={},faces={},motion=[],width=640,height=480)
                vision.observe.side_effect=observe
                keys=iter((32,255,27))
                def key(_):
                    result=next(keys)
                    clock.value=10.5 if result==32 else 12.5
                    return result
                vision.cv2.waitKey.side_effect=key
                camera=Mock();camera.error=None
                def snapshot():
                    frames.value+=1
                    return frames.value,clock.value,None
                camera.snapshot.side_effect=snapshot
                robot=Mock();robot.client.async_client.request_face_headpose=AsyncMock()
                def schedule(coro,loop):
                    done=Future();done.set_result(asyncio.run(coro));return done
                stack.enter_context(patch('furhat_interaction.session.runtime.mp.get_context',return_value=context))
                stack.enter_context(patch('furhat_interaction.session.runtime.load_environment'))
                stack.enter_context(patch('furhat_interaction.session.runtime.create_llm'))
                stack.enter_context(patch('furhat_interaction.session.runtime.Session',return_value=scene))
                threads=stack.enter_context(patch('furhat_interaction.session.runtime.Thread'))
                stack.enter_context(patch('furhat_interaction.multi_person.vision.Vision',return_value=vision))
                stack.enter_context(patch('furhat_interaction.vision.Camera',return_value=camera))
                stack.enter_context(patch('furhat_interaction.group.robot.Robot',return_value=robot))
                stack.enter_context(patch('furhat_interaction.session.runtime.asyncio.run_coroutine_threadsafe',side_effect=schedule))
                stack.enter_context(patch('furhat_interaction.session.runtime.perf_counter',side_effect=lambda:clock.value))
                printed=stack.enter_context(patch('builtins.print'))
                self.assertEqual(run(capacity,[]),0,str(printed.call_args_list))
                jobs=threads.call_args.kwargs['args'][2]
                self.assertEqual(jobs.get_nowait()['next'],'chat')
                accepted=jobs.get_nowait()
                self.assertEqual(accepted['next'],'rules')
                self.assertEqual(accepted['response_source'],'nod')
                self.assertEqual(scene.rules_players,ids)


if __name__=='__main__': unittest.main()
