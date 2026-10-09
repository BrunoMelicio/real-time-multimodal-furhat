"""Offline policy checks only: no microphone, camera, SDK or model inference."""
import unittest
import numpy as np
from furhat_interaction.dialogue import Arrival, VisualHistory, Utterance, extract_name, clean_reply, object_acknowledgment, Ollama


class Policies(unittest.TestCase):
    def test_name(self):
        for text in ('My name is Bruno.', "I'm Bruno", 'Bruno'):
            self.assertEqual(extract_name(text),'Bruno')
        self.assertIsNone(extract_name('I am fine'))

    def test_arrival_and_absence(self):
        policy = Arrival()
        self.assertEqual(policy.update((10,10,50,70),100,100,1),(False,False))
        # Feed continuously; a long missing interval must reset arrival.
        for at in (1.2,1.4,1.6,1.8,2.,2.2,2.4):
            present, still = policy.update((10,10,50,70),100,100,at)
        self.assertTrue(present and still)
        self.assertEqual(policy.update(None,100,100,3),(False,False))
        self.assertEqual(policy.update((10,10,50,70),100,100,3.1),(False,False))

    def test_movement_delays_settling(self):
        policy = Arrival()
        policy.update((10,10,50,70),100,100,1)
        policy.update((20,10,60,70),100,100,1.2)
        self.assertFalse(policy.update((20,10,60,70),100,100,1.4)[1])

    def test_context_is_from_speech_not_decode_time(self):
        history = VisualHistory()
        for at in (1.,1.2,1.4):
            history.add(at,[dict(label='bottle',score=.8),dict(label='person',score=.9)])
        for at in (3.,3.2):
            history.add(at,[dict(label='book',score=.9)])
        self.assertEqual([r['object'] for r in history.context(1,1.5)],['bottle'])
        self.assertEqual(history.context(8,9),[])

    def test_recent_object_survives_drinking_pause(self):
        history = VisualHistory()
        for at in (1.,1.2,1.4):
            history.add(at,[dict(label='bottle',score=.8)])
        context = history.context(3.,3.8)
        self.assertEqual(context[0]['object'],'bottle')
        self.assertIn('before',context[0]['observation'])
        self.assertEqual(history.summary(3.,3.8),{'bottle':3})

    def test_object_safeguard(self):
        context = [dict(object='bottle')]
        text, fallback = object_acknowledgment('Glad your day went well. How is the weather?',context)
        self.assertTrue(fallback)
        self.assertEqual(text,'Glad your day went well. I noticed a bottle; what are you drinking?')
        text = 'Glad your day went well. What is in that bottle?'
        self.assertEqual(object_acknowledgment(text,context),(text,False))
        self.assertEqual(object_acknowledgment('Glad to hear it.',[]),('Glad to hear it.',False))

    def test_prompt_contains_explicit_observation(self):
        model = Ollama('test','http://unused')
        requests = []
        def fake_request(route,payload):
            requests.append(payload)
            return {'message':{'content':'Glad your day went well. What is in that bottle?'}}
        model.request = fake_request
        model.reply('Bruno','Quite good.',[dict(object='bottle',observation='visible shortly before the speaking turn')])
        messages = requests[0]['messages']
        self.assertIn('explicitly mentions the bottle',messages[0]['content'])
        self.assertIn('Furhat saw a bottle',messages[-1]['content'])
        self.assertNotIn('holding a bottle',messages[-1]['content'])

    def test_one_detection_not_enough(self):
        history = VisualHistory()
        history.add(1,[dict(label='cup',score=.99)])
        self.assertEqual(history.context(0,2),[])

    def test_silence_does_not_start_turn(self):
        policy = Utterance()
        for i in range(50):
            self.assertIsNone(policy.feed(np.zeros(512,dtype='float32'),i*.032,False))
        self.assertEqual(policy.turn,0)

    def test_partial_final_and_next_turn(self):
        policy = Utterance(); jobs = []
        for i in range(90):
            job = policy.feed(np.ones(512,dtype='float32'),(i+1)*.032,i<60)
            if job:
                jobs.append(job)
        self.assertTrue(any(not j['final'] for j in jobs))
        self.assertEqual(sum(j['final'] for j in jobs),1)
        self.assertAlmostEqual(jobs[-1]['end'],60*.032)
        for i in range(4):
            policy.feed(np.ones(512,dtype='float32'),3+i*.032,True)
        self.assertEqual(policy.turn,2)

    def test_spoken_text(self):
        self.assertEqual(clean_reply('Furhat: Okay. Nice bottle! Third sentence.'),'Okay. Nice bottle!')
        with self.assertRaises(ValueError):
            clean_reply('{"text":"hello"}')

    def test_preview_is_unmodified_and_geometry_matches(self):
        import cv2
        from furhat_interaction.vision import Vision
        view = Vision.__new__(Vision); view.cv2 = cv2
        frame = np.zeros((240,320,3),dtype=np.uint8)
        frame[:,20:40] = 200
        saved = frame.copy()
        observation = dict(points=None,angles=None,objects=[dict(label='bottle',score=.9,box=(20,30,40,80))])
        clean, diagram = view.render(frame,observation,None,'None','Listening...',[],True)
        np.testing.assert_array_equal(frame,saved)
        np.testing.assert_array_equal(clean,cv2.flip(saved,1))
        self.assertEqual(diagram.shape,(460,320,3))
        # Mirrored box must appear at original-width minus raw x.
        np.testing.assert_array_equal(diagram[30,299],(80,190,255))


if __name__ == '__main__':
    unittest.main()
