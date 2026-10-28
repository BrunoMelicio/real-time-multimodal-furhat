"""Provider configuration and HTTP contracts; no real keys, network or hardware."""
import argparse
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from furhat_interaction.dialogue import Ollama
from furhat_interaction.llm import CompatibleChat, add_llm_arguments, create_llm, load_environment


class Configuration(unittest.TestCase):
    def setUp(self):
        environment=patch.dict(os.environ,{},clear=True)
        environment.start();self.addCleanup(environment.stop)

    def args(self,*flags):
        parser=argparse.ArgumentParser();add_llm_arguments(parser)
        return parser.parse_args(flags)

    def test_default_and_legacy_options_keep_local_ollama(self):
        llm=create_llm(self.args())
        self.assertIsInstance(llm,Ollama)
        self.assertEqual(llm.model,'llama3.2:1b')
        llm=create_llm(self.args('--model=custom','--ollama-url=http://localhost:1234'))
        self.assertEqual((llm.model,llm.url),('custom','http://localhost:1234'))

    def test_groq_default_uses_its_key_and_model(self):
        os.environ['GROQ_API_KEY']='test-key'
        llm=create_llm(self.args('--llm=groq'))
        self.assertEqual(llm.model,'openai/gpt-oss-20b')
        self.assertEqual(llm.url,'https://api.groq.com/openai/v1')

    def test_missing_key_fails_with_location_hint(self):
        with self.assertRaisesRegex(ValueError,'GROQ_API_KEY.*.env'):
            create_llm(self.args('--llm=groq'))

    def test_openai_and_other_compatible_endpoints(self):
        os.environ.update(OPENAI_API_KEY='openai-test',CUSTOM_KEY='compatible-test')
        llm=create_llm(self.args('--llm=openai','--llm-model=chosen-model'))
        self.assertEqual(llm.url,'https://api.openai.com/v1')
        llm=create_llm(self.args('--llm=compatible','--llm-model=other-model',
                                 '--llm-base-url=http://localhost:1234/v1','--llm-key-env=CUSTOM_KEY'))
        self.assertEqual(llm.model,'other-model')
        self.assertEqual(llm.key_env,'CUSTOM_KEY')

    def test_explicit_cli_overrides_environment_defaults(self):
        os.environ.update(LLM_PROVIDER='groq',LLM_MODEL='env-model')
        llm=create_llm(self.args('--llm=ollama','--llm-model=cli-model'))
        self.assertIsInstance(llm,Ollama)
        self.assertEqual(llm.model,'cli-model')

    def test_generic_endpoint_and_model_must_be_configured(self):
        os.environ.update(LLM_API_KEY='test-key',OPENAI_API_KEY='test-key')
        with self.assertRaisesRegex(ValueError,'Select a model'):
            create_llm(self.args('--llm=openai'))
        with self.assertRaisesRegex(ValueError,'base-url'):
            create_llm(self.args('--llm=compatible','--llm-model=test'))
        with self.assertRaisesRegex(ValueError,'ollama-url'):
            create_llm(self.args('--llm=openai','--llm-model=test','--ollama-url=http://localhost'))

    def test_dotenv_is_checkout_specific_and_does_not_override_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'.env').write_text('GROQ_API_KEY="file-test-key"\nLLM_PROVIDER=groq\n')
            os.environ['GROQ_API_KEY']='shell-test-key'
            with patch('furhat_interaction.llm.ROOT',root):
                load_environment()
            self.assertEqual(os.environ['GROQ_API_KEY'],'shell-test-key')
            self.assertEqual(create_llm(self.args()).provider,'groq')


class HTTPContract(unittest.TestCase):
    def client(self,provider='groq',model='openai/gpt-oss-20b'):
        return CompatibleChat(provider,model,'https://api.example/v1','fake-secret','TEST_KEY')

    def test_groq_json_request_and_separate_reasoning(self):
        response=Mock()
        response.__enter__=Mock(return_value=io.StringIO(json.dumps({'choices':[
            {'finish_reason':'stop','message':{'content':'{"action":"chat","reply":"Hello"}',
                                               'reasoning':'not spoken'}}]})))
        response.__exit__=Mock(return_value=False)
        with patch('furhat_interaction.llm.urlopen',return_value=response) as opened:
            text=self.client().chat_json([{'role':'user','content':'Say hello in JSON.'}])
        request=opened.call_args.args[0]
        self.assertEqual(request.full_url,'https://api.example/v1/chat/completions')
        self.assertEqual(request.get_header('Authorization'),'Bearer fake-secret')
        payload=json.loads(request.data)
        self.assertEqual(payload['response_format'],{'type':'json_object'})
        self.assertEqual(payload['reasoning_effort'],'low')
        self.assertFalse(payload['include_reasoning'])
        self.assertNotIn('reasoning_format',payload)
        self.assertNotIn('options',payload)
        self.assertEqual(json.loads(text)['reply'],'Hello')

    def test_openai_payload_has_no_ollama_or_groq_parameters(self):
        llm=self.client('openai','chosen-model')
        llm.request=Mock(return_value={'choices':[{'message':{'content':'{}'}}]})
        llm.chat_json([])
        payload=llm.request.call_args.args[1]
        self.assertIn('max_completion_tokens',payload)
        for key in ('temperature','include_reasoning','reasoning_effort','options','keep_alive'):
            self.assertNotIn(key,payload)

    def test_catalog_and_empty_or_truncated_response_validation(self):
        llm=self.client();llm.request=Mock(return_value={'data':[{'id':llm.model}]})
        llm.check()
        llm.request.return_value={'data':[]}
        with self.assertRaisesRegex(RuntimeError,'not available'): llm.check()
        for response in ({'choices':[]},{'choices':[{'message':{'content':None}}]},
                         {'choices':[{'finish_reason':'length','message':{'content':'{}'}}]}):
            llm.request.return_value=response
            with self.assertRaises(ValueError): llm.chat_json([])

    def test_errors_do_not_expose_provider_body_or_key(self):
        error=HTTPError('https://api.example',401,'fake-secret',{},io.BytesIO(b'fake-secret'))
        with patch('furhat_interaction.llm.urlopen',side_effect=error):
            with self.assertRaises(RuntimeError) as result: self.client().request('/models')
        self.assertIn('TEST_KEY',str(result.exception))
        self.assertNotIn('fake-secret',str(result.exception))
        with patch('furhat_interaction.llm.urlopen',side_effect=URLError('fake-secret')):
            with self.assertRaises(RuntimeError) as result: self.client().request('/models')
        self.assertNotIn('fake-secret',str(result.exception))

    def test_ollama_adapter_keeps_native_json_contract(self):
        llm=Ollama('local-test','http://unused')
        llm.request=Mock(return_value={'message':{'content':'{}'}})
        self.assertEqual(llm.chat_json([]),'{}')
        route,payload=llm.request.call_args.args
        self.assertEqual(route,'/api/chat')
        self.assertEqual(payload['format'],'json')

    def test_check_llm_does_not_start_hardware(self):
        from furhat_interaction.session.runtime import run
        llm=Mock(provider='groq',model='test')
        llm.chat_json.return_value='{"ready":true}'
        with patch('furhat_interaction.session.runtime.load_environment'), \
             patch('furhat_interaction.session.runtime.create_llm',return_value=llm), \
             patch('furhat_interaction.session.runtime.mp.get_context') as context, \
             patch('builtins.print'):
            self.assertEqual(run(1,['--llm=groq','--check-llm']),0)
            context.assert_not_called()


if __name__=='__main__': unittest.main()
