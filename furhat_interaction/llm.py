"""Language providers for the continuous app; perception and ASR stay local."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from furhat_interaction.dialogue import Ollama
from furhat_interaction.paths import ROOT


PROVIDERS = {
    'groq': ('https://api.groq.com/openai/v1', 'GROQ_API_KEY', 'openai/gpt-oss-20b'),
    'openai': ('https://api.openai.com/v1', 'OPENAI_API_KEY', None),
    'compatible': (None, 'LLM_API_KEY', None),
}


def load_environment():
    # Use only this checkout's file. Exported shell variables take precedence.
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)


def add_llm_arguments(parser):
    parser.add_argument('--llm', '--llm-provider', dest='llm',
                        choices=('ollama', *PROVIDERS), default=os.getenv('LLM_PROVIDER', 'ollama'),
                        help='Language provider (default: ollama, or LLM_PROVIDER).')
    parser.add_argument('--llm-model', '--model', dest='llm_model', default=os.getenv('LLM_MODEL'),
                        help='Provider model ID; --model remains an alias.')
    parser.add_argument('--llm-base-url', default=os.getenv('LLM_BASE_URL'),
                        help='Override the provider API base URL; required for compatible.')
    parser.add_argument('--llm-key-env', default=os.getenv('LLM_API_KEY_ENV'),
                        help='Name of the environment variable containing the API key.')
    parser.add_argument('--ollama-url', default=None, help='Legacy Ollama base URL override.')
    parser.add_argument('--check-llm', action='store_true',
                        help='Test provider connectivity and JSON generation without hardware.')


def create_llm(args):
    if args.llm not in ('ollama', *PROVIDERS):
        raise ValueError('LLM_PROVIDER must be ollama, groq, openai, or compatible.')
    if args.ollama_url and args.llm != 'ollama':
        raise ValueError('--ollama-url requires --llm=ollama; use --llm-base-url for other providers.')
    if args.llm == 'ollama':
        return Ollama(args.llm_model or 'llama3.2:1b',
                      args.llm_base_url or args.ollama_url or 'http://127.0.0.1:11434')
    url, key_env, model = PROVIDERS[args.llm]
    key_env = args.llm_key_env or key_env
    key = os.getenv(key_env, '').strip()
    if not key:
        raise ValueError(f'Set {key_env} in the project .env file or export it in your shell.')
    return CompatibleChat(args.llm, args.llm_model or model, args.llm_base_url or url, key, key_env)


class CompatibleChat:
    """OpenAI Chat Completions protocol, shared by Groq and compatible services."""

    def __init__(self, provider, model, base_url, api_key, key_env):
        if not model:
            raise ValueError(f'Select a model with --llm-model for {provider}.')
        parts = urlsplit(base_url or '')
        if (parts.scheme not in ('http', 'https') or not parts.netloc
                or parts.username or parts.password or parts.query or parts.fragment):
            raise ValueError('Set --llm-base-url to an http(s) API base URL without credentials or query parameters.')
        self.provider, self.model = provider, model
        self.url, self._api_key, self.key_env = base_url.rstrip('/'), api_key, key_env

    def request(self, route, payload=None):
        request = Request(self.url + route,
                          data=json.dumps(payload).encode() if payload is not None else None,
                          headers={'Authorization': 'Bearer ' + self._api_key,
                                   'Content-Type': 'application/json', 'User-Agent': 'furhat-interaction'})
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            # Provider error bodies can echo request data. Never log them or the key.
            hint = {401: f'Check {self.key_env}.', 403: 'Check model/account permissions.',
                    404: 'Check the model ID and API base URL.',
                    429: 'Rate limit or quota reached; try again later.'}.get(exc.code,
                    'Check the model and provider API settings.')
            raise RuntimeError(f'{self.provider} HTTP {exc.code}. {hint}') from None
        except URLError:
            raise RuntimeError(f'Cannot connect to {self.provider}; check the API base URL and network.') from None

    def check(self):
        # Some compatible servers implement chat but do not expose a model catalog.
        if self.provider == 'compatible':
            return
        models = self.request('/models').get('data', [])
        if self.model not in {model.get('id') for model in models}:
            raise RuntimeError(f'Model {self.model!r} is not available from {self.provider}.')

    def chat_json(self, messages):
        return self._chat(messages,json_mode=True)

    def chat_text(self, messages):
        return self._chat(messages,json_mode=False)

    def _chat(self, messages, json_mode):
        payload = dict(model=self.model, messages=messages, stream=False)
        if json_mode: payload['response_format']={'type':'json_object'}
        if self.provider == 'compatible':
            payload['max_tokens'] = 2048
        else:
            payload['max_completion_tokens'] = 2048
        if self.provider == 'groq':
            payload['temperature'] = .2 if json_mode else .7
            if self.model.startswith('openai/gpt-oss-'):
                payload.update(reasoning_effort='low', include_reasoning=False)
        data = self.request('/chat/completions', payload)
        choices = data.get('choices') or []
        if not choices or choices[0].get('finish_reason') == 'length':
            raise ValueError('Language provider returned an empty or truncated response.')
        content = choices[0].get('message', {}).get('content')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Language provider returned no spoken content.')
        return content
