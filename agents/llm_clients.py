"""Provider adapters for the shared complete(messages, tools) interface.

No SDK dependency, automatic retries, simulator ownership or memory selection.
Pass independent clients to independent conversations. Public API defaults follow
DeepSeek Chat Completions and Moonshot Kimi K3 Chat Completions documentation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Mapping


class ModelAPIError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward a provider credential to an unexpected redirect host.
        return None


def _message_key(message: dict) -> str:
    calls = message.get('tool_calls')
    value = {'tool_calls': calls} if calls else {'content': message.get('content')}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ChatCompletionsClient:
    """Transport shared by provider-specific request policies.

    The EEF policy reconstructs assistant messages, losing provider extension
    fields. Restore the exact previously returned message when the tool calls
    (including IDs and arguments), or text-only content, match. Never invent a
    reasoning field. Cached responses stay in memory, not in logs.
    """
    def __init__(self, api_key: str, *, model: str, base_url: str,
                 timeout: float = 90, max_tokens: int = 4096):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError('API credential required')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('explicit model required')
        url = urllib.parse.urlsplit(base_url)
        local = url.hostname in ('localhost', '127.0.0.1', '::1')
        if (url.scheme not in ('http', 'https') or not url.hostname
                or (url.scheme == 'http' and not local) or url.username or url.password
                or url.query or url.fragment):
            raise ValueError('HTTPS base URL required (HTTP allowed only for local tests)')
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('finite positive timeout required')
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('positive integer token budget required')
        self._key, self.model = api_key.strip(), model.strip()
        self.base_url, self.timeout, self.max_tokens = base_url.rstrip('/'), timeout, max_tokens
        self.last_metadata: dict = {}
        self._assistant_messages: dict[str, dict] = {}
        self._opener = urllib.request.build_opener(_NoRedirect())

    def reset_history(self) -> None:
        self._assistant_messages.clear()

    def _messages(self, messages: list) -> list:
        result = copy.deepcopy(messages)
        if not result:
            raise ValueError('messages required')
        retained = {}
        for index, message in enumerate(result):
            if not isinstance(message,dict) or message.get('role') != 'assistant':
                continue
            key = _message_key(message)
            original = self._assistant_messages.get(key)
            if original is not None:
                # Matching tool IDs/arguments bind this to the original response.
                # Caller-supplied provider fields take precedence when present.
                result[index] = dict(copy.deepcopy(original), **message)
                retained[key] = original
        self._assistant_messages = retained
        return result

    def parameters(self) -> dict:
        raise NotImplementedError

    def complete(self, messages: list, tools: list) -> dict:
        self.last_metadata = {}
        payload = dict(model=self.model, messages=self._messages(messages), stream=False,
                       **self.parameters())
        if tools:
            payload['tools'] = copy.deepcopy(tools)
        else:
            payload.pop('tool_choice', None)
            payload.pop('parallel_tool_calls', None)
        request = urllib.request.Request(self.base_url + '/chat/completions',
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
            headers={'Authorization':'Bearer '+self._key, 'Content-Type':'application/json'})
        start = time.monotonic()
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            # Providers may echo prompt/headers in their error body; do not log it.
            code = error.code
            error.close()
            raise ModelAPIError(f'{self.provider} HTTP {code}; request was not retried') from None
        except (OSError, ValueError) as error:
            raise ModelAPIError(f'{self.provider} transport/JSON error: {type(error).__name__}') from None
        try:
            choices = result['choices']
            if not isinstance(choices,list) or len(choices) != 1 or not isinstance(choices[0],dict) or choices[0].get('finish_reason') in ('length','content_filter'):
                raise ValueError('incomplete or non-single response')
            message = choices[0]['message']
            if not isinstance(message,dict) or message.get('role') != 'assistant':
                raise ValueError('assistant response required')
        except (KeyError, TypeError, ValueError, IndexError):
            raise ModelAPIError(f'{self.provider} invalid/incomplete completion; no action authorized') from None
        self._assistant_messages[_message_key(message)] = copy.deepcopy(message)
        self.last_metadata = dict(provider=self.provider, model=result.get('model', self.model),
            response_id=result.get('id'), usage=result.get('usage'), latency_s=time.monotonic()-start)
        return result


class DeepSeekClient(ChatCompletionsClient):
    provider = 'deepseek'

    def __init__(self, api_key: str, *, model: str = 'deepseek-flash',
                 base_url: str = 'https://api.deepseek.com', timeout: float = 55,
                 thinking: bool = False, max_tokens: int = 1800,
                 reasoning_effort: str = 'low'):
        if type(thinking) is not bool or reasoning_effort not in ('low','high','max'):
            raise ValueError('DeepSeek requires boolean thinking and low/high/max effort')
        self.thinking, self.reasoning_effort = thinking, reasoning_effort
        super().__init__(api_key,model=model,base_url=base_url,timeout=timeout,max_tokens=max_tokens)

    def parameters(self) -> dict:
        result = dict(max_tokens=self.max_tokens, parallel_tool_calls=False,
            tool_choice='auto' if self.thinking else 'required',
            thinking={'type':'enabled' if self.thinking else 'disabled'})
        if self.thinking:
            result['reasoning_effort'] = self.reasoning_effort
        return result


class KimiClient(ChatCompletionsClient):
    """Kimi K3-family interface; model aliases remain explicitly configurable.

    K3 always thinks. K2.x requires different parameter policies and must not be
    passed into this adapter as an implicit fallback.
    """
    provider = 'kimi'

    def __init__(self, api_key: str, *, model: str = 'kimi-k3',
                 base_url: str = 'https://api.moonshot.cn/v1', timeout: float = 90,
                 max_tokens: int = 4096, reasoning_effort: str = 'low'):
        if reasoning_effort not in ('low','high','max'):
            raise ValueError('Kimi K3 reasoning_effort must be low/high/max')
        if model.lower().startswith(('kimi-k2','moonshot-v1')):
            raise ValueError('this adapter uses the Kimi K3 parameter contract')
        self.reasoning_effort = reasoning_effort
        super().__init__(api_key,model=model,base_url=base_url,timeout=timeout,max_tokens=max_tokens)

    def parameters(self) -> dict:
        return dict(max_completion_tokens=self.max_tokens, reasoning_effort=self.reasoning_effort,
                    tool_choice='required')


def client_from_env(env: Mapping[str,str]) -> ChatCompletionsClient:
    """Explicit selection. Never print credentials or fall back to another provider."""
    provider = env.get('RSI_LLM_PROVIDER','').strip().lower()
    classes = {'deepseek':DeepSeekClient, 'kimi':KimiClient}
    if provider not in classes:
        raise ValueError('RSI_LLM_PROVIDER must be deepseek or kimi')
    key_name = 'DEEPSEEK_API_KEY' if provider == 'deepseek' else 'MOONSHOT_API_KEY'
    args = dict(timeout=float(env.get('RSI_LLM_TIMEOUT_S','90')),
                max_tokens=int(env.get('RSI_LLM_MAX_TOKENS','4096')),
                reasoning_effort=env.get('RSI_LLM_REASONING_EFFORT','low'))
    for setting, name in [('RSI_LLM_MODEL','model'),('RSI_LLM_BASE_URL','base_url')]:
        if env.get(setting):
            args[name] = env[setting]
    if provider == 'deepseek':
        thinking = env.get('RSI_LLM_THINKING','disabled')
        if thinking not in ('enabled','disabled'):
            raise ValueError('RSI_LLM_THINKING must be enabled or disabled')
        args['thinking'] = thinking == 'enabled'
    return classes[provider](env.get(key_name,''),**args)
