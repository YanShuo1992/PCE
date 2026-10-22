from __future__ import annotations

import os

_DEFAULT_CLAUDE_MODEL_OPUS = 'bedrock/global.anthropic.claude-opus-4-5-20251101-v1:0'
_DEFAULT_CLAUDE_MODEL_HAIKU = 'bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0'
_DEFAULT_CLAUDE_MODEL_SONNET = 'bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0'

_NONSTREAMING_MAX_TOKENS = 21333

def get_claude_model():
    env_model = os.getenv('CODING_AGENT_CLAUDE_MODEL')
    if env_model in ('claude_haiku_4.5', 'claude-haiku-4-5'):
        return _DEFAULT_CLAUDE_MODEL_HAIKU
    if env_model in ('claude_sonnet_4.5', 'claude-sonnet-4-5'):
        return _DEFAULT_CLAUDE_MODEL_SONNET
    if env_model in ('claude_opus_4.5', 'claude-opus-4-5'):
        return _DEFAULT_CLAUDE_MODEL_OPUS
    if env_model:
        return env_model
    return _DEFAULT_CLAUDE_MODEL_OPUS

class _ClaudeModel:
    def __str__(self): return get_claude_model()
    def __repr__(self): return f"'{get_claude_model()}'"
    def __eq__(self, other): return str(self) == str(other)
    def __ne__(self, other): return str(self) != str(other)
    def __hash__(self): return hash(str(self))
    def __contains__(self, item): return item in str(self)
    def lower(self): return str(self).lower()
    def upper(self): return str(self).upper()
    def startswith(self, p, s=None, e=None): return str(self).startswith(p, s, e)
    def endswith(self, p, s=None, e=None): return str(self).endswith(p, s, e)
    def split(self, sep=None, maxsplit=-1): return str(self).split(sep, maxsplit)
    def replace(self, old, new, count=-1): return str(self).replace(old, new, count)

CLAUDE_MODEL = _ClaudeModel()
OPENAI_MODEL = 'o3-mini-2025-01-31'
GLM_MODEL = 'glm-4.6'

def coding_max_tokens(default: int) -> int:
    v = os.getenv('CODING_MAX_TOKENS')
    n = int(v) if v and v.isdigit() else default
    if n > _NONSTREAMING_MAX_TOKENS:
        print(f"[MAXTOK] CODING_MAX_TOKENS={n} is above the non-streaming ceiling "
              f"{_NONSTREAMING_MAX_TOKENS}, clamping to it. Uncapped, every call would fail with "
              f"'Streaming is required' and the child would end with an empty patch.")
        n = _NONSTREAMING_MAX_TOKENS
    return n

def thinking_param(model: str):
    if not str(model).startswith('dsclaude/'):
        return None
    v = os.getenv('DEEPSEEK_THINKING', '').strip().lower()
    if v == 'disabled':
        return {"type": "disabled"}
    if v == 'enabled':
        return {"type": "enabled"}
    return None

def provider_model_id(model: str) -> str:
    m = str(model)
    return m.split('dsclaude/', 1)[-1] if m.startswith('dsclaude/') else m

def build_create_kwargs(model, messages, tools, tool_choice) -> dict:
    default = 8192 if 'opus' in str(model).lower() else 4096
    kw = dict(model=provider_model_id(model), messages=messages,
              max_tokens=coding_max_tokens(default), tool_choice=tool_choice, tools=tools)
    th = thinking_param(model)
    if th is not None:
        kw['thinking'] = th
    return kw
