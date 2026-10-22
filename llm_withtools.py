import ast
import json
import os
import re
import time
import anthropic
import backoff
import openai
import copy

from llm import create_client, get_response_from_llm
from prompts.tooluse_prompt import get_tooluse_prompt
from tools import load_all_tools

MAX_TRUNCATION_RETRIES = 3

def _truncation_retries(explain: bool = False):
    n = None
    v = os.getenv('SELFIMPROVE_TRUNCATION_RETRIES')
    if v and v.isdigit():
        n, source = int(v), f"SELFIMPROVE_TRUNCATION_RETRIES={v} (self-improvement container)"
    if n is None:
        try:
            import harness_def as _hd
            n = int(_hd.truncation_retry_budget())
            source = f"harness_def.truncation_retry_budget()={n}"
        except Exception:
            n, source = 0, "harness_def unreadable, defaulting to 0"
    n = max(0, n)
    if n > MAX_TRUNCATION_RETRIES:
        print(f"[TRUNC-RETRY] budget {n} exceeds the ceiling {MAX_TRUNCATION_RETRIES}, clamping. "
              f"Recovering more than that trades an empty patch for a wall-clock timeout, which is "
              f"the worse of the two.", flush=True)
        n = MAX_TRUNCATION_RETRIES
        source += f", clamped to {MAX_TRUNCATION_RETRIES}"
    return (n, source) if explain else n

from model_io import CLAUDE_MODEL, OPENAI_MODEL, GLM_MODEL, build_create_kwargs

def process_tool_call(tools_dict, tool_name, tool_input):
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]['function'](**tool_input)
        else:
            return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError, anthropic.RateLimitError, anthropic.APIStatusError),
    max_time=600,
    max_value=60,
)
def get_response_withtools(
    client, model, messages, tools, tool_choice,
    logging=None, max_retry=3, retry_count=0
):
    model = str(model) if not isinstance(model, str) else model

    try:
        if 'claude' in model:
            _create_kwargs = build_create_kwargs(model, messages, tools, tool_choice)
            max_tokens = _create_kwargs['max_tokens']
            response = client.messages.create(**_create_kwargs)
        elif model.startswith('o3-') or model == 'gpt-5.1-2025-11-13' or model == 'gpt-5.1-codex-mini':
            response = client.responses.create(
                model=model,
                input=messages,
                tool_choice=tool_choice,
                tools=tools,
                parallel_tool_calls=False
            )
            response = response
        elif model.startswith('glm-'):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=0.2,
                max_tokens=4096,
            )
            response = response.choices[0].message
        else:
            raise ValueError(f"Unsupported model: {model}")
        try:
            _sr = getattr(response, 'stop_reason', None)
            if _sr is None:
                _sr = getattr(response, 'finish_reason', None)
            _u = getattr(response, 'usage', None)
            _out = getattr(_u, 'output_tokens', None) if _u else None
            if _sr in ('max_tokens', 'length'):
                _m = f"[TRUNC] OUTPUT TRUNCATED stop_reason={_sr} output_tokens={_out} cap={max_tokens} model={model}"
                print(_m, flush=True)
                if logging:
                    logging(_m)
        except Exception:
            pass
        return response
    except Exception as e:
        error_str = str(e)
        is_rate_limit = (
            '429' in error_str or
            'Too many tokens' in error_str or
            'rate limit' in error_str.lower() or
            'RateLimitError' in error_str or
            isinstance(e, (openai.RateLimitError, anthropic.RateLimitError))
        )

        if logging:
            logging(f"Error in get_response_withtools: {error_str}")

        if is_rate_limit:
            if retry_count >= max_retry:
                if logging:
                    logging(f"Max retries ({max_retry}) exceeded for rate limit error. Raising exception.")
                raise

            wait_time = min(2 ** retry_count, 120)
            if logging:
                logging(f"Rate limit error (429) detected. Waiting {wait_time} seconds before retry {retry_count + 1}/{max_retry}")

            time.sleep(wait_time)

            return get_response_withtools(
                client, model, messages, tools, tool_choice,
                logging, max_retry, retry_count + 1
            )

        if max_retry > 0 and retry_count < max_retry:
            if logging:
                logging(f"Retrying after error (attempt {retry_count + 1}/{max_retry})")
            return get_response_withtools(
                client, model, messages, tools, tool_choice,
                logging, max_retry - 1, retry_count + 1
            )

        if 'Input is too long for requested model' in error_str:
            if logging:
                logging("Input is too long for requested model. Skipping retry.")
            pass

        raise

def check_for_tool_use(response, model=''):
    model = str(model) if not isinstance(model, str) else model

    if 'claude' in model:
        if response.stop_reason == "tool_use":
            tool_use_block = next(block for block in response.content if block.type == "tool_use")
            return {
                'tool_id': tool_use_block.id,
                'tool_name': tool_use_block.name,
                'tool_input': tool_use_block.input,
            }

    elif model.startswith('o3-') or model == 'gpt-5.1-2025-11-13' or model == 'gpt-5.1-codex-mini':
        tool_call = None
        for tool_call in response.output:
            if tool_call.type == "function_call":
                break

        if tool_call:
            return {
                'tool_id': tool_call.call_id,
                'tool_name': tool_call.name,
                'tool_input': json.loads(tool_call.arguments),
            }

    elif model.startswith('glm-'):
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            try:
                tool_input = json.loads(tool_call.function.arguments)
            except:
                tool_input = tool_call.function.arguments
            return {
                'tool_id': tool_call.id,
                'tool_name': tool_call.function.name,
                'tool_input': tool_input,
            }
        elif hasattr(response, 'content') and isinstance(response.content, str):
            pattern = r'<tool_use>(.*?)</tool_use>'
            match = re.search(pattern, response.content, re.DOTALL)
            if match:
                tool_use_str = match.group(1).strip()
                try:
                    tool_use_dict = ast.literal_eval(tool_use_str)
                    if isinstance(tool_use_dict, dict) and 'tool_name' in tool_use_dict and 'tool_input' in tool_use_dict:
                        return tool_use_dict
                except Exception:
                    pass

    else:
        pattern = r'<tool_use>(.*?)</tool_use>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            tool_use_str = match.group(1).strip()
            try:
                tool_use_dict = ast.literal_eval(tool_use_str)
                if isinstance(tool_use_dict, dict) and 'tool_name' in tool_use_dict and 'tool_input' in tool_use_dict:
                    return tool_use_dict
            except Exception:
                pass

    return None

def convert_tool_info(tool_info, model=None):
    if model is not None:
        model = str(model) if not isinstance(model, str) else model

    if model is None or 'claude' in model:
        return {
            'name': tool_info['name'],
            'description': tool_info['description'],
            'input_schema': tool_info['input_schema'],
        }
    elif model.startswith('o3-') or model == 'gpt-5.1-2025-11-13' or model == 'gpt-5.1-codex-mini':
        def add_additional_properties(d):
            if isinstance(d, dict):
                if 'properties' in d:
                    d['additionalProperties'] = False
                for k, v in d.items():
                    add_additional_properties(v)
        add_additional_properties(tool_info['input_schema'])
        for p in tool_info['input_schema']['properties'].keys():
            if not p in tool_info['input_schema']['required']:
                tool_info['input_schema']['required'].append(p)
                t = copy.deepcopy(tool_info['input_schema']['properties'][p]["type"])
                if isinstance(t, str):
                    tool_info['input_schema']['properties'][p]["type"] = [t, "null"]
                elif isinstance(t, list):
                    tool_info['input_schema']['properties'][p]["type"] = t + ["null"]

        return {
            'type': 'function',
            'name': tool_info['name'],
            'description': tool_info['description'],
            'parameters': tool_info['input_schema'],
            "strict": True,
        }
    elif model.startswith('glm-'):
        tool_schema = copy.deepcopy(tool_info['input_schema'])
        if 'required' not in tool_schema:
            tool_schema['required'] = []

        return {
            'type': 'function',
            'function': {
                'name': tool_info['name'],
                'description': tool_info['description'],
                'parameters': tool_schema,
            }
        }
    else:
        return tool_info

THINKING_KEEP_CHARS = 2000

def _clip_thinking(text):
    if len(text) <= 2 * THINKING_KEEP_CHARS:
        return text
    elided = len(text) - 2 * THINKING_KEEP_CHARS
    return (text[:THINKING_KEEP_CHARS]
            + f"\n<elided_chars>{elided} characters elided from the middle of this thinking block</elided_chars>\n"
            + text[-THINKING_KEEP_CHARS:])

def convert_block_claude(block):
    if isinstance(block, dict):
        block_type = block.get('type')
        text = block.get('text')
        tool_name = block.get('name')
        tool_input = block.get('input')
        tool_result = block.get('content')
    else:
        block_type = getattr(block, 'type', None)
        text = getattr(block, 'text', None)
        tool_name = getattr(block, 'name', None)
        tool_input = getattr(block, 'input', None)
        tool_result = getattr(block, 'content', None)

    text = text or ""

    if block_type == "text":
        return {
            "type": "text",
            "text": text
        }
    elif block_type == "tool_use":
        return {
            "type": "text",
            "text": f"<tool_use>\n{{'tool_name': {tool_name}, 'tool_input': {tool_input}}}\n</tool_use>"
        }
    elif block_type == "tool_result":
        return {
            "type": "text",
            "text": f"Tool Result: {tool_result}"
        }
    elif block_type in ("thinking", "redacted_thinking"):
        thinking_text = ""
        if isinstance(block, dict):
            thinking_text = block.get("thinking") or block.get("data") or ""
        else:
            thinking_text = getattr(block, "thinking", None) or getattr(block, "data", None) or ""
        return {
            "type": "text",
            "text": f"<thinking>\n{_clip_thinking(str(thinking_text))}\n</thinking>",
        }
    else:
        return {
            "type": "text",
            "text": str(block)
        }

def convert_msg_history_claude(msg_history):
    new_msg_history = []
    for msg in msg_history:
        role = msg.get('role', '')
        content_blocks = msg.get('content', [])
        new_content = []

        for block in content_blocks:
            new_content.append(convert_block_claude(block))

        new_msg_history.append({
            "role": role,
            "content": new_content
        })
    return new_msg_history

def convert_msg_history_openai(msg_history):
    new_msg_history = []

    for msg in msg_history:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')

            if role == 'tool':
                new_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool Result: {content}",
                        }
                    ],
                }
            else:
                new_msg = {
                    "role": role,
                    "content": content,
                }
        else:
            role = getattr(msg, 'role', None)
            content = getattr(msg, 'content', None)
            tool_calls = getattr(msg, 'tool_calls', None)

            if tool_calls:
                tool_call = tool_calls[0]
                function_name = getattr(tool_call.function, 'name', '')
                function_args = getattr(tool_call.function, 'arguments', '')
                new_msg = {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": f"<tool_use>\n{{'tool_name': {function_name}, 'tool_input': {function_args}}}\n</tool_use>",
                        }
                    ],
                }
            else:
                new_msg = {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                        }
                    ],
                }

        new_msg_history.append(new_msg)

    return new_msg_history

def convert_msg_history(msg_history, model=None):
    if model is not None:
        model = str(model) if not isinstance(model, str) else model

    if model is None or 'claude' in model:
        return convert_msg_history_claude(msg_history)
    elif model.startswith('o3-') or model == 'gpt-5.1-2025-11-13' or model == 'gpt-5.1-codex-mini':
        return convert_msg_history_openai(msg_history)
    else:
        return msg_history

def chat_with_agent_manualtools(msg, model, msg_history=None, logging=print):
    model = str(model) if not isinstance(model, str) else model

    if msg_history is None:
        msg_history = []
    system_message = f'You are a coding agent.\n\n{get_tooluse_prompt()}'
    new_msg_history = msg_history

    try:

        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}

        client, client_model = create_client(model)

        logging(f"Input: {msg}")
        response, new_msg_history = get_response_from_llm(
            msg=msg,
            client=client,
            model=client_model,
            system_message=system_message,
            print_debug=False,
            msg_history=new_msg_history,
        )
        logging(f"Output: {response}")

        tool_use = check_for_tool_use(response, model=client_model)
        while tool_use:
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            tool_msg = f'Tool Used: {tool_name}\nTool Input: {tool_input}\nTool Result: {tool_result}'
            logging(tool_msg)
            response, new_msg_history = get_response_from_llm(
                msg=tool_msg,
                client=client,
                model=client_model,
                system_message=system_message,
                print_debug=False,
                msg_history=new_msg_history,
            )
            logging(f"Output: {response}")

            tool_use = check_for_tool_use(response, model=client_model)

    except Exception as e:
        if model.startswith('glm-'):
            logging(f"[GLM Self-Modification] ERROR in chat_with_agent_manualtools: {str(e)}")
            import traceback
            logging(f"[GLM Self-Modification] Traceback: {traceback.format_exc()}")
        raise

    return new_msg_history

def chat_with_agent_glm(
        msg,
        model='glm-4.6',
        msg_history=None,
        logging=print,
    ):

    if msg_history is None:
        msg_history = []
    system_message = f'You are a coding agent.\n\n{get_tooluse_prompt()}'
    new_msg_history = msg_history

    try:
        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}

        client, client_model = create_client(model)

        response, new_msg_history = get_response_from_llm(
            msg=msg,
            client=client,
            model=client_model,
            system_message=system_message,
            print_debug=False,
            msg_history=new_msg_history,
        )

        tool_use = check_for_tool_use(response, model=client_model)

        while tool_use:
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            tool_msg = f'Tool Used: {tool_name}\nTool Input: {tool_input}\nTool Result: {tool_result}'
            logging(tool_msg)
            response, new_msg_history = get_response_from_llm(
                msg=tool_msg,
                client=client,
                model=client_model,
                system_message=system_message,
                print_debug=False,
                msg_history=new_msg_history,
            )

            tool_use = check_for_tool_use(response, model=client_model)

    except Exception as e:
        logging(f"[GLM Self-Modification] ERROR in chat_with_agent_glm: {str(e)}")
        import traceback
        logging(f"[GLM Self-Modification] Traceback: {traceback.format_exc()}")
        raise

    return new_msg_history

def chat_with_agent_glm_native(
        msg,
        model='glm-4.6',
        msg_history=None,
        logging=print,
    ):
    model = str(model) if not isinstance(model, str) else model

    if msg_history is None:
        msg_history = []
    new_msg_history = [
        {
            "role": "user",
            "content": msg,
        }
    ]

    try:
        client, client_model = create_client(model)

        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        tools = [convert_tool_info(tool['info'], model=client_model) for tool in all_tools]

        response = get_response_withtools(
            client=client,
            model=client_model,
            messages=msg_history + new_msg_history,
            tool_choice="auto",
            tools=tools,
            logging=logging,
        )

        tool_use = check_for_tool_use(response, model=client_model)

        while tool_use:
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            new_msg_history.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tool_use['tool_id'],
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_input),
                        }
                    }
                ] if hasattr(response, 'tool_calls') and response.tool_calls else None,
            })

            new_msg_history.append({
                "role": "tool",
                "tool_call_id": tool_use['tool_id'],
                "content": str(tool_result),
            })

            response = get_response_withtools(
                client=client,
                model=client_model,
                messages=msg_history + new_msg_history,
                tool_choice="auto",
                tools=tools,
                logging=logging,
            )

            tool_use = check_for_tool_use(response, model=client_model)

        if hasattr(response, 'content') and response.content:
            new_msg_history.append({
                "role": "assistant",
                "content": response.content,
            })

    except Exception as e:
        logging(f"[GLM Self-Modification] ERROR in chat_with_agent_glm_native: {str(e)}")
        import traceback
        logging(f"[GLM Self-Modification] Traceback: {traceback.format_exc()}")
        raise

    return new_msg_history

def chat_with_agent_claude(
        msg,
        model='bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        msg_history=None,
        logging=print,
    ):
    model = str(model) if not isinstance(model, str) else model

    if msg_history is None:
        msg_history = []
    new_msg_history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": msg,
                }
            ],
        }
    ]

    try:
        client, client_model = create_client(model)

        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        tools = [convert_tool_info(tool['info'], model=client_model) for tool in all_tools]

        _budget, _budget_source = _truncation_retries(explain=True)
        logging(f"[TRUNC-POLICY] truncation retry budget {_budget} ({_budget_source})")
        trunc_budget = [_budget]

        def _query():
            r = get_response_withtools(
                client=client, model=client_model, messages=msg_history + new_msg_history,
                tool_choice={"type": "auto"}, tools=tools, logging=logging)
            while trunc_budget[0] > 0:
                if getattr(r, 'stop_reason', None) != 'max_tokens':
                    return r
                if check_for_tool_use(r, model=client_model):
                    return r
                trunc_budget[0] -= 1
                logging("[TRUNC-RETRY] the reply hit the output cap without a complete tool call; "
                        f"asking again instead of treating it as 'finished' "
                        f"({trunc_budget[0]} recoveries left this episode)")
                try:
                    import harness_def as _hd
                    _keep = bool(_hd.keep_partial_text_on_retry())
                    _msg = (_hd.build_truncation_retry_message() or "").strip() \
                           or _hd.DEFAULT_TRUNCATION_RETRY_MESSAGE
                except Exception:
                    _keep, _msg = True, ("Your previous reply was cut off at the output token limit "
                                         "before it contained a complete tool call. Keep this reply "
                                         "short: skip the explanation and issue the single next tool call.")
                text_blocks = [{"type": "text", "text": b.text}
                               for b in (getattr(r, 'content', None) or [])
                               if getattr(b, "type", None) == "text" and getattr(b, "text", "")] if _keep else []
                new_msg_history.append({
                    "role": "assistant",
                    "content": text_blocks or [{"type": "text", "text": "(cut off)"}]})
                new_msg_history.append({"role": "user", "content": [{"type": "text", "text": _msg}]})
                r = get_response_withtools(
                    client=client, model=client_model, messages=msg_history + new_msg_history,
                    tool_choice={"type": "auto"}, tools=tools, logging=logging)
            return r

        response = _query()
        tool_use = check_for_tool_use(response, model=client_model)
        while tool_use:
            tool_use_blocks = []
            if hasattr(response, 'content') and response.content:
                for block in response.content:
                    block_type = None
                    if hasattr(block, 'type'):
                        block_type = block.type
                    elif isinstance(block, dict):
                        block_type = block.get('type')

                    if block_type == "tool_use":
                        tool_use_blocks.append(block)

            tool_results = []
            for tool_use_block in tool_use_blocks:
                if hasattr(tool_use_block, 'id'):
                    tool_id = tool_use_block.id
                    tool_name = tool_use_block.name
                    tool_input = tool_use_block.input
                elif isinstance(tool_use_block, dict):
                    tool_id = tool_use_block.get('id')
                    tool_name = tool_use_block.get('name')
                    tool_input = tool_use_block.get('input')
                else:
                    continue

                tool_result = process_tool_call(tools_dict, tool_name, tool_input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": tool_result,
                })

            if not tool_results:
                tool_name = tool_use['tool_name']
                tool_input = tool_use['tool_input']
                tool_result = process_tool_call(tools_dict, tool_name, tool_input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use['tool_id'],
                    "content": tool_result,
                })

            new_msg_history.append({"role": "assistant", "content": response.content})
            new_msg_history.append({
                "role": "user",
                "content": tool_results,
            })

            response = _query()

            tool_use = check_for_tool_use(response, model=client_model)

        final_response = next((block.text for block in response.content if hasattr(block, "text")), None)
        new_msg_history.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": final_response,
                }
            ],
        })

    except Exception as e:
        logging(f"Error in chat_with_agent_claude: {e}")
        import traceback
        logging(f"Traceback: {traceback.format_exc()}")
        return new_msg_history

    return new_msg_history

def chat_with_agent_openai(
        msg,
        model='o3-mini-2025-01-31',
        msg_history=None,
        logging=print,
    ):
    model = str(model) if not isinstance(model, str) else model

    if msg_history is None:
        msg_history = []
    new_msg_history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": msg,
                }
            ],
        }
    ]
    separator = '=' * 10
    logging(f"\n{separator} User Instruction {separator}\n{msg}")
    try:
        client, client_model = create_client(model)

        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        tools = [convert_tool_info(tool['info'], model=client_model) for tool in all_tools]

        response = get_response_withtools(
            client=client,
            model=client_model,
            messages=msg_history + new_msg_history,
            tool_choice="auto",
            tools=tools,
            logging=logging,
        )
        logging(f"\n{separator} Agent Response {separator}\n{response}")

        tool_use = check_for_tool_use(response, model=client_model)
        logging(tool_use)
        while tool_use:
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            logging(f"Tool Used: {tool_name}")
            logging(f"Tool Input: {tool_input}")
            logging(f"Tool Result: {tool_result}")

            for tool_call in response.output:
                if tool_call.type == "function_call":
                    break
            new_msg_history.append(tool_call)
            new_msg_history.append({
                "type": "function_call_output",
                "call_id": tool_use['tool_id'],
                "output": tool_result,
            })
            response = get_response_withtools(
                client=client,
                model=client_model,
                messages=msg_history + new_msg_history,
                tool_choice="auto",
                tools=tools,
                logging=logging,
            )

            tool_use = check_for_tool_use(response, model=client_model)

            logging(f"Tool Response: {response}")

        new_msg_history.append(response)

    except Exception:
        pass

    return new_msg_history

def chat_with_agent(
    msg,
    model=str(CLAUDE_MODEL),
    msg_history=None,
    logging=print,
    convert=False,
):
    model = str(model) if not isinstance(model, str) else model

    msg_str = str(msg)
    msg_preview = msg_str[:200] + "..." if len(msg_str) > 200 else msg_str
    if msg_history is None:
        msg_history = []

    if 'claude' in model:
        new_msg_history = chat_with_agent_claude(msg, model=model, msg_history=msg_history, logging=logging)
        conv_msg_history = convert_msg_history(new_msg_history, model=model)
        logging(conv_msg_history)
        if convert:
            new_msg_history = conv_msg_history
        new_msg_history = msg_history + new_msg_history

    elif model.startswith('o3-') or model == 'gpt-5.1-2025-11-13' or model == 'gpt-5.1-codex-mini':
        new_msg_history = chat_with_agent_openai(msg, model=model, msg_history=msg_history, logging=logging)
        new_msg_history = msg_history + new_msg_history

    elif model.startswith('glm-'):
        new_msg_history = chat_with_agent_glm_native(msg, model=model, msg_history=msg_history, logging=logging)
        conv_msg_history = convert_msg_history(new_msg_history, model=model)
        if convert:
            new_msg_history = conv_msg_history

    else:
        new_msg_history = chat_with_agent_manualtools(msg, model=model, msg_history=msg_history, logging=logging)
        conv_msg_history = convert_msg_history(new_msg_history, model=model)
        if convert:
            new_msg_history = conv_msg_history

    return new_msg_history

if __name__ == "__main__":
    msg = "hello!"
    chat_with_agent(msg)
