# Methods for using a remote vLLM server (OpenAI-compatible API)
# Connects to a remote GPU host via an SSH jump host using port forwarding.

import json
import os
import subprocess
import time
from openai import OpenAI
from pydantic import BaseModel
from horizon_bench.Config import MEMCACHE_PATH
from ai_researcher.classes.Tools import ExecuteCommandTool, ReadFileTool, CodeStructureTool
from horizon_bench.Config import DEFAULT_MODEL_NAME

from joblib import Memory
memory = Memory(MEMCACHE_PATH + "_vllm", verbose=0)

AVAILABLE_TOOLS = {
    "execute_command": ExecuteCommandTool(),
    "read_file": ReadFileTool(),
    "gen_code_tree_structure": CodeStructureTool()
}

REMOTE_DIR = "vllm_server"


# Remote server connection details (mirrors deploy_vllm.sh)
JUMP_USER = None
JUMP_HOST = None
TARGET_HOST = None
REMOTE_PORT = None
LOCAL_TUNNEL_PORT = None

VLLM_BASE_URL = f"http://localhost:{LOCAL_TUNNEL_PORT}/v1"
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # horizon_bench/src/

_tunnel_process: subprocess.Popen | None = None


def cache_validation_callback(func):
    args = func["input_args"]
    if 'invalidate_cache' in args and args['invalidate_cache'] == 'True':
        return False
    return True


@memory.cache(cache_validation_callback=cache_validation_callback)
def chat(user_prompt: str, system_prompt: str = None, message_history: list = [], model_name=DEFAULT_MODEL_NAME, invalidate_cache=False) -> str:
    """
    Provides plain text response from the local vLLM server for a given user prompt,
    optional system prompt, and message history.
    Whenever you need structured response output, use get_formatted_chat_response instead.
    """
    client = _get_client()
    messages = list(message_history)
    messages.extend(format_messages(user_prompt, system_prompt) if system_prompt else format_messages(user_prompt))

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return response.choices[0].message.content


@memory.cache
def get_formatted_chat_response(response_format: type[BaseModel], user_prompt: str, system_prompt: str = None, message_history: list = None, model_name: str = DEFAULT_MODEL_NAME) -> BaseModel:
    """
    Get a structured response from the local vLLM server using a Pydantic model.

    Uses vLLM's structured outputs via beta.chat.completions.parse() to constrain
    the output to the given Pydantic schema.
    """
    client = _get_client()

    if message_history:
        messages = message_history.copy()
        messages.append({"role": "user", "content": user_prompt})
    else:
        messages = format_messages(user_prompt, system_prompt) if system_prompt else format_messages(user_prompt)

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        extra_body={"structured_outputs": {"json": response_format.model_json_schema()}},
        response_format=response_format,
        max_tokens=4096,
    )
    return completion.choices[0].message.parsed



@memory.cache
def chat_with_agentic_tool_use(user_prompt: str, tools: list, response_format: BaseModel, system_prompt: str = None, model=DEFAULT_MODEL_NAME, **kwargs) -> BaseModel:
    validate_tools_availability(tools)
    MAX_TURNS = 15
    client = _get_client()

    messages = format_messages(user_prompt, system_prompt) if system_prompt else format_messages(user_prompt)

    # Chat completions uses {"type": "function", "function": {...}} nesting (unlike Responses API)
    tool_dicts = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.parameters,
                    "required": list(tool.parameters.keys()),
                },
            },
        }
        for tool in tools
    ]

    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_dicts,
            **kwargs,
        )
        msg = response.choices[0].message

        # Append assistant message (with tool_calls if any)
        assistant_message = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_message)

        if response.choices[0].finish_reason != "tool_calls" or not msg.tool_calls:
            break

        print(f"Tool calls in this turn: {len(msg.tool_calls)}")
        print("\n".join(str(tc) for tc in msg.tool_calls))
        print(f"Rounds of tool use so far: {turn + 1}")

        for tc in msg.tool_calls:
            tool_response = AVAILABLE_TOOLS[tc.function.name](**json.loads(tc.function.arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_response,
            })

    # Get final structured response
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        extra_body={"structured_outputs": {"json": response_format.model_json_schema()}},
        response_format=response_format,
        max_tokens=4096,
    )
    return completion.choices[0].message.parsed



def validate_tools_availability(tools: list) -> None:
    assert all(tool.name in AVAILABLE_TOOLS for tool in tools), \
        f"One or more tools are not available. Tools requested: {tools}, Tools available: {AVAILABLE_TOOLS}"


def format_messages(user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> list[dict]:
    assert isinstance(user_prompt, str), "User prompt must be a string."
    assert isinstance(system_prompt, str), "System prompt must be a string."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def _port_is_open(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_tunnel() -> None:
    """Start an SSH tunnel: localhost:LOCAL_TUNNEL_PORT -> TARGET_HOST:REMOTE_PORT via JUMP_HOST."""
    global _tunnel_process
    if _tunnel_process is not None and _tunnel_process.poll() is None:
        return  # already running

    if _port_is_open(LOCAL_TUNNEL_PORT):
        return  # tunnel already forwarded (e.g. from a prior process)

    cmd = [
        "ssh", "-N", "-o", "ExitOnForwardFailure=yes",
        "-L", f"{LOCAL_TUNNEL_PORT}:{TARGET_HOST}:{REMOTE_PORT}",
        f"{JUMP_USER}@{JUMP_HOST}",
    ]
    _tunnel_process = subprocess.Popen(cmd)
    time.sleep(2)  # give SSH a moment to establish the tunnel


def stop_tunnel() -> None:
    """Terminate the SSH tunnel if running."""
    global _tunnel_process
    if _tunnel_process is not None:
        _tunnel_process.terminate()
        _tunnel_process = None


def restart_vllm_server(model_name: str = DEFAULT_MODEL_NAME, timeout: int = 120000, poll_interval: int = 5) -> None:
    """Redeploy the vLLM server on the remote host with the given model,
    then wait until the server is up and serving the model.

    Raises TimeoutError if the server does not become ready within `timeout` seconds.
    """
    script = os.path.join(_SCRIPT_DIR, "deploy_vllm.sh")
    subprocess.Popen(["bash", script, model_name], cwd=_SCRIPT_DIR, env=os.environ.copy())
    
    time.sleep(30)  # give the server some time to start before polling

    start_tunnel()
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            models = client.models.list()
            loaded = [m.id for m in models.data]
            assert model_name in loaded, f"Expected model {model_name!r} but server has {loaded}"
            print(f"vLLM server is up with model {model_name}")
            return
        except Exception:
            pass
        time.sleep(poll_interval)

    raise TimeoutError(f"vLLM server did not become ready within {timeout}s")


def _get_client() -> OpenAI:
    start_tunnel()
    return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


if __name__ == "__main__":
    from ai_researcher.classes.Tools import ReadFileTool

    # 0. Test restart_vllm_server()
    print("=== Test 0: restart_vllm_server() ===")
    # restart_vllm_server(DEFAULT_MODEL_NAME)
    print(f"Server restarted and ready with model {DEFAULT_MODEL_NAME}")

    # 1. Test chat()
    print("=== Test 1: chat() ===")
    reply = chat(
        user_prompt="What is 2 + 2? Answer in one sentence. a",
        system_prompt="You are a concise assistant.",
        invalidate_cache=True,
    )
    print(reply)

    # 2. Test get_formatted_chat_response()
    print("\n=== Test 2: get_formatted_chat_response() ===")

    class MathAnswer(BaseModel):
        question: str
        answer: int
        explanation: str

    result = get_formatted_chat_response(
        response_format=MathAnswer,
        user_prompt="What is 7 multiplied by 6? ",
        system_prompt="You are a math assistant. respond ONLY with JSON.",
    )
    print(result)

    # 3. Test chat_with_agentic_tool_use()
    print("\n=== Test 3: chat_with_agentic_tool_use() ===")

    class FileContent(BaseModel):
        filename: str
        summary: str

    read_tool = ReadFileTool()
    result = chat_with_agentic_tool_use(
        user_prompt=f"Read the file {os.path.join(_SCRIPT_DIR, 'horizon_bench', 'Config.py')} and summarize what it does in one sentence. ",
        tools=[read_tool],
        response_format=FileContent,
        system_prompt="You are a helpful assistant with access to file reading tools.",
    )
    print(result)
