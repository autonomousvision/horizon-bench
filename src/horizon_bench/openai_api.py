# Methods for using the OpenAI API

import json
from openai import OpenAI
from pydantic import BaseModel
from horizon_bench.Config import MEMCACHE_PATH
from horizon_bench.Config import DEFAULT_MODEL_NAME, OPENAI_API_KEY
from ai_researcher.classes.Tools import ExecuteCommandTool, ReadFileTool, CodeStructureTool

from joblib import Memory


def cache_validation_callback(func):
    args = func["input_args"]

    if 'invalidate_cache' in args and args['invalidate_cache'] == 'True':
        return False  # Do not use cache
    return True  # Use cache

memory = Memory(MEMCACHE_PATH, verbose=0, )


# Dict of tool names to tool functions
AVAILABLE_TOOLS = {
    "execute_command": ExecuteCommandTool(),
    "read_file": ReadFileTool(),
    "gen_code_tree_structure": CodeStructureTool()
}


@memory.cache(cache_validation_callback=cache_validation_callback)
def chat(user_prompt: str, system_prompt: str = None, message_history: list = [], model_name=DEFAULT_MODEL_NAME, invalidate_cache=False) -> dict:
    """
    Provides plain text response from the LLM for a given user prompt, optional system prompt, and message history.
    Whenever you need structured response output, always use get_formatted_chat_response instead of this function. 
    """
    response = batch_chat(n=1, user_prompt=user_prompt, system_prompt=system_prompt, message_history=message_history, model_name=model_name)
    return response.choices[0].message.content


def batch_chat(n: int, user_prompt: str, system_prompt: str = None, message_history: list = [], model_name=DEFAULT_MODEL_NAME) -> dict:
    client = OpenAI(api_key=OPENAI_API_KEY)
    message_history.extend(format_messages(user_prompt, system_prompt) if system_prompt else format_messages(user_prompt))
    
    response = client.chat.completions.create(
        model=DEFAULT_MODEL_NAME,
        messages=message_history,
        n=n
    )
    return response

@memory.cache
def get_formatted_chat_response(response_format: BaseModel, user_prompt: str, system_prompt: str = None, message_history: list = None) -> BaseModel:
    """
    Get a structured response from the LLM using a Pydantic model.

    Args:
        response_format: Pydantic BaseModel class defining the expected response structure
        user_prompt: The user's message
        system_prompt: Optional system prompt (default: "You are a helpful assistant.")
        message_history: Optional list of previous messages to maintain conversation context

    Example response_format:
    class Recipe(BaseModel):
        name: str
        ingredients: list[str]
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build message list
    if message_history:
        messages = message_history.copy()
        messages.append({"role": "user", "content": user_prompt})
    else:
        messages = format_messages(user_prompt, system_prompt) if system_prompt else format_messages(user_prompt)

    response = client.responses.parse(
        model=DEFAULT_MODEL_NAME,
        input=messages,
        text_format=response_format,
        # tools=[{"type": "web_search_preview"}], # if websearch is enabled, then the knowledge cutoff can be bypassed.
    )
    return response.output_parsed

@memory.cache
def chat_with_agentic_tool_use(user_prompt: str, tools: list, response_format: BaseModel, system_prompt: str = None, model=DEFAULT_MODEL_NAME,**kwargs) -> dict:
    validate_tools_availability(tools)
    MAX_TURNS = 15

    input_list = format_messages(user_prompt, system_prompt) if system_prompt else format_messages(user_prompt)

    # openai takes the tool descriptions and parameters as dicts
    tool_dicts = [tool.to_dict() for tool in tools]
    for _ in range(MAX_TURNS):
        response = openai_parse(model=model,
                                tools=tool_dicts,
                                input=input_list,
                                response_format=response_format,
                                **kwargs)
        input_list += response.output

        # function_call means the model is asking to use a tool
        new_tool_calls = [output for output in response.output if output.type == "function_call"]
        if not new_tool_calls:
            # Ready to give a response
            break

        print(f"Tool calls in this turn: {str(len(new_tool_calls))}")
        print("\n".join(str(call) for call in new_tool_calls))
        print(f"Rounds of tool use so far: {_+1}")

        for tool_call in new_tool_calls:
            # next() gets the first item that matches the condition here (tool name matches)
            tool_name = next((t for t in tools if t.name == tool_call.name), None).name
            tool_response = AVAILABLE_TOOLS[tool_name](**json.loads(tool_call.arguments))
            input_list.append({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": tool_response
            })

    all_messages = input_list + response.output # in case you want to log this for debugging
    print(all_messages)
    return response.output_parsed

client = OpenAI(api_key=OPENAI_API_KEY)

# TODO: figure out memory.cache for OpenAIP calls
# @memory.cache() Can't cache this, because: 
# Tool calls have ids and timestamps
# Response objects cannot be pickled by joblib
def openai_parse(model: str, tools: list, input: list[dict], response_format: BaseModel, **kwargs) -> dict:
    try:
        response = client.responses.parse(model=model,
                                           tools=tools,
                                           input=input,
                                           text_format=response_format,
                                           **kwargs)
        return response
    except Exception as e:
        print(input)
        print(f"Error occurred while parsing OpenAI response: {e}")
        raise e

def validate_tools_availability(tools: list) -> bool:
    """
    I needed to make the AVAILABLE_TOOLS dict, to have a string -> function mapping of available tools.
    This function checks if all requested tools are in AVAILABLE_TOOLS.
    """
    assert all(tool.name in AVAILABLE_TOOLS for tool in tools), "One or more tools are not available. Tools requested: {tools}, Tools available: {AVAILABLE_TOOLS}"

def format_messages(user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> list[dict]:
    assert isinstance(user_prompt, str), "User prompt must be a string."
    assert isinstance(system_prompt, str), "System prompt must be a string."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]



class PartyName(BaseModel):
    party_name: str

if __name__=='__main__':
    # message_reply = chat(user_prompt="Count up by only one integer in your reply. Starting count is 1", system_prompt="You are a helpful assistant.")
    # print(message_reply)
    # message_history = [message_reply]
    # message_reply = chat(user_prompt="What is the next number?", message_history=message_history, invalidate_cache=True)
    # print(message_reply)
    from time import time
    start_time = time() 
    prompt = """Which political party won the 2026 elections in Baden-Württemberg, Germany? Search the web for up to date information"""

    # response = get_formatted_chat_response(user_prompt=prompt, response_format=PartyName)
    # t1 = time() - start_time
    # print(response)
    # print('Response time with formatted chat:', t1)

    
    response = chat(user_prompt=prompt)
    print(response)


#     t2 = time() - start_time - t1
#     print('Response time with cached chat:', t2)

#     client.responses.create(
#     model=DEFAULT_MODEL_NAME,
#     input=prompt,
# )
#     t3 = time() - start_time - t1 - t2
#     print('Response time with direct client call:', t3)
