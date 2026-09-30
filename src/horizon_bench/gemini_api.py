# Methods for using the Google Gemini API

import time

from google import genai
from google.genai import types
from google.genai.errors import ServerError
from pydantic import BaseModel
from horizon_bench.Config import MEMCACHE_PATH, DEFAULT_MODEL_NAME, GOOGLE_API_KEY, DEFAULT_MODEL_NAME

from joblib import Memory


def cache_validation_callback(func):
    args = func["input_args"]

    if 'invalidate_cache' in args and args['invalidate_cache'] == 'True':
        return False  # Do not use cache
    return True  # Use cache

memory = Memory(MEMCACHE_PATH, verbose=0)

client = genai.Client(api_key=GOOGLE_API_KEY)



OPENAI_TO_GEMINI_ROLE = {
    "user": "user",
    "assistant": "model",
}


def convert_message_history(message_history: list) -> tuple[list[types.Content], str | None]:
    """Convert OpenAI-format message history to Gemini Content objects.

    Extracts any system messages as a system prompt and converts
    user/assistant messages to Gemini's Content format.

    Returns:
        (contents, system_prompt) where system_prompt is extracted from
        any 'system' role messages, or None if there are none.
    """
    contents = []
    system_parts = []
    for msg in message_history:
        role = msg["role"]
        text = msg["content"]
        if role == "system":
            system_parts.append(text)
        else:
            gemini_role = OPENAI_TO_GEMINI_ROLE[role]
            contents.append(types.Content(role=gemini_role, parts=[types.Part(text=text)]))
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return contents, system_prompt


def format_messages(user_prompt: str, system_prompt: str = None) -> list[types.Content]:
    """Format messages into Gemini Content objects."""
    assert isinstance(user_prompt, str), "User prompt must be a string."
    messages = []
    if system_prompt:
        assert isinstance(system_prompt, str), "System prompt must be a string."
    messages.append(types.Content(role="user", parts=[types.Part(text=user_prompt)]))
    return messages


@memory.cache(cache_validation_callback=cache_validation_callback)
def chat(user_prompt: str, system_prompt: str = None, message_history: list = None, model_name=DEFAULT_MODEL_NAME, invalidate_cache=False) -> str:
    """
    Provides plain text response from Gemini for a given user prompt, optional system prompt, and message history.
    Whenever you need structured response output, always use get_formatted_chat_response instead of this function.
    """
    config = types.GenerateContentConfig()

    contents = []
    if message_history:
        history_contents, history_system = convert_message_history(message_history)
        contents.extend(history_contents)
        # System prompt from history is used if no explicit system_prompt is given
        if not system_prompt and history_system:
            system_prompt = history_system

    if system_prompt:
        config.system_instruction = system_prompt

    contents.extend(format_messages(user_prompt))

    for _attempt in range(10):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            break
        except ServerError as e:
            if _attempt < 9 and "503" in str(e):
                print(f"Gemini 503 error, retrying ({_attempt + 1}/10)...")
                time.sleep(5)
            else:
                raise
    return response.text


@memory.cache
def get_formatted_chat_response(response_format: BaseModel, user_prompt: str, system_prompt: str = None, message_history: list = None, model_name=DEFAULT_MODEL_NAME) -> BaseModel:
    """
    Get a structured response from Gemini using a Pydantic model.

    Args:
        response_format: Pydantic BaseModel class defining the expected response structure
        user_prompt: The user's message
        system_prompt: Optional system prompt
        message_history: Optional list of previous messages to maintain conversation context.
            Accepts OpenAI-format dicts (role/content) which are auto-converted.

    Example response_format:
    class Recipe(BaseModel):
        name: str
        ingredients: list[str]
    """
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_format,
    )

    contents = []
    if message_history:
        history_contents, history_system = convert_message_history(message_history)
        contents.extend(history_contents)
        if not system_prompt and history_system:
            system_prompt = history_system

    if system_prompt:
        config.system_instruction = system_prompt

    contents.extend(format_messages(user_prompt))

    for _attempt in range(10):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            break
        except ServerError as e:
            if _attempt < 9 and "503" in str(e):
                print(f"Gemini 503 error, retrying ({_attempt + 1}/10)...")
                time.sleep(5)
            else:
                raise
    return response_format.model_validate_json(response.text)


if __name__ == '__main__':
    from pydantic import BaseModel
    from time import time

    class PartyName(BaseModel):
        party_name: str

    start_time = time()
    prompt = "Which political party won the 2026 elections in Baden-Württemberg, Germany? d"

    response = get_formatted_chat_response(user_prompt=prompt, response_format=PartyName)
    t1 = time() - start_time
    print(response)
    print('Response time with formatted chat:', t1)

    start_time = time()
    response = chat(user_prompt=prompt)
    t2 = time() - start_time
    print(response)
    print('Response time with plain chat:', t2)
