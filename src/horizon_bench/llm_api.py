# Methods for using the OpenAI API

from pydantic import BaseModel
from horizon_bench.Config import DEFAULT_MODEL_NAME
from horizon_bench import openai_api
from horizon_bench import gemini_api
from horizon_bench import vllm_api
from horizon_bench import Config

def chat(user_prompt: str, system_prompt: str = None, message_history: list = [], model_name=DEFAULT_MODEL_NAME, invalidate_cache=False) -> dict:
    if Config.LLMAPINAME == Config.LLMAPIType.OPENAI:
        return openai_api.chat(user_prompt=user_prompt, 
                            system_prompt=system_prompt, 
                            message_history=message_history, 
                            model_name=model_name, 
                            invalidate_cache=invalidate_cache)
    elif Config.LLMAPINAME == Config.LLMAPIType.GEMINI:
        return gemini_api.chat(user_prompt=user_prompt,
                            system_prompt=system_prompt,
                            message_history=message_history,
                            model_name=model_name,
                            invalidate_cache=invalidate_cache)
    elif Config.LLMAPINAME == Config.LLMAPIType.VLLM:
        return vllm_api.chat(user_prompt=user_prompt,
                            system_prompt=system_prompt,
                            message_history=message_history,
                            model_name=model_name,
                            invalidate_cache=invalidate_cache)


def get_formatted_chat_response(response_format: BaseModel, user_prompt: str, system_prompt: str = None, message_history: list = None) -> BaseModel:
    if Config.LLMAPINAME == Config.LLMAPIType.OPENAI:
        return openai_api.get_formatted_chat_response(response_format=response_format, 
                                                   user_prompt=user_prompt, 
                                                   system_prompt=system_prompt, 
                                                   message_history=message_history)
    elif Config.LLMAPINAME == Config.LLMAPIType.GEMINI:
        return gemini_api.get_formatted_chat_response(response_format=response_format,
                                                   user_prompt=user_prompt,
                                                   system_prompt=system_prompt,
                                                   message_history=message_history)
    elif Config.LLMAPINAME == Config.LLMAPIType.VLLM:
        return vllm_api.get_formatted_chat_response(response_format=response_format,
                                                   user_prompt=user_prompt,
                                                   system_prompt=system_prompt,
                                                   message_history=message_history)
    raise ValueError(f"Unsupported LLM API: {Config.LLMAPINAME}")
    


def chat_with_agentic_tool_use(user_prompt: str, tools: list, response_format: BaseModel, system_prompt: str = None, model=DEFAULT_MODEL_NAME,**kwargs) -> dict:
    if Config.LLMAPINAME == Config.LLMAPIType.OPENAI:
        return openai_api.chat_with_agentic_tool_use(user_prompt=user_prompt,
                                                 tools=tools,
                                                 response_format=response_format,
                                                 system_prompt=system_prompt,
                                                 model=model,
                                                 **kwargs)
    elif Config.LLMAPINAME == Config.LLMAPIType.GEMINI:
        raise NotImplementedError("Agentic tool use is not yet implemented for Gemini API")
    elif Config.LLMAPINAME == Config.LLMAPIType.VLLM:
        return vllm_api.chat_with_agentic_tool_use(user_prompt=user_prompt,
                                                   tools=tools,
                                                   response_format=response_format,
                                                   system_prompt=system_prompt,
                                                   model=model,
                                                   **kwargs)


if __name__=='__main__':
    # class PartyName(BaseModel):
    #     date: str


    # from time import time
    # start_time = time() 
    # prompt = """Which political party won the 2026 elections in Baden-Württemberg, Germany?"""

    # response = get_formatted_chat_response(user_prompt=prompt, response_format=PartyName)
    # t1 = time() - start_time
    # print(response)
    # print('Response time with formatted chat:', t1)
    from pydantic import BaseModel, Field
    from typing import Literal


    class IdeaDetails(BaseModel):
        """Structure for the final research idea."""
        Name: str = Field(description="A short descriptor of the idea. Lowercase, no spaces, underscores allowed.")
        Title: str = Field(description="A catchy and informative title for the proposal.")
        Short_Hypothesis: str = Field(
            alias="Short Hypothesis",
            description="A concise statement of the main hypothesis or research question."
        )
        Related_Work: str = Field(
            alias="Related Work",
            description="A brief discussion of the most relevant related work and how the proposal distinguishes from it."
        )
        Abstract: str = Field(description="An abstract that summarizes the proposal in conference format (approximately 250 words).")
        Experiments: str = Field(description="A list of experiments that would be conducted to validate the proposal.")
        Risk_Factors_and_Limitations: str = Field(
            alias="Risk Factors and Limitations",
            description="A list of potential risks and limitations of the proposal."
        )

    class AIScientistResponse(BaseModel):
        """Union type for AI Scientist v2 responses."""
        action_type: Literal["SearchSemanticScholar", "FinalizeIdea"] = Field(
            description="The type of action to take"
        )
        search_query: str | None = Field(
            default=None,
            description="Search query if action is SearchSemanticScholar"
        )
        idea: IdeaDetails | None = Field(
            default=None,
            description="Idea details if action is FinalizeIdea"
        )
        reasoning: str = Field(
            default="",
            description="Your reasoning for this action"
        )
    prompt_text = 'To establish the fundamental information-theoretic and algorithmic conditions required for successful sparse recovery when utilizing datasets comprising measurements of varying noise levels.\n\nBegin by generating an interestingly new high-level research proposal for the topic above.\n'
    message_history = [
        {'role': 'system', 'content': 'You are an experienced AI researcher who aims to propose high-impact research ideas resembling exciting grant proposals. Feel free to propose any novel ideas or experiments; make sure they are novel. Be very creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis about the topic. For example, they could involve very interesting and simple interventions or investigations that explore new possibilities or challenge existing assumptions. Clearly clarify how the proposal distinguishes from the existing literature.\n\nEnsure that the proposal does not require resources beyond what an academic lab could afford. These proposals should lead to papers that are publishable at top ML conferences.\n\nYou have access to the following actions:\n\n1. **SearchSemanticScholar**: Search for relevant literature using Semantic Scholar to inform your research idea.\n   - Set action_type to "SearchSemanticScholar"\n   - Provide a search_query with your search terms\n   - Explain your reasoning for the search\n\n2. **FinalizeIdea**: When you\'re ready to finalize your research idea.\n   - Set action_type to "FinalizeIdea"\n   - Provide complete idea details with all required fields:\n     * Name: A short descriptor (lowercase, no spaces, underscores allowed)\n     * Title: A catchy and informative title\n     * Short Hypothesis: A concise statement of the main hypothesis or research question. Clarify the need for this specific direction, ensure this is the best setting to investigate this idea, and there are not obvious other simpler ways to answer the question.\n     * Related Work: A brief discussion of the most relevant related work and how the proposal clearly distinguishes from it, and is not a trivial extension.\n     * Abstract: An abstract that summarizes the proposal in conference format (approximately 250 words)\n     * Experiments: A list of experiments that would be conducted to validate the proposal. Ensure these are simple and feasible. Be specific in exactly how you would test the hypothesis, and detail precise algorithmic changes. Include the evaluation metrics you would use.\n     * Risk Factors and Limitations: A list of potential risks and limitations of the proposal\n   - Explain your reasoning for the final idea\n\nNote: You should perform at least one literature search before finalizing your idea to ensure it is well-informed by existing research.'},
        {'role': 'user', 'content': 'To establish the fundamental information-theoretic and algorithmic conditions required for successful sparse recovery when utilizing datasets comprising measurements of varying noise levels.\n\nBegin by generating an interestingly new high-level research proposal for the topic above.\n'}
    ]
    response = get_formatted_chat_response(
                response_format=AIScientistResponse,
                user_prompt=prompt_text,
                message_history=message_history[:-1],  # Exclude the user message we just added
            )
    print(response)