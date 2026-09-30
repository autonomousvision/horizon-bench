from ai_researcher.task_to_references import task_to_references
from horizon_bench.load_goal_prompt import load_goal_prompt
from horizon_bench.logger import initialize_logger
from pydantic import BaseModel
import datetime
from horizon_bench.Config import DEFAULT_MODEL_NAME
from ai_researcher.classes.Tools import ReadFileTool
from horizon_bench.llm_api import chat_with_agentic_tool_use
from ai_researcher.prompts.idea_prompt import idea_user_prompt, idea_system_prompt
from ai_researcher.prompts.reference_codebases_prompt import get_reference_codebases_user_prompt, reference_codebases_system_prompt
from horizon_bench.llm_api import chat_with_agentic_tool_use
from ai_researcher.classes.Tools import ExecuteCommandTool, ReadFileTool, CodeStructureTool
from horizon_bench.arxiv_search import search_download_arxiv, string_format_arxiv_path_title
from ai_researcher.github_search import search_github_batched


class CodeBasesResult(BaseModel):
    codebases: list[str]
    paper_titles: list[str]


class IdeaResult(BaseModel):
    challenges: str
    existing_methods: str
    motivation: str
    proposed_method: str
    technical_details: str
    expected_outcomes: str

    
def run_ai_researcher(question_filename: str):
    initialize_logger("ai_researcher", question_filename)
    task_description = load_goal_prompt(question_filename)

    REFERENCES = task_to_references(task_description)[:5]
    # assert len(REFERENCES) >= 5, "Please provide at least 5 reference paper titles."

    MODEL_NAME = DEFAULT_MODEL_NAME
    # AI Researcher used GPT 4o. gpt-5-nano is better and cheaper. 
    # Best and most expensive is gpt-5

    github_result = search_github_batched(REFERENCES)
    user_prompt = get_reference_codebases_user_prompt(REFERENCES, github_result)
    system_prompt = reference_codebases_system_prompt

    chosen_codebases = chat_with_agentic_tool_use(
        user_prompt=user_prompt,
        tools=[ExecuteCommandTool(), ReadFileTool(), CodeStructureTool()],
        system_prompt=system_prompt,
        response_format=CodeBasesResult,
        model=MODEL_NAME
    )

    arxiv_search_results = search_download_arxiv(chosen_codebases.paper_titles)
    if arxiv_search_results == []:
         raise Exception("No arXiv papers were found or downloaded.")
    formatted_arxiv_search_results = string_format_arxiv_path_title(arxiv_search_results)

    response = chat_with_agentic_tool_use(
        model=MODEL_NAME,
        user_prompt=idea_user_prompt.format(task=task_description,
                    references=chosen_codebases.paper_titles,
                    github_codebases=chosen_codebases.codebases,
                    download_res=formatted_arxiv_search_results),
        tools=[ReadFileTool()],
        system_prompt=idea_system_prompt,
        response_format=IdeaResult,
    )

    res = f"""
    Challenges:
    {response.challenges}
    ### Existing Methods:
    {response.existing_methods}
    ### Motivation:
    {response.motivation}
    ### Proposed Method:
    {response.proposed_method}
    ### Technical Details:
    {response.technical_details}
    ### Expected Outcomes:
    {response.expected_outcomes}
"""
    from horizon_bench.save_idea import save_idea
    save_idea(question_filename, idea=res, model_name='ai_researcher')

if __name__ == "__main__":
    run_ai_researcher('2406.04412_1.txt')
    pass