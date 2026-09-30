from horizon_bench.load_goal_prompt import load_goal_prompt
from horizon_bench.llm_api import get_formatted_chat_response
from horizon_bench.logger import initialize_logger
from horizon_bench.save_idea import save_idea
from horizon_bench.semantic_scholar_search import search_papers, NoSearchResultsError
from ai_scientist_v2.prompts import system_prompt, idea_generation_prompt, idea_reflection_prompt
from ai_scientist_v2.parse_response import format_papers, idea_json_to_markdown
from ai_scientist_v2.response_models import AIScientistResponse

NUM_REFLECTIONS = 5
SEARCH_FIELDS = "title,abstract,citationCount,year,authors,venue"
SEARCH_LIMIT = 10


def run_ai_scientist_v2(question_filename: str):
    initialize_logger("ai_scientist_v2", question_filename)
    task_description = load_goal_prompt(question_filename)

    last_tool_results = ""
    idea_finalized = False
    message_history = [{"role": "system", "content": system_prompt}]
    idea = None

    for reflection_round in range(NUM_REFLECTIONS):
        if reflection_round == 0:
            prompt_text = idea_generation_prompt.format(
                task_description=task_description,
            )
        else:
            prompt_text = idea_reflection_prompt.format(
                current_round=reflection_round + 1,
                num_reflections=NUM_REFLECTIONS,
                last_tool_results=last_tool_results or "No new results.",
            )

        # Add user message to history
        message_history.append({"role": "user", "content": prompt_text})

        # Get structured response
        try:
            response = get_formatted_chat_response(
                response_format=AIScientistResponse,
                user_prompt=prompt_text,
                message_history=message_history[:-1],  # Exclude the user message we just added
            )
        except Exception as e:
            print(f"Failed to get structured response on round {reflection_round + 1}: {e}")
            # Add a placeholder assistant message to maintain conversation flow
            message_history.append({
                "role": "assistant",
                "content": f"Error: {str(e)}"
            })
            last_tool_results = ""
            continue

        # Add assistant response to history (as a summary for context)
        assistant_summary = f"Action: {response.action_type}"
        if response.reasoning:
            assistant_summary += f"\nReasoning: {response.reasoning}"
        message_history.append({"role": "assistant", "content": assistant_summary})

        # Process the action
        if response.action_type == "SearchSemanticScholar":
            if not response.search_query:
                last_tool_results = "Error: No search query provided."
                continue

            print(f"Round {reflection_round + 1}: Searching for '{response.search_query}'")
            try:
                papers = search_papers(params={
                    "query": response.search_query,
                    "fields": SEARCH_FIELDS,
                    "limit": SEARCH_LIMIT,
                })
                last_tool_results = format_papers(papers)
            except NoSearchResultsError:
                last_tool_results = f"No papers found for query: {response.search_query}"
            except Exception as e:
                last_tool_results = f"Error searching Semantic Scholar: {e}"

        elif response.action_type == "FinalizeIdea":
            if response.idea:
                print(f"Round {reflection_round + 1}: Finalizing idea '{response.idea.Name}'")
                idea = response.idea
                idea_finalized = True
                break
            else:
                last_tool_results = "Error: No idea details provided in FinalizeIdea action."

    if idea:
        # Convert Pydantic model to dict for idea_json_to_markdown
        idea_dict = {
            "Name": idea.Name,
            "Title": idea.Title,
            "Short Hypothesis": idea.Short_Hypothesis,
            "Related Work": idea.Related_Work,
            "Abstract": idea.Abstract,
            "Experiments": idea.Experiments,
            "Risk Factors and Limitations": idea.Risk_Factors_and_Limitations,
        }
        markdown_idea = idea_json_to_markdown(idea_dict)
        save_idea(question_filename, idea=markdown_idea, model_name='ai_scientist_v2')
        print(f"Successfully saved idea for {question_filename}")
    else:
        print(f"Warning: AI Scientist v2 failed to generate an idea for {question_filename}")


if __name__ == "__main__":
    run_ai_scientist_v2("1PIfB5w05x_0.txt")