from horizon_bench.pipeline.D_extract_targets import Challenge, Method, ResearchQuestion
from horizon_bench.llm_api import get_formatted_chat_response
from horizon_bench.Config import INPUT_DIR, TargetType

def naive_baseline(idea: str, target_type: TargetType):
    if target_type == TargetType.CHALLENGE:
        prompt = f"outline the main challenge of the following idea. Answer one sentence, only.\n\n{idea}"
        response = get_formatted_chat_response(
            response_format=Challenge,
            user_prompt=prompt,
        )
    elif target_type == TargetType.METHOD:
        prompt = f"outline the main method of the following idea. Answer one sentence, only.\n\n{idea}"
        response = get_formatted_chat_response(
            response_format=Method,
            user_prompt=prompt,
        )
    elif target_type == TargetType.RESEARCH_QUESTION:
        prompt = f"outline the main research question of the following idea. Answer one sentence, only.\n\n{idea}"
        response = get_formatted_chat_response(
            response_format=ResearchQuestion,
            user_prompt=prompt,
        )
    else:
        raise ValueError(f"Unsupported target type: {target_type}")
    return response 


def load_ideas_from_input_dir() -> dict[str, str]:
    import os
    ideas = dict()
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            arxiv_id = filename[:-4]  # Remove .txt extension
            filepath = os.path.join(INPUT_DIR, filename)
            with open(filepath, "r") as f:
                idea = f.read()
                ideas[arxiv_id] = idea
    return ideas

from horizon_bench.save_idea import save_idea

def run_naive_baseline(goal_filename: str, target_type: TargetType):
    from horizon_bench.load_goal_prompt import load_goal_prompt
    goal_prompt = load_goal_prompt(goal_filename)
    if target_type == TargetType.CHALLENGE:
        prediction = naive_baseline(goal_prompt, target_type).challenge
    elif target_type == TargetType.METHOD:
        prediction = naive_baseline(goal_prompt, target_type).method
    elif target_type == TargetType.RESEARCH_QUESTION:
        prediction = naive_baseline(goal_prompt, target_type).research_question
        
    save_idea(goal_filename, prediction, model_name = 'naive_baseline')
    return prediction



if __name__ == "__main__":
    pass
    