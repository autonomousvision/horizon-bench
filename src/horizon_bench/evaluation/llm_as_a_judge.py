"""
LLM as a judge. Used just for AI Scientist. 
There is a separate version for Horizon Bench in src/horizon_bench/
"""

import os
from horizon_bench.llm_api import get_formatted_chat_response
from horizon_bench.evaluation.load_idea import load_idea
from horizon_bench.prompts.judge_prompt import judge_user_prompt, judge_system_prompt
from pydantic import BaseModel

class JudgeReview(BaseModel):
    summary: str
    questions: str
    limitations: str
    ethical_concerns: bool
    soundness: int
    presentation: int
    contribution: int
    overall: int
    confidence: int


def llm_as_a_judge(idea_filepath: str, n_fewshot=1, n_ensemble=1):
    idea_text = load_idea(idea_filepath)
    user_prompt = judge_user_prompt.format(idea_text=idea_text)
    responses = []
    for _ in range(n_ensemble):
        responses.append(get_formatted_chat_response(
            user_prompt=user_prompt,
            system_prompt=judge_system_prompt,
            response_format=JudgeReview,
        ))

    # AI Scientist provides LLM aggregation of results. 
    # But since we use OpenAIs parser, we can average the numerical scores directly.
    res = JudgeReview(
        summary="\n\n".join([r.summary for r in responses]),
        questions="\n\n".join([r.questions for r in responses]),
        limitations="\n\n".join([r.limitations for r in responses]),
        ethical_concerns=False,
        soundness=sum([r.soundness for r in responses]) / n_ensemble,
        presentation=sum([r.presentation for r in responses]) / n_ensemble,
        contribution=sum([r.contribution for r in responses]) / n_ensemble,
        overall=sum([r.overall for r in responses]) / n_ensemble,
        confidence=sum([r.confidence for r in responses]) / n_ensemble,
    )
    save(idea_filepath, res)
    return res
    
def save(idea_filepath: str, review: JudgeReview):
    import os
    import json
    output_path = os.path.join(*os.path.split(idea_filepath)[:-1], 'llm_as_a_judge', f"{os.path.split(idea_filepath)[-1].replace('.txt', '.json')}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(review.dict(), f, indent=4)
    return output_path

if __name__ == "__main__":
    from horizon_bench.Config import DATA_ROOT
    print(llm_as_a_judge(os.path.join(DATA_ROOT, "ground_truth", "2508.13148.json")))