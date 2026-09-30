"""
LLM as a judge. Used just for AI Scientist. 
There is a separate version for Horizon Bench in src/horizon_bench/
"""

from ai_scientist.llm_as_a_judge.load_few_shot_examples import get_review_fewshot_examples
from ai_scientist.llm_as_a_judge.judge_prompt import judge_user_prompt, judge_system_prompt
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


def llm_as_a_judge(idea_text, n_fewshot=1, n_ensemble=1):
    
    fewshot_prompt = ""
    if n_fewshot > 0:
        fewshot_prompt = get_review_fewshot_examples(num_fs_examples=n_fewshot)

    user_prompt = judge_user_prompt + fewshot_prompt

    user_prompt += f"""
Here is the paper you are asked to review:
```
{idea_text}
```"""
    from horizon_bench.llm_api import get_formatted_chat_response

    responses = []
    for _ in range(n_ensemble):
        responses.append(get_formatted_chat_response(
        user_prompt=user_prompt,
        system_prompt=judge_system_prompt,
        response_format=JudgeReview,
        )
        )

    # AI Scientist provides LLM aggregation of results. 
    # But since we use OpenAIs parser, we can average the numerical scores directly.
    return JudgeReview(
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
    
if __name__ == "__main__":
    test_idea = '\nName: mode_conditioned_diffusion_v5\nTitle: Unified, robust mode-conditioned diffusion with adaptive conditioning policy for multimodal 2D data\nExperiment: 1) Data: load four datasets, balance modes, create a mixed dataset with labels 0..3. 2) Model: implement a single conditioning engine supporting three modalities: (i) concatenation: embed(mode) -> concat to x/time embeddings; (ii) FiLM: embed(mode) -> gamma/beta; (iii) mode-token cross-attention: a small attention block that fuses mode information into activations. Introduce a lightweight gating network G(mode) -> weights w1,w2,w3 for combining modalities. Include conditioning_strength schedule (warmup) and an EMA for mode embeddings to improve stability. 3) Training: use mixed batches with random modes; apply conditioning via the gated combination; enable classifier-free conditioning with dropout p_cf. The gating policy can either follow G predictions or be fixed by a simple heuristic; gradient clipping is used. 4) Sampling: support hard conditioning to target_mode, soft conditioning via weighted combination of modalities, and adaptive modality weights from G. 5) Evaluation: per-mode KL, MMD-like metrics, lightweight real-vs-generated classifier accuracy, interpolation smoothness along paths in embedding/attn space, and an analysis of the gating weights across runs. 6) Ablations: compare all three conditioning modalities with and without gating, test fixed vs adaptive policy, vary embedding dimension and warmup schedule, test with/without p_cf. 7) Visualization: attention maps for cross-attention variant, and qualitative interpolation trajectories. 8) Feasibility: moderate complexity; aims for a robust, reusable conditioning module with a principled evaluation suite. 9) Significance: delivers a practical, reproducible, and interpretable framework for controllable diffusion across multimodal 2D datasets, with insights into when and why different conditioning schemes work best.\n'

    review = llm_as_a_judge(test_idea, n_fewshot=1, n_ensemble=1, llm_accumulation=False)
