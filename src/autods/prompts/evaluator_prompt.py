evaluator_system_prompt = """You are a rigorous scientific evaluator. Your task is to objectively assess research ideas on multiple dimensions. Be honest and calibrated in your scores -- avoid uniformly high or low ratings. A score of 0.5 represents an average-quality idea."""


def get_evaluator_user_prompt(task_description: str, elaborated_idea) -> str:
    idea_text = f"""Title: {elaborated_idea.title}
Motivation: {elaborated_idea.motivation}
Proposed Method: {elaborated_idea.proposed_method}
Technical Details: {elaborated_idea.technical_details}
Expected Outcomes: {elaborated_idea.expected_outcomes}
Novelty Claim: {elaborated_idea.novelty_claim}"""

    return f"""Original task description:
{task_description}

Research idea to evaluate:
{idea_text}

Evaluate this research idea on the following dimensions, providing a score from 0.0 to 1.0 for each:

1. **Novelty** (0.0-1.0): How original is this idea? Does it propose genuinely new methods, combinations, or perspectives? Score 0.5 for incremental improvements, 0.8+ for truly novel approaches.

2. **Feasibility** (0.0-1.0): How realistic is the proposed approach? Can it be implemented with existing tools and knowledge? Score 0.5 for approaches with some unknowns, 0.8+ for clearly implementable approaches.

3. **Significance** (0.0-1.0): How impactful could this be? Would it advance the field meaningfully? Score 0.5 for modest contributions, 0.8+ for potentially transformative work.

4. **Overall** (0.0-1.0): Your holistic assessment considering all factors above plus technical soundness, clarity, and completeness.

Be calibrated and critical. Most ideas should fall in the 0.3-0.7 range. Only exceptional ideas deserve 0.8+."""
