reviser_system_prompt = """You are a research scientist who excels at incorporating feedback to improve research proposals. You take review comments seriously and make substantive improvements while maintaining the core vision of the original idea."""


def get_reviser_user_prompt(task_description: str, elaborated_idea, review) -> str:
    idea_text = f"""Title: {elaborated_idea.title}
Motivation: {elaborated_idea.motivation}
Proposed Method: {elaborated_idea.proposed_method}
Technical Details: {elaborated_idea.technical_details}
Expected Outcomes: {elaborated_idea.expected_outcomes}
Novelty Claim: {elaborated_idea.novelty_claim}"""

    review_text = f"""Strengths: {', '.join(review.strengths)}
Weaknesses: {', '.join(review.weaknesses)}
Suggestions: {', '.join(review.suggestions)}
Overall: {review.overall_assessment}"""

    return f"""Original task description:
{task_description}

Original research proposal:
{idea_text}

Review feedback:
{review_text}

Revise the research proposal to address the weaknesses and incorporate the suggestions from the review. Maintain the core vision but make substantive improvements. Clearly document what changes you made and why."""
