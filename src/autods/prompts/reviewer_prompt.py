reviewer_system_prompt = """You are a senior researcher reviewing a submitted research proposal. Be thorough, constructive, and critical. Identify both strengths and weaknesses, and provide actionable suggestions for improvement."""


def get_reviewer_user_prompt(task_description: str, elaborated_idea) -> str:
    idea_text = f"""Title: {elaborated_idea.title}
Motivation: {elaborated_idea.motivation}
Proposed Method: {elaborated_idea.proposed_method}
Technical Details: {elaborated_idea.technical_details}
Expected Outcomes: {elaborated_idea.expected_outcomes}
Novelty Claim: {elaborated_idea.novelty_claim}"""

    return f"""Original task description:
{task_description}

Research proposal to review:
{idea_text}

Provide a structured review:
1. **Strengths**: What are the strong points of this proposal?
2. **Weaknesses**: What are the weak points, gaps, or concerns?
3. **Suggestions**: Concrete, actionable suggestions for improvement
4. **Overall Assessment**: A brief summary of your assessment

Be specific and constructive. Your review will be used to improve the proposal."""
