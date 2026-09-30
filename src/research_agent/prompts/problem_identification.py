"""
Prompts for Problem Identification phase (Phase 1) of ResearchAgent.
"""


def get_problem_identification_prompt(
    task_description: str,
    papers_context: str,
    entities_context: str,
    feedback: str = None
) -> str:
    """
    Generate prompt for problem identification.

    Args:
        task_description: Original research topic/task
        papers_context: Formatted string of relevant papers
        entities_context: Formatted string of relevant entities
        feedback: Optional feedback from previous iteration

    Returns:
        Prompt string
    """
    prompt = f"""You are a research idea generator tasked with identifying novel and significant research problems.

**Research Topic**:
{task_description}

**Relevant Literature**:
{papers_context}

{entities_context}

Your task is to identify a **specific, well-defined research problem** that:
1. Is grounded in the current literature and research topic
2. Represents a gap or limitation in existing work
3. Is novel and has not been fully addressed
4. Is significant and would advance the field if solved
5. Is feasible to investigate with current methods and resources

"""

    if feedback:
        prompt += f"""
**Feedback from Previous Iteration**:
{feedback}

Please refine your problem statement based on this feedback.

"""

    prompt += """
Provide:
1. **Statement**: A clear, concise problem statement (2-3 sentences)
2. **Rationale**: Why this problem is important and worth investigating (3-4 sentences)
3. **Novelty**: What makes this problem novel or underexplored (2-3 sentences)
4. **Background** (optional): Additional context from literature that motivates this problem

Focus on creating a problem that is specific enough to be actionable but broad enough to be significant.
"""

    return prompt
