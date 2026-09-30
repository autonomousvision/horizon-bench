"""
Prompts for Method Development phase (Phase 2) of ResearchAgent.
"""


def get_method_development_prompt(
    problem_statement: str,
    papers_context: str,
    entities_context: str,
    feedback: str = None
) -> str:
    """
    Generate prompt for method development.

    Args:
        problem_statement: The research problem from Phase 1
        papers_context: Formatted string of relevant papers
        entities_context: Formatted string of relevant entities
        feedback: Optional feedback from previous iteration

    Returns:
        Prompt string
    """
    prompt = f"""You are a research idea generator tasked with developing novel methods to address research problems.

**Research Problem**:
{problem_statement}

**Relevant Literature**:
{papers_context}

{entities_context}

Your task is to develop a **technical method or approach** that addresses the research problem. The method should:
1. Directly tackle the core challenges of the problem
2. Be technically sound and feasible to implement
3. Incorporate insights from the relevant literature
4. Introduce novel elements or innovations
5. Have clear advantages over existing approaches

"""

    if feedback:
        prompt += f"""
**Feedback from Previous Iteration**:
{feedback}

Please refine your method based on this feedback.

"""

    prompt += """
Provide:
1. **Technical Approach**: Detailed description of your proposed method (4-6 sentences)
   - What is the core idea?
   - How does it work technically?
   - What are the key algorithmic or architectural components?

2. **Key Innovations**: What makes this method novel (2-3 sentences)
   - What is new compared to existing approaches?
   - What insights enable this method?

3. **Assumptions**: Key assumptions your method makes (2-3 sentences)
   - What conditions must hold for the method to work?
   - What are the limitations of these assumptions?

4. **Expected Advantages** (optional): Why this method should outperform existing approaches

Be specific and technical. Avoid vague descriptions.
"""

    return prompt
