"""
Prompts for Experiment Design phase (Phase 3) of ResearchAgent.
"""


def get_experiment_design_prompt(
    problem_statement: str,
    method_description: str,
    papers_context: str,
    feedback: str = None
) -> str:
    """
    Generate prompt for experiment design.

    Args:
        problem_statement: The research problem from Phase 1
        method_description: The proposed method from Phase 2
        papers_context: Formatted string of relevant papers
        feedback: Optional feedback from previous iteration

    Returns:
        Prompt string
    """
    prompt = f"""You are a research idea generator tasked with designing experiments to validate research methods.

**Research Problem**:
{problem_statement}

**Proposed Method**:
{method_description}

**Relevant Literature**:
{papers_context}

Your task is to design a **rigorous experimental setup** that validates the proposed method. The experiment should:
1. Test the method's ability to address the research problem
2. Compare against relevant baselines from the literature
3. Use appropriate datasets and evaluation metrics
4. Be reproducible and clearly specified
5. Cover multiple scenarios or settings for robustness

"""

    if feedback:
        prompt += f"""
**Feedback from Previous Iteration**:
{feedback}

Please refine your experiment design based on this feedback.

"""

    prompt += """
Provide:
1. **Experimental Setup**: Detailed description of the experimental design (4-6 sentences)
   - What datasets will be used?
   - What is the experimental protocol?
   - What variations or ablations will be tested?

2. **Evaluation Metrics**: How will you measure success? (2-3 sentences)
   - What metrics are most appropriate?
   - Why are these metrics suitable for this problem?

3. **Expected Results**: What outcomes do you anticipate? (2-3 sentences)
   - What results would validate the method?
   - What would constitute strong performance?

4. **Baselines** (optional): What existing methods will you compare against?

Be concrete and specific. Specify actual datasets, metrics, and protocols where possible.
"""

    return prompt
