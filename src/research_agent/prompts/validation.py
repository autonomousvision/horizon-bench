"""
Prompts for Validation agents across all three phases.
"""

from research_agent.models.validation_criteria import (
    format_metrics_for_prompt,
    get_metrics_for_phase,
)


def get_validation_prompt(
    phase: str,
    generated_content: str,
    task_context: str = ""
) -> str:
    """
    Generate validation prompt for any phase.

    Args:
        phase: Phase name ("problem", "method", or "experiment")
        generated_content: The content to validate
        task_context: Additional context (e.g., research topic, problem statement)

    Returns:
        Validation prompt string
    """
    metrics = get_metrics_for_phase(phase)
    metrics_formatted = format_metrics_for_prompt(metrics)

    prompt = f"""You are an expert research reviewer tasked with evaluating a {phase} {'identification' if phase == 'problem' else 'development' if phase == 'method' else 'design'}.

**Context**:
{task_context}

**{phase.capitalize()} to Evaluate**:
{generated_content}

Your task is to evaluate this {phase} on the following criteria:

{metrics_formatted}

For each metric:
1. Assign a score from 1-10 (1 = very poor, 10 = excellent)
2. Provide brief feedback explaining the score (1-2 sentences)

Also provide:
- **Overall Feedback**: Constructive suggestions for improvement (2-3 sentences)

Be critical but fair. Focus on actionable feedback that can improve the {phase}.
"""

    return prompt


def get_problem_validation_prompt(problem: str, task_context: str) -> str:
    """Generate validation prompt specifically for problem identification."""
    return get_validation_prompt(
        phase="problem",
        generated_content=problem,
        task_context=f"Research Topic: {task_context}"
    )


def get_method_validation_prompt(method: str, problem: str) -> str:
    """Generate validation prompt specifically for method development."""
    return get_validation_prompt(
        phase="method",
        generated_content=method,
        task_context=f"Research Problem: {problem}"
    )


def get_experiment_validation_prompt(experiment: str, problem: str, method: str) -> str:
    """Generate validation prompt specifically for experiment design."""
    context = f"""Research Problem: {problem}

Proposed Method: {method}"""

    return get_validation_prompt(
        phase="experiment",
        generated_content=experiment,
        task_context=context
    )
