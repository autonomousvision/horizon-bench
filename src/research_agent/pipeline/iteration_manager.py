"""
Iteration Manager for ResearchAgent.

Manages refinement cycles based on validator feedback.
"""

from typing import Callable, Any, Tuple
from research_agent.models.schemas import ValidationScores
from research_agent.config import MAX_ITERATIONS_PER_PHASE, MIN_VALIDATION_SCORE


def refine_with_feedback(
    generator_func: Callable,
    validator_func: Callable,
    generator_kwargs: dict,
    validator_kwargs: dict,
    max_iterations: int = MAX_ITERATIONS_PER_PHASE,
    min_score: float = MIN_VALIDATION_SCORE,
    phase_name: str = "Unknown"
) -> Tuple[Any, ValidationScores]:
    """
    Iteratively refine generation based on validator feedback.

    Args:
        generator_func: Function that generates output (takes feedback kwarg)
        validator_func: Function that validates output and returns ValidationScores
        generator_kwargs: Keyword arguments for generator (without feedback)
        validator_kwargs: Keyword arguments for validator
        max_iterations: Maximum number of refinement iterations
        min_score: Minimum average score to stop iterating
        phase_name: Name of phase for logging

    Returns:
        Tuple of (best_output, final_validation_scores)
    """
    print(f"\n{'='*60}")
    print(f"Starting {phase_name} Phase")
    print(f"{'='*60}")

    best_output = None
    best_scores = None
    best_avg_score = 0.0

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # Generate output (with feedback from previous iteration if applicable)
        if iteration == 1:
            # First iteration: no feedback
            output = generator_func(**generator_kwargs)
        else:
            # Subsequent iterations: include feedback
            feedback = best_scores.overall_feedback if best_scores else ""
            output = generator_func(**generator_kwargs, feedback=feedback)

        # Convert output to string for validation
        if hasattr(output, 'model_dump_json'):
            output_str = output.model_dump_json(indent=2)
        else:
            output_str = str(output)

        # Validate output
        # Update validator_kwargs with the current output
        validator_call_kwargs = validator_kwargs.copy()
        # Most validators expect the output as the first positional arg or specific kwarg
        # For problem validator: (problem, task_context)
        # For method validator: (method, problem)
        # For experiment validator: (experiment, problem, method)
        # We'll pass output_str as the first value in the validator_kwargs

        scores = validator_func(output_str, **validator_kwargs)

        print(f"Average Score: {scores.average_score:.2f}/10.0")
        print(f"Feedback: {scores.overall_feedback}")

        # Track best output
        if scores.average_score > best_avg_score:
            best_output = output
            best_scores = scores
            best_avg_score = scores.average_score

        # Check if we've met the minimum score threshold
        if scores.average_score >= min_score:
            print(f"✓ Met score threshold ({min_score}). Stopping refinement.")
            break

        # If this is the last iteration, keep the best result
        if iteration == max_iterations:
            print(f"Reached maximum iterations. Using best result (score: {best_avg_score:.2f})")

    print(f"\n{phase_name} Phase Complete")
    print(f"Final Average Score: {best_avg_score:.2f}/10.0")
    print(f"{'='*60}\n")

    return best_output, best_scores


def format_output_for_validation(output: Any) -> str:
    """
    Convert generated output to string format for validation.

    Args:
        output: Generated output (Pydantic model or string)

    Returns:
        String representation for validation
    """
    if isinstance(output, str):
        return output

    # For Pydantic models, format as readable text
    if hasattr(output, 'statement'):  # ProblemIdentification
        parts = [f"Statement: {output.statement}"]
        if output.background:
            parts.append(f"Background: {output.background}")
        parts.append(f"Rationale: {output.rationale}")
        parts.append(f"Novelty: {output.novelty}")
        return "\n\n".join(parts)

    elif hasattr(output, 'technical_approach'):  # MethodDevelopment
        parts = [f"Technical Approach: {output.technical_approach}"]
        parts.append(f"Key Innovations: {output.key_innovations}")
        parts.append(f"Assumptions: {output.assumptions}")
        if output.expected_advantages:
            parts.append(f"Expected Advantages: {output.expected_advantages}")
        return "\n\n".join(parts)

    elif hasattr(output, 'setup'):  # ExperimentDesign
        parts = [f"Setup: {output.setup}"]
        parts.append(f"Metrics: {output.metrics}")
        parts.append(f"Expected Results: {output.expected_results}")
        if output.baselines:
            parts.append(f"Baselines: {output.baselines}")
        return "\n\n".join(parts)

    # Fallback
    return str(output)
