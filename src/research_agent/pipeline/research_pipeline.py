"""
Research Pipeline for ResearchAgent.

Orchestrates the three-phase iterative refinement process:
1. Problem Identification
2. Method Development
3. Experiment Design
"""

from research_agent.models.schemas import (
    KnowledgeEntity,
    ResearchOutput,
)
from research_agent.agents.problem_identifier import ProblemIdentifier
from research_agent.agents.problem_validator import ProblemValidator
from research_agent.agents.method_developer import MethodDeveloper
from research_agent.agents.method_validator import MethodValidator
from research_agent.agents.experiment_designer import ExperimentDesigner
from research_agent.agents.experiment_validator import ExperimentValidator
from research_agent.pipeline.iteration_manager import refine_with_feedback, format_output_for_validation
from research_agent.paper_retrieval import format_papers_for_context
from research_agent.knowledge_builder import format_entities_for_context, get_relevant_entities
from research_agent.config import GENERATOR_MODEL, VALIDATOR_MODEL


def run_research_pipeline(
    papers: list[dict],
    entities: list[KnowledgeEntity],
    task_description: str
) -> ResearchOutput:
    """
    Run the complete three-phase research pipeline.

    Args:
        papers: List of relevant papers
        entities: List of knowledge entities from papers
        task_description: Original research task description

    Returns:
        ResearchOutput with problem, method, experiment, and validation scores
    """
    print("\n" + "="*80)
    print("RESEARCHAGENT PIPELINE")
    print("="*80)

    # Format papers for context
    papers_context = format_papers_for_context(papers)

    # ===== Phase 1: Problem Identification =====
    print("\n### PHASE 1: Problem Identification ###\n")

    # Get relevant entities for problem context
    problem_entities = get_relevant_entities(entities, task_description)
    entities_context = format_entities_for_context(problem_entities)

    # Initialize agents
    problem_identifier = ProblemIdentifier(model=GENERATOR_MODEL)
    problem_validator = ProblemValidator(model=VALIDATOR_MODEL)

    # Run refinement loop
    problem, problem_scores = refine_with_feedback(
        generator_func=problem_identifier.generate,
        validator_func=problem_validator.generate,
        generator_kwargs={
            "task_description": task_description,
            "papers_context": papers_context,
            "entities_context": entities_context,
        },
        validator_kwargs={
            "task_context": task_description,
        },
        phase_name="Problem Identification"
    )

    # ===== Phase 2: Method Development =====
    print("\n### PHASE 2: Method Development ###\n")

    # Get relevant entities for method context
    problem_statement = format_output_for_validation(problem)
    method_entities = get_relevant_entities(entities, problem_statement)
    entities_context = format_entities_for_context(method_entities)

    # Initialize agents
    method_developer = MethodDeveloper(model=GENERATOR_MODEL)
    method_validator = MethodValidator(model=VALIDATOR_MODEL)

    # Run refinement loop
    method, method_scores = refine_with_feedback(
        generator_func=method_developer.generate,
        validator_func=method_validator.generate,
        generator_kwargs={
            "problem_statement": problem_statement,
            "papers_context": papers_context,
            "entities_context": entities_context,
        },
        validator_kwargs={
            "problem": problem_statement,
        },
        phase_name="Method Development"
    )

    # ===== Phase 3: Experiment Design =====
    print("\n### PHASE 3: Experiment Design ###\n")

    # Initialize agents
    experiment_designer = ExperimentDesigner(model=GENERATOR_MODEL)
    experiment_validator = ExperimentValidator(model=VALIDATOR_MODEL)

    # Format method for context
    method_description = format_output_for_validation(method)

    # Run refinement loop
    experiment, experiment_scores = refine_with_feedback(
        generator_func=experiment_designer.generate,
        validator_func=experiment_validator.generate,
        generator_kwargs={
            "problem_statement": problem_statement,
            "method_description": method_description,
            "papers_context": papers_context,
        },
        validator_kwargs={
            "problem": problem_statement,
            "method": method_description,
        },
        phase_name="Experiment Design"
    )

    # ===== Compile Final Output =====
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Problem Score: {problem_scores.average_score:.2f}/10.0")
    print(f"Method Score: {method_scores.average_score:.2f}/10.0")
    print(f"Experiment Score: {experiment_scores.average_score:.2f}/10.0")
    print(f"Overall Average: {(problem_scores.average_score + method_scores.average_score + experiment_scores.average_score) / 3:.2f}/10.0")
    print("="*80 + "\n")

    return ResearchOutput(
        problem=problem,
        method=method,
        experiment=experiment,
        problem_validation=problem_scores,
        method_validation=method_scores,
        experiment_validation=experiment_scores,
    )
