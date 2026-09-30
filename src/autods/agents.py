from horizon_bench.llm_api import get_formatted_chat_response
from autods.models import (
    GeneratedIdea, ElaboratedIdea, IdeaEvaluation,
    IdeaReview, RevisedIdea, BranchDirections
)
from autods.prompts.generator_prompt import (
    generator_system_prompt, get_generator_user_prompt, get_branch_directions_prompt
)
from autods.prompts.elaborator_prompt import elaborator_system_prompt, get_elaborator_user_prompt
from autods.prompts.evaluator_prompt import evaluator_system_prompt, get_evaluator_user_prompt
from autods.prompts.reviewer_prompt import reviewer_system_prompt, get_reviewer_user_prompt
from autods.prompts.reviser_prompt import reviser_system_prompt, get_reviser_user_prompt


def generate_seed_ideas(task_description: str, n_ideas: int = 3) -> list[GeneratedIdea]:
    """
    IdeaGenerator agent: Generate n_ideas seed research ideas from the task description.
    Maps to AutoDS's ExperimentGenerator role.
    """
    ideas = []
    for i in range(n_ideas):
        user_prompt = get_generator_user_prompt(
            task_description=task_description,
            existing_ideas=[idea.title for idea in ideas],
            idea_index=i + 1,
            total_ideas=n_ideas,
        )
        idea = get_formatted_chat_response(
            response_format=GeneratedIdea,
            user_prompt=user_prompt,
            system_prompt=generator_system_prompt,
        )
        ideas.append(idea)
    return ideas


def generate_branch_directions(task_description: str, parent_direction: str, parent_elaboration: str = None) -> list[str]:
    """Generate child idea directions branching from a parent idea. Used during MCTS expansion."""
    user_prompt = get_branch_directions_prompt(
        task_description=task_description,
        parent_direction=parent_direction,
        parent_elaboration=parent_elaboration,
    )
    result = get_formatted_chat_response(
        response_format=BranchDirections,
        user_prompt=user_prompt,
        system_prompt=generator_system_prompt,
    )
    return result.directions


def elaborate_idea(task_description: str, idea_direction: str) -> ElaboratedIdea:
    """
    IdeaElaborator agent: Takes a seed idea direction and produces a detailed research proposal.
    Maps to AutoDS's ExperimentProgrammer + CodeExecutor.
    """
    user_prompt = get_elaborator_user_prompt(
        task_description=task_description,
        idea_direction=idea_direction,
    )
    return get_formatted_chat_response(
        response_format=ElaboratedIdea,
        user_prompt=user_prompt,
        system_prompt=elaborator_system_prompt,
    )


def evaluate_idea(task_description: str, elaborated_idea: ElaboratedIdea) -> IdeaEvaluation:
    """
    IdeaEvaluator agent: Scores the elaborated idea on novelty, feasibility,
    significance, and overall quality. Maps to AutoDS's ExperimentAnalyst.
    The overall_score feeds into the MCTS belief model.
    """
    user_prompt = get_evaluator_user_prompt(
        task_description=task_description,
        elaborated_idea=elaborated_idea,
    )
    return get_formatted_chat_response(
        response_format=IdeaEvaluation,
        user_prompt=user_prompt,
        system_prompt=evaluator_system_prompt,
    )


def review_idea(task_description: str, elaborated_idea: ElaboratedIdea) -> IdeaReview:
    """IdeaReviewer agent: Provides structured critique. Maps to AutoDS's ExperimentReviewer."""
    user_prompt = get_reviewer_user_prompt(
        task_description=task_description,
        elaborated_idea=elaborated_idea,
    )
    return get_formatted_chat_response(
        response_format=IdeaReview,
        user_prompt=user_prompt,
        system_prompt=reviewer_system_prompt,
    )


def revise_idea(task_description: str, elaborated_idea: ElaboratedIdea, review: IdeaReview) -> RevisedIdea:
    """IdeaReviser agent: Improves the idea based on review feedback. Maps to AutoDS's ExperimentReviser."""
    user_prompt = get_reviser_user_prompt(
        task_description=task_description,
        elaborated_idea=elaborated_idea,
        review=review,
    )
    return get_formatted_chat_response(
        response_format=RevisedIdea,
        user_prompt=user_prompt,
        system_prompt=reviser_system_prompt,
    )
