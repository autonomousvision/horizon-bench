from horizon_bench.load_goal_prompt import load_goal_prompt
from horizon_bench.save_idea import save_idea
from horizon_bench.logger import initialize_logger
from autods.mcts import MCTSNode, select, expand, backpropagate, get_best_idea
from autods.agents import (
    generate_seed_ideas, generate_branch_directions,
    elaborate_idea, evaluate_idea, review_idea, revise_idea
)
from autods.models import ElaboratedIdea


# MCTS Configuration
MAX_ITERATIONS = 6
N_SEED_IDEAS = 3
MAX_DEPTH = 2
ENABLE_REVISION = True


def run_autods(question_filename: str) -> None:
    """
    Main entry point for the AutoDS agent.

    Adapts AutoDiscovery's MCTS-based hypothesis exploration for research idea generation.
    Uses Bayesian surprise (KL divergence) to select the most informative idea.

    Algorithm:
    1. Create root node from task description
    2. Generate seed ideas as root children (initial expansion)
    3. Run MCTS iterations: select -> expand -> simulate -> backpropagate
    4. Select best idea by Bayesian surprise
    5. Format and save output
    """
    initialize_logger("autods", question_filename)
    task_description = load_goal_prompt(question_filename)

    # Phase 1: Initialize tree
    print("Phase 1: Initializing tree with seed ideas...")
    root = MCTSNode(idea_direction=task_description, depth=0)

    seed_ideas = generate_seed_ideas(task_description, n_ideas=N_SEED_IDEAS)
    for seed in seed_ideas:
        child = MCTSNode(
            idea_direction=f"{seed.title}: {seed.core_concept}",
            parent=root,
            depth=1,
        )
        root.children.append(child)

    # Phase 2: MCTS iterations
    print("Phase 2: Running MCTS iterations...")
    for iteration in range(MAX_ITERATIONS):
        # SELECT: find most promising leaf
        leaf = select(root)

        # EXPAND: if leaf has been visited and not at max depth, generate children
        if leaf.visit_count > 0 and leaf.depth < MAX_DEPTH:
            elaboration_text = None
            if leaf.elaborated_idea is not None:
                elaboration_text = leaf.elaborated_idea.proposed_method

            directions = generate_branch_directions(
                task_description=task_description,
                parent_direction=leaf.idea_direction,
                parent_elaboration=elaboration_text,
            )
            new_children = expand(leaf, directions)

            if new_children:
                leaf = new_children[0]

        # SIMULATE: elaborate and evaluate the idea
        elaborated = elaborate_idea(task_description, leaf.idea_direction)
        evaluation = evaluate_idea(task_description, elaborated)

        leaf.elaborated_idea = elaborated
        leaf.evaluation = evaluation

        # Optional: review and revise cycle
        if ENABLE_REVISION:
            review = review_idea(task_description, elaborated)
            revised = revise_idea(task_description, elaborated, review)
            leaf.revised_idea = ElaboratedIdea(
                title=revised.title,
                motivation=revised.motivation,
                proposed_method=revised.proposed_method,
                technical_details=revised.technical_details,
                expected_outcomes=revised.expected_outcomes,
                novelty_claim=revised.novelty_claim,
            )
            revised_eval = evaluate_idea(task_description, leaf.revised_idea)
            value = max(evaluation.overall_score, revised_eval.overall_score)
            if revised_eval.overall_score > evaluation.overall_score:
                leaf.elaborated_idea = leaf.revised_idea
                leaf.evaluation = revised_eval
        else:
            value = evaluation.overall_score

        # BACKPROPAGATE: update scores up the tree
        backpropagate(leaf, value)

    # Phase 3: Select best idea and save
    print("Phase 3: Selecting best idea and saving output...")
    best_node = get_best_idea(root)
    idea = best_node.revised_idea if best_node.revised_idea else best_node.elaborated_idea

    if idea is None:
        seed = generate_seed_ideas(task_description, n_ideas=1)[0]
        elaborated = elaborate_idea(task_description, f"{seed.title}: {seed.core_concept}")
        output_text = _format_idea(elaborated)
    else:
        output_text = _format_idea(idea)

    save_idea(question_filename, idea=output_text, model_name='autods')


def _format_idea(idea: ElaboratedIdea) -> str:
    """Format an ElaboratedIdea into a markdown string for saving."""
    return f"""### Title
{idea.title}

### Motivation
{idea.motivation}

### Proposed Method
{idea.proposed_method}

### Technical Details
{idea.technical_details}

### Expected Outcomes
{idea.expected_outcomes}

### Novelty
{idea.novelty_claim}"""


if __name__ == "__main__":
    run_autods("2408.03314_0.txt")