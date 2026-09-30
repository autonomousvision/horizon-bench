import math
from dataclasses import dataclass, field
from typing import Optional

from autods.beliefs import BetaBelief
from autods.models import ElaboratedIdea, IdeaEvaluation


@dataclass
class MCTSNode:
    """
    A node in the MCTS tree representing a research idea.

    Root node: created from the task description (no idea yet).
    Child nodes: each represents a specific research idea direction.
    """
    idea_direction: str
    parent: Optional['MCTSNode'] = None
    children: list['MCTSNode'] = field(default_factory=list)

    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0

    # Belief model for this idea's quality
    prior_belief: BetaBelief = field(default_factory=BetaBelief)
    posterior_belief: BetaBelief = field(default_factory=BetaBelief)

    # Idea content (populated after simulation)
    elaborated_idea: Optional[ElaboratedIdea] = None
    evaluation: Optional[IdeaEvaluation] = None
    revised_idea: Optional[ElaboratedIdea] = None

    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def average_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def bayesian_surprise(self) -> float:
        """KL divergence from prior to posterior — measures information gain."""
        return self.posterior_belief.kl_divergence_from(self.prior_belief)

    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """UCB1 selection criterion balancing exploitation and exploration."""
        if self.visit_count == 0:
            return float('inf')

        parent_visits = self.parent.visit_count if self.parent else self.visit_count
        exploitation = self.average_value
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploitation + exploration


def select(node: MCTSNode) -> MCTSNode:
    """Selection phase: traverse tree using UCB1 until a leaf node is reached."""
    current = node
    while not current.is_leaf:
        current = max(current.children, key=lambda c: c.ucb1_score())
    return current


def expand(node: MCTSNode, branch_directions: list[str]) -> list[MCTSNode]:
    """
    Expansion phase: create child nodes for the given research directions.

    Uses progressive widening: only expand if visit_count >= len(children).
    """
    if node.visit_count < len(node.children):
        return []

    new_children = []
    for direction in branch_directions:
        child = MCTSNode(
            idea_direction=direction,
            parent=node,
            depth=node.depth + 1,
            prior_belief=node.posterior_belief.copy(),
        )
        child.posterior_belief = child.prior_belief.copy()
        node.children.append(child)
        new_children.append(child)
    return new_children


def backpropagate(node: MCTSNode, value: float) -> None:
    """Backpropagation phase: update visit counts and values up to the root."""
    current = node
    while current is not None:
        current.visit_count += 1
        current.total_value += value
        current.posterior_belief.update(value)
        current = current.parent


def get_best_idea(root: MCTSNode) -> MCTSNode:
    """
    After MCTS iterations, select the best idea by Bayesian surprise (KL divergence).
    Falls back to average value as tiebreaker.
    Only considers nodes that have been evaluated.
    """
    all_evaluated = []
    _collect_evaluated_nodes(root, all_evaluated)

    if not all_evaluated:
        return root

    return max(all_evaluated, key=lambda n: (n.bayesian_surprise, n.average_value))


def _collect_evaluated_nodes(node: MCTSNode, result: list) -> None:
    """Recursively collect all nodes that have been evaluated."""
    if node.evaluation is not None:
        result.append(node)
    for child in node.children:
        _collect_evaluated_nodes(child, result)
