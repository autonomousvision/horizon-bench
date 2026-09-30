"""
Pydantic models for ResearchAgent structured outputs.

Defines data models for all three phases (Problem, Method, Experiment)
and their associated validation scores and knowledge entities.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ===== Knowledge Entity Models =====

class KnowledgeEntity(BaseModel):
    """Represents a knowledge entity extracted from research papers."""
    entity: str = Field(description="The entity text (e.g., method name, technique, dataset)")
    occurrences: int = Field(description="Number of papers where this entity appears")
    co_occurrence_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Co-occurrence scores with other entities"
    )


# ===== Phase 1: Problem Identification =====

class ProblemIdentification(BaseModel):
    """Generated research problem from Phase 1."""
    statement: str = Field(description="Clear statement of the research problem")
    rationale: str = Field(description="Rationale explaining why this problem is important")
    novelty: str = Field(description="What makes this problem novel or underexplored")
    background: Optional[str] = Field(
        default=None,
        description="Background context from literature"
    )


# ===== Phase 2: Method Development =====

class MethodDevelopment(BaseModel):
    """Generated research method from Phase 2."""
    technical_approach: str = Field(description="Technical approach to solve the problem")
    key_innovations: str = Field(description="Key innovations or novel aspects of the method")
    assumptions: str = Field(description="Key assumptions made by the method")
    expected_advantages: Optional[str] = Field(
        default=None,
        description="Expected advantages over existing methods"
    )


# ===== Phase 3: Experiment Design =====

class ExperimentDesign(BaseModel):
    """Generated experiment design from Phase 3."""
    setup: str = Field(description="Experimental setup and configuration")
    metrics: str = Field(description="Evaluation metrics to measure success")
    expected_results: str = Field(description="Expected outcomes and results")
    baselines: Optional[str] = Field(
        default=None,
        description="Baseline methods for comparison"
    )


# ===== Validation Models =====

class MetricScore(BaseModel):
    """Score for a single validation metric."""
    metric_name: str = Field(description="Name of the metric (e.g., 'Clarity', 'Novelty')")
    score: float = Field(description="Score from 1-10", ge=1.0, le=10.0)
    feedback: str = Field(description="Textual feedback explaining the score")


class ValidationScores(BaseModel):
    """Validation scores for a generated output."""
    scores: list[MetricScore] = Field(description="List of metric scores")
    average_score: float = Field(description="Average score across all metrics", ge=1.0, le=10.0)
    overall_feedback: str = Field(description="Overall assessment and suggestions for improvement")

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-calculate average if not provided
        if 'average_score' not in data and self.scores:
            self.average_score = sum(s.score for s in self.scores) / len(self.scores)


# ===== Entity Extraction Models =====

class EntityExtractionResult(BaseModel):
    """Result from entity extraction on a paper."""
    paper_id: str = Field(description="Paper identifier")
    entities: list[str] = Field(description="List of extracted entities")
    entity_types: Optional[dict[str, str]] = Field(
        default=None,
        description="Mapping of entity to its type (method, dataset, technique, etc.)"
    )


# ===== Complete Research Output =====

class ResearchOutput(BaseModel):
    """Complete output from all three phases."""
    problem: ProblemIdentification
    method: MethodDevelopment
    experiment: ExperimentDesign
    problem_validation: ValidationScores
    method_validation: ValidationScores
    experiment_validation: ValidationScores
