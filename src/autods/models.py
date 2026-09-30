from pydantic import BaseModel, Field


class GeneratedIdea(BaseModel):
    """Output of the IdeaGenerator agent. A concise seed idea for MCTS node creation."""
    title: str = Field(description="Short title for the research idea")
    core_concept: str = Field(description="One paragraph describing the core concept")
    research_direction: str = Field(description="The broad research direction this falls under")


class ElaboratedIdea(BaseModel):
    """Output of the IdeaElaborator agent. A fully fleshed-out research proposal."""
    title: str
    motivation: str = Field(description="Why this problem matters and what gap it fills")
    proposed_method: str = Field(description="Detailed technical approach")
    technical_details: str = Field(description="Specific algorithmic or architectural details")
    expected_outcomes: str = Field(description="What results are anticipated")
    novelty_claim: str = Field(description="What is specifically new about this approach")


class IdeaEvaluation(BaseModel):
    """Output of the IdeaEvaluator agent. Structured scores used for belief model updates."""
    novelty_score: float = Field(ge=0.0, le=1.0, description="0-1 score for novelty")
    feasibility_score: float = Field(ge=0.0, le=1.0, description="0-1 score for feasibility")
    significance_score: float = Field(ge=0.0, le=1.0, description="0-1 score for potential impact")
    overall_score: float = Field(ge=0.0, le=1.0, description="0-1 composite quality score")
    rationale: str = Field(description="Brief justification for the scores")


class IdeaReview(BaseModel):
    """Output of the IdeaReviewer agent."""
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    overall_assessment: str


class RevisedIdea(BaseModel):
    """Output of the IdeaReviser agent. The improved idea after incorporating review feedback."""
    title: str
    motivation: str
    proposed_method: str
    technical_details: str
    expected_outcomes: str
    novelty_claim: str
    changes_made: str = Field(description="Summary of what was changed based on review")


class BranchDirections(BaseModel):
    """Output for generating child idea directions from a parent idea."""
    directions: list[str] = Field(description="List of 2-3 distinct research directions branching from the parent idea")
