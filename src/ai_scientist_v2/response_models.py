"""Pydantic models for structured AI Scientist v2 responses."""

from pydantic import BaseModel, Field
from typing import Literal


class IdeaDetails(BaseModel):
    """Structure for the final research idea."""
    Name: str = Field(description="A short descriptor of the idea. Lowercase, no spaces, underscores allowed.")
    Title: str = Field(description="A catchy and informative title for the proposal.")
    Short_Hypothesis: str = Field(
        alias="Short Hypothesis",
        description="A concise statement of the main hypothesis or research question."
    )
    Related_Work: str = Field(
        alias="Related Work",
        description="A brief discussion of the most relevant related work and how the proposal distinguishes from it."
    )
    Abstract: str = Field(description="An abstract that summarizes the proposal in conference format (approximately 250 words).")
    Experiments: str = Field(description="A list of experiments that would be conducted to validate the proposal.")
    Risk_Factors_and_Limitations: str = Field(
        alias="Risk Factors and Limitations",
        description="A list of potential risks and limitations of the proposal."
    )


class SearchSemanticScholarAction(BaseModel):
    """Response for searching Semantic Scholar."""
    action: Literal["SearchSemanticScholar"] = Field(description="The action to take")
    query: str = Field(description="The search query for Semantic Scholar")
    reasoning: str = Field(description="Brief explanation of why you're searching for this", default="")


class FinalizeIdeaAction(BaseModel):
    """Response for finalizing the research idea."""
    action: Literal["FinalizeIdea"] = Field(description="The action to take")
    idea: IdeaDetails = Field(description="The complete research idea details")
    reasoning: str = Field(description="Brief explanation of the final idea", default="")


class AIScientistResponse(BaseModel):
    """Union type for AI Scientist v2 responses."""
    action_type: Literal["SearchSemanticScholar", "FinalizeIdea"] = Field(
        description="The type of action to take"
    )
    search_query: str | None = Field(
        default=None,
        description="Search query if action is SearchSemanticScholar"
    )
    idea: IdeaDetails | None = Field(
        default=None,
        description="Idea details if action is FinalizeIdea"
    )
    reasoning: str = Field(
        default="",
        description="Your reasoning for this action"
    )
