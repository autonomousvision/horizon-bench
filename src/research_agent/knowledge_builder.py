"""
Knowledge entity extraction and co-occurrence scoring for ResearchAgent.

Extracts key entities from research papers and builds co-occurrence matrices
to identify related concepts for grounding idea generation.
"""

import torch
from pydantic import BaseModel
from typing import Optional
from horizon_bench.llm_api import get_formatted_chat_response
from research_agent.config import (
    MAX_ENTITIES_PER_PAPER,
    TOP_K_RELEVANT_ENTITIES,
    GENERATOR_MODEL,
)
from research_agent.models.schemas import KnowledgeEntity, EntityExtractionResult


class EntityList(BaseModel):
    """List of entities extracted from a paper."""
    entities: list[str]


def extract_entities_from_paper(paper: dict) -> EntityExtractionResult:
    """
    Extract knowledge entities from a single paper.

    Args:
        paper: Paper dictionary with title, abstract, and optionally full_text

    Returns:
        EntityExtractionResult with extracted entities
    """
    # Build prompt for entity extraction
    prompt = f"""Extract key knowledge entities from the following research paper. Focus on:
- Novel methods or algorithms
- Datasets or benchmarks
- Technical techniques or approaches
- Evaluation metrics
- Domain-specific concepts

Extract up to {MAX_ENTITIES_PER_PAPER} of the most important entities.

**Paper Title**: {paper['title']}

**Abstract**: {paper['abstract']}
"""

    # If full text is available, include a snippet
    if "full_text" in paper:
        prompt += f"\n\n**Full Text (excerpt)**: {paper['full_text'][:1000]}..."

    prompt += f"\n\nProvide a list of {MAX_ENTITIES_PER_PAPER} key entities (brief phrases, 1-4 words each)."

    try:
        result = get_formatted_chat_response(
            response_format=EntityList,
            user_prompt=prompt,
            # Note: model and temperature not supported by Horizon Bench's get_formatted_chat_response
        )

        return EntityExtractionResult(
            paper_id=paper['paperId'],
            entities=result.entities
        )

    except Exception as e:
        print(f"Error extracting entities from paper {paper['paperId']}: {e}")
        # Return empty result on error
        return EntityExtractionResult(
            paper_id=paper['paperId'],
            entities=[]
        )


def extract_entities_from_papers(papers: list[dict]) -> list[KnowledgeEntity]:
    """
    Extract entities from a list of papers and build entity index.

    Args:
        papers: List of paper dictionaries

    Returns:
        List of KnowledgeEntity objects with occurrence counts
    """
    print(f"Extracting entities from {len(papers)} papers...")

    # Extract entities from each paper
    all_extractions = []
    for paper in papers:
        extraction = extract_entities_from_paper(paper)
        all_extractions.append(extraction)

    # Aggregate entities across papers
    entity_to_papers = {}  # Maps entity to list of paper IDs where it appears

    for extraction in all_extractions:
        for entity in extraction.entities:
            # Normalize entity (lowercase for matching)
            normalized = entity.lower().strip()
            if normalized not in entity_to_papers:
                entity_to_papers[normalized] = []
            entity_to_papers[normalized].append(extraction.paper_id)

    # Build KnowledgeEntity objects
    knowledge_entities = []
    for entity, paper_ids in entity_to_papers.items():
        knowledge_entities.append(
            KnowledgeEntity(
                entity=entity,
                occurrences=len(paper_ids),
                co_occurrence_scores={}  # Will be populated by build_cooccurrence_matrix
            )
        )

    print(f"Extracted {len(knowledge_entities)} unique entities")

    # Build co-occurrence matrix
    knowledge_entities = build_cooccurrence_matrix(knowledge_entities, all_extractions)

    return knowledge_entities


def build_cooccurrence_matrix(
    entities: list[KnowledgeEntity],
    extractions: list[EntityExtractionResult]
) -> list[KnowledgeEntity]:
    """
    Build co-occurrence matrix for entities.

    Entities that appear in the same papers get higher co-occurrence scores.

    Args:
        entities: List of KnowledgeEntity objects
        extractions: List of EntityExtractionResult objects

    Returns:
        Updated list of KnowledgeEntity objects with co_occurrence_scores populated
    """
    if not entities:
        return entities

    print("Building entity co-occurrence matrix...")

    # Create mapping from entity to index
    entity_to_idx = {e.entity: i for i, e in enumerate(entities)}

    # Create co-occurrence matrix using torch
    n_entities = len(entities)
    cooccurrence_matrix = torch.zeros((n_entities, n_entities))

    # Build co-occurrence counts
    for extraction in extractions:
        # Normalize entities in this paper
        paper_entities = [e.lower().strip() for e in extraction.entities]

        # For each pair of entities in the same paper, increment co-occurrence
        for i, entity1 in enumerate(paper_entities):
            if entity1 not in entity_to_idx:
                continue

            idx1 = entity_to_idx[entity1]

            for entity2 in paper_entities[i + 1:]:
                if entity2 not in entity_to_idx:
                    continue

                idx2 = entity_to_idx[entity2]

                # Increment co-occurrence (symmetric)
                cooccurrence_matrix[idx1, idx2] += 1
                cooccurrence_matrix[idx2, idx1] += 1

    # Normalize co-occurrence scores
    # Use Jaccard-like coefficient: co-occurrence / (occurrence1 + occurrence2 - co-occurrence)
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i == j:
                continue

            cooccur_count = cooccurrence_matrix[i, j].item()
            if cooccur_count > 0:
                # Jaccard coefficient
                occur1 = entity1.occurrences
                occur2 = entity2.occurrences
                jaccard = cooccur_count / (occur1 + occur2 - cooccur_count)

                entity1.co_occurrence_scores[entity2.entity] = jaccard

    print("Co-occurrence matrix built successfully")

    return entities


def get_relevant_entities(
    entities: list[KnowledgeEntity],
    context: str,
    top_k: int = TOP_K_RELEVANT_ENTITIES
) -> list[str]:
    """
    Get most relevant entities for a given context.

    Uses simple keyword matching and co-occurrence scores.

    Args:
        entities: List of KnowledgeEntity objects
        context: Context string (e.g., problem statement)
        top_k: Number of top entities to return

    Returns:
        List of top-k most relevant entity strings
    """
    if not entities:
        return []

    context_lower = context.lower()

    # Score entities by relevance
    entity_scores = []

    for entity in entities:
        score = 0.0

        # Direct keyword match
        if entity.entity in context_lower:
            score += 10.0  # Strong signal

        # Partial match (entity appears as substring)
        if any(word in context_lower.split() for word in entity.entity.split()):
            score += 5.0

        # Frequency boost
        score += entity.occurrences * 0.5

        entity_scores.append((entity.entity, score))

    # Sort by score and return top-k
    entity_scores.sort(key=lambda x: x[1], reverse=True)
    top_entities = [entity for entity, score in entity_scores[:top_k]]

    return top_entities


def format_entities_for_context(entities: list[str]) -> str:
    """
    Format entities for inclusion in LLM prompts.

    Args:
        entities: List of entity strings

    Returns:
        Formatted string representation
    """
    if not entities:
        return "No specific entities identified."

    return "**Relevant Concepts from Literature**: " + ", ".join(entities)
