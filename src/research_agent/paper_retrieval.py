"""
Paper retrieval for ResearchAgent.

Converts Horizon Bench's single-text topics into relevant paper lists
by searching Semantic Scholar and downloading arXiv full text.

Uses ss_search_for_papers to ensure GLOBAL_KNOWLEDGE_CUTOFF is respected.
"""

from horizon_bench.ss_search_for_papers import ss_search_for_papers, NoPapersFoundInPostProcessingError
from horizon_bench.semantic_scholar_search import NoSearchResultsError
from horizon_bench.rerank_papers import rerank_papers
from research_agent.config import MAX_PAPERS, PAPER_RETRIEVAL_TIMEOUT
import signal


class PaperRetrievalError(Exception):
    """Raised when paper retrieval fails."""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Paper retrieval timed out")


def extract_keywords_from_topic(topic: str) -> str:
    """
    Extract search keywords from topic.

    Creates concise search queries by extracting key technical terms.

    Args:
        topic: Research topic description

    Returns:
        Search query string for Semantic Scholar
    """
    # Remove common filler words and phrases
    filler_words = [
        "we", "are", "is", "the", "a", "an", "to", "for",
        "in", "on", "at", "with", "by", "from", "of", "and",
        "developing", "using", "via", "through"
    ]

    # Split and filter
    words = topic.lower().split()
    keywords = [w for w in words if w not in filler_words and len(w) > 2]

    # Take only the most distinctive words (first 4-6 meaningful keywords)
    # This creates more focused queries
    keywords = keywords[:6]

    # Join and limit total length
    query = " ".join(keywords)
    if len(query) > 60:
        query = query[:60].rsplit(' ', 1)[0]  # Cut at word boundary

    return query


def topic_to_papers(topic: str, max_papers: int = MAX_PAPERS) -> list[dict]:
    """
    Convert research topic to list of relevant papers.

    Uses ss_search_for_papers which respects GLOBAL_KNOWLEDGE_CUTOFF.

    Args:
        topic: Single-sentence research topic from Horizon Bench input
        max_papers: Number of papers to retrieve (default from config)

    Returns:
        List of paper dictionaries with fields:
        - paperId: Semantic Scholar paper ID (ss_id)
        - title: Paper title
        - abstract: Paper abstract
        - year: Publication year
        - citations: List of citing papers (limited to 5)
        - references: List of referenced papers (limited to 5)
        - full_text: Full text from arXiv
        - arxiv_id: arXiv ID

    Raises:
        PaperRetrievalError: If paper retrieval fails
    """
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(PAPER_RETRIEVAL_TIMEOUT)

    try:
        # Step 1: Extract keywords from topic
        query = extract_keywords_from_topic(topic)
        print(f"Searching Semantic Scholar for: {query} (respecting knowledge cutoff)")

        # Step 2: Search using ss_search_for_papers (automatically observes knowledge cutoff)
        # Request extra papers (2x) to ensure we have enough after reranking
        search_limit = max_papers * 2

        try:
            papers = ss_search_for_papers(search_query=query, max_results=search_limit)
        except (NoPapersFoundInPostProcessingError, NoSearchResultsError) as e:
            print(f"No papers found for query '{query}': {e}")
            # Fallback: try a simpler query with just first 3 words
            simple_query = " ".join(topic.split()[:3])
            print(f"Retrying with simpler query: {simple_query}")
            try:
                papers = ss_search_for_papers(search_query=simple_query, max_results=search_limit)
            except (NoPapersFoundInPostProcessingError, NoSearchResultsError):
                # If still no results, try the full topic as-is
                print(f"Still no results. Trying full topic as query: {topic[:50]}...")
                try:
                    papers = ss_search_for_papers(search_query=topic, max_results=search_limit)
                except (NoPapersFoundInPostProcessingError, NoSearchResultsError):
                    raise PaperRetrievalError(f"No papers found for topic after multiple attempts: {topic}")

        if not papers:
            raise PaperRetrievalError(f"No papers found for topic: {topic}")

        print(f"Found {len(papers)} papers with arXiv full text")

        # Step 3: Rerank by relevance to topic
        if len(papers) > max_papers:
            print(f"Reranking papers by relevance to topic...")
            ranked_papers = rerank_papers(papers, topic)
            papers = ranked_papers[:max_papers]

        print(f"Selected {len(papers)} papers for ResearchAgent")

        # Step 4: Format papers for ResearchAgent
        # ss_search_for_papers returns SSPaper objects with all fields populated
        formatted_papers = []
        for paper in papers:
            formatted = {
                "paperId": paper['ss_id'],  # Map ss_id to paperId for consistency
                "title": paper['title'],
                "abstract": paper['abstract'],
                "year": paper['year'],
                "arxiv_id": paper['arxiv_id'],
                "full_text": paper['full_text'][:10000],  # Limit to first 10000 chars
                "citations": paper.get('citations', [])[:5],  # Limit to 5
                "references": paper.get('references', [])[:5],  # Limit to 5
            }
            print(f"  ✓ {paper['title'][:60]}... ({paper['year']})")
            formatted_papers.append(formatted)

        # Cancel timeout
        signal.alarm(0)

        return formatted_papers

    except TimeoutError:
        raise PaperRetrievalError(f"Paper retrieval timed out after {PAPER_RETRIEVAL_TIMEOUT}s")
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        raise PaperRetrievalError(f"Error retrieving papers: {e}")


def format_papers_for_context(papers: list[dict], include_full_text: bool = False) -> str:
    """
    Format papers for inclusion in LLM prompts.

    Args:
        papers: List of paper dictionaries
        include_full_text: Whether to include full text (can be very long)

    Returns:
        Formatted string representation of papers
    """
    formatted = []
    for i, paper in enumerate(papers, 1):
        formatted.append(f"**Paper {i}: {paper['title']}**")
        formatted.append(f"Year: {paper.get('year', 'N/A')}")
        formatted.append(f"Abstract: {paper['abstract']}")

        if include_full_text and "full_text" in paper:
            formatted.append(f"Full Text (excerpt): {paper['full_text'][:2000]}...")

        formatted.append("")  # Blank line

    return "\n".join(formatted)
