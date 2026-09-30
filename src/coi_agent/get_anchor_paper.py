from horizon_bench.ss_search_for_papers import ss_search_for_papers
from coi_agent.is_paper_relevant_to_topic import is_paper_relevant_to_topic
from horizon_bench.arxiv_get_full_text import arxiv_get_full_text
from horizon_bench.rerank_papers import rerank_papers

class PaperIrrelevantError(Exception):
    pass

def get_anchor_paper(query: str, source_topic: str) -> dict:
    """
    Uses SS API search and reranking to get an anchor paper for a given query.
    Possible exceptions:
    - NoSearchResultsError (if no papers found via ss api)
    - TimeoutError (if ss api times out, it is very slow for big papers >10s)

    Args:
        query (str): The search query.
        source_topic (str): The topic to check relevance against. This is the original user input
    Returns:
        dict: The top-ranked paper matching the query.
        For the structure, see semantic_scholar_search function.
    """
    papers = ss_search_for_papers(query)
    anchor_paper = rerank_papers(papers, query)[0]
    paper_is_relevant = is_paper_relevant_to_topic(topic=source_topic, title=anchor_paper['title'], abstract=anchor_paper['abstract'])
    if not paper_is_relevant:
        raise PaperIrrelevantError(f"The top paper for query '{query}' is not relevant to the topic.")

    anchor_paper['full_text'] = arxiv_get_full_text(arxiv_id=anchor_paper['arxiv_id'])
    return anchor_paper


if __name__ == "__main__":
    query = "Nemotron-CC-Math: A 133 Billion-Token-Scale High Quality Math Pretraining Dataset"
    # source_topic = "graph neural networks"
    # anchor_paper = get_anchor_paper(query, source_topic)
    # print(anchor_paper['title'])
    # print(anchor_paper['arxiv_id'])
    # print(anchor_paper['full_text'][:500])  # Print first 500 characters of full text
    print(ss_search_for_papers(query))