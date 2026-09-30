from horizon_bench.get_gte_embedding import get_paper_embeddings, get_gte_embeddings
from horizon_bench.cosine import get_cosine_similarity

def rerank_papers(papers: list[dict], query: str) -> list[dict]:
    """
    Rerank a list of papers based on their relevance to a query.
    Args:
        papers (list[dict]): A list of papers, each represented as a dictionary with 'title' and 'abstract' keys.
        query (str): The query string to rank the papers against.
    Returns:
        list[dict]: The list of papers ranked by relevance to the query.
    """
    assert len(papers) > 0, "No papers to rerank."
    query_embedding = get_gte_embeddings(query)
    paper_embeddings = get_paper_embeddings(titles=[x['title'] for x in papers], abstracts=[x['abstract'] for x in papers])
    similarities = get_cosine_similarity(query_embedding, paper_embeddings)
    ranked_papers = [paper for _, paper in sorted(zip(similarities, papers), key=lambda pair: pair[0], reverse=True)]
    return ranked_papers


if __name__ == "__main__":
    sample_papers = [
        {"title": "Paper A", "abstract": "This paper discusses machine learning techniques."},
        {"title": "Paper B", "abstract": "This study explores deep learning applications."},
        {"title": "Paper C", "abstract": "An analysis of reinforcement learning methods."}
    ]
    query = "deep learning in computer vision"
    ranked = rerank_papers(sample_papers, query)
    for paper in ranked:
        print(paper['title'])