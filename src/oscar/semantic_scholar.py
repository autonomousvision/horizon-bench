from horizon_bench.ss_search_for_papers import ss_search_for_paper_titles_abstracts
from horizon_bench.llm_api import get_formatted_chat_response
from pydantic import BaseModel, Field


class SemanticScholarSearchQuery(BaseModel):
    queries: list[str] = Field(description="A list of search queries to find the most relevant papers on Semantic Scholar. Each query should be a concise phrase targeting a specific aspect of the research goal.")


def predict_ss_query(task_description: str) -> list[str]:
    system_prompt = "You are a research assistant. Given a research goal, predict effective Semantic Scholar search queries that would find the most relevant papers."
    user_prompt = f"Generate a list of search queries that would find the papers most related to the following research goal:\n\n{task_description}"

    response = get_formatted_chat_response(
        response_format=SemanticScholarSearchQuery,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )
    return response.queries

class RelevanceJudgement(BaseModel):
    relevant_indices: list[int] = Field(description="Indices (0-based) of papers that are highly related to the research goal.")


def search_ss_until_n_papers_found(task_description: str, search_queries: list[str], n: int = 3) -> list[dict]:
    """Search Semantic Scholar for 2*n candidates across all queries, then select the top n via a single LLM judgement."""
    candidates = []
    seen_titles = set()

    # Phase 1: Gather 2*n unique candidates across all queries
    target_candidates = 2 * n
    for query in search_queries:
        if len(candidates) >= target_candidates:
            break
        print(f"Searching: '{query}'")
        try:
            results = ss_search_for_paper_titles_abstracts(query, max_results=2*n)
            results = [r for r in results if (r['title'] is not None and r['abstract'] is not None)][:n]
        except Exception as e:
            print(f"  Search failed: {e}")
            continue

        for paper in results:
            if paper['title'] not in seen_titles:
                seen_titles.add(paper['title'])
                candidates.append(paper)
                if len(candidates) >= target_candidates:
                    break

    if not candidates:
        raise ValueError(f"No papers found for any of the search queries. Queries: {search_queries}")

    print(f"Collected {len(candidates)} unique candidates")

    # Phase 2: Single LLM call to select the top n
    papers_list = "\n".join(
        f"[{i}] {p['title']}\n    Abstract: {p['abstract'][:4000]}"
        for i, p in enumerate(candidates)
    )
    judgement = get_formatted_chat_response(
        response_format=RelevanceJudgement,
        user_prompt=f"Research goal:\n{task_description}\n\nCandidate papers:\n{papers_list}\n\nSelect the {n} most relevant papers to the research goal. Return their indices. Only include papers that are directly relevant.",
        system_prompt="You are a research assistant judging paper relevance. Be selective — only pick papers that are highly related to the research goal.",
    )

    found_papers = []
    for idx in judgement.relevant_indices:
        if idx < len(candidates):
            found_papers.append(candidates[idx])
            if len(found_papers) >= n:
                break

    return found_papers

