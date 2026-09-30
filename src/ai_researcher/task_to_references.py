from pydantic import BaseModel
from horizon_bench.ss_search_for_papers import ss_search_for_paper_titles_abstracts
from horizon_bench.rerank_papers import rerank_papers

class ReferencePapersResult(BaseModel):
    reference_papers: list[str]

class SemanticScholarSearchQueriesResult(BaseModel):
    ss_search_queries: list[str]


def task_to_references(task_description: str) -> list[str]:
    """
    No caching required, as subfunctions already have caching.
    """
    query = 'list of the most relevant search queries for semantic scholar to find relevant related work for this task: ' + task_description
    from horizon_bench.llm_api import get_formatted_chat_response
    reference_papers_response = get_formatted_chat_response(user_prompt=query,
                                                            response_format=SemanticScholarSearchQueriesResult,
                                                            )
    gpt_references = reference_papers_response.ss_search_queries

    ss_search_queries = [task_description]
    ss_search_queries.extend(gpt_references)
    
    added_titles = set()
    references = []
    for query in ss_search_queries:
        try:
            ss_references = ss_search_for_paper_titles_abstracts(query, max_results=10)
        except Exception as e:
            print(f"No search results")
            continue
        if ss_references == []:
            continue
        for reference in ss_references:
            if reference['title'] not in added_titles:
                references.append(reference)
                added_titles.add(reference['title'])

    references = rerank_papers(references, task_description)
    references = [paper['title'] for paper in references]
    return references


if __name__ == "__main__":
    TASK_DESCRIPTION = "Generate an idea for a novel scientific impact prediction model."
    references = task_to_references(TASK_DESCRIPTION)
    print('\n'.join(references))
    