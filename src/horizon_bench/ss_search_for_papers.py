from horizon_bench import semantic_scholar_search
from horizon_bench.arxiv_get_full_text import arxiv_get_full_text
from horizon_bench import Config
NUMBER_OF_SEARCH_RESULTS_PER_QUERY_BEFORE_RERANKING = 6


class RelatedSSPaper(dict):
    def __init__(self, ss_id: str, title: str, abstract: str):
        super().__init__()
        assert ss_id is not None
        assert title is not None
        assert abstract is not None
        self['ss_id'] = ss_id
        self['title'] = title
        self['abstract'] = abstract


class SSPaper(dict):
    def __init__(self, title: str, abstract: str, full_text: str, year: int, ss_id: str, arxiv_id: str, citations: list[RelatedSSPaper], references: list[RelatedSSPaper]):
        super().__init__()
        assert title is not None
        assert abstract is not None
        assert full_text is not None
        assert year is not None
        assert ss_id is not None
        assert arxiv_id is not None
        self['title'] = title
        self['abstract'] = abstract
        self['full_text'] = full_text
        self['year'] = year
        self['ss_id'] = ss_id
        self['arxiv_id'] = arxiv_id
        self['citations'] = citations
        self['references'] = references


SS_SEARCH_FIELDS_DEFAULT = "title,abstract,citationCount,year,externalIds,citations.title,citations.abstract,references.title,references.abstract"

class NoPapersFoundInPostProcessingError(Exception):
    pass

def ss_get_paper(ss_id: str) -> SSPaper:
    """
    Only gets papers that have arxiv id, and all other fields required by SSPaper. 
    """
    params = {"fields": SS_SEARCH_FIELDS_DEFAULT, 'publicationDateOrYear': f':{Config.GLOBAL_KNOWLEDGE_CUTOFF.strftime("%Y-%m-%d")}'}
    ss_search_result = semantic_scholar_search.paper(id=ss_id, params=params)

    paper = post_process_ss_search_results([ss_search_result])
    return paper[0]


def ss_search_for_paper_titles_abstracts(search_query, max_results=None):
    """
    Searches for papers matching the given search query using Semantic Scholar API,
    retrieves their titles.
    returns list of paper titles
    """
    params = {"query":search_query, "fields": "title,abstract", 'publicationDateOrYear': f':{Config.GLOBAL_KNOWLEDGE_CUTOFF.strftime("%Y-%m-%d")}'}
    if max_results is not None:
        params["limit"] = max_results

    ss_search_result = semantic_scholar_search.search_papers(params=params)
    ss_search_result = [{'title': paper['title'], 'abstract': paper['abstract']} for paper in ss_search_result if 'title' in paper and 'abstract' in paper]

    return ss_search_result



def ss_search_for_papers(search_query, search_fields = SS_SEARCH_FIELDS_DEFAULT, max_results=NUMBER_OF_SEARCH_RESULTS_PER_QUERY_BEFORE_RERANKING):
    """
    Searches for papers matching the given search query using Semantic Scholar API,
    retrieves their arXiv IDs, and fetches the full text from arXiv.
    This function observes the GLOBAL_KNOWLEDGE_CUTOFF date defined in Config, so it only retrieves papers published before that date.
    returns list of papers:
    [
        {
            "title": str,
            "abstract": str,
            "citationCount": int,
            "year": int,
            "arxiv_id": str,
            "full_text": str,
            "citations": [ {title, abstract, ...}, ... ],
            "references": [ {title, abstract, ...}, ... ]
        },
        ...
    ]
    """
    params = {"query":search_query, "fields": search_fields, 'publicationDateOrYear': f':{Config.GLOBAL_KNOWLEDGE_CUTOFF.strftime("%Y-%m-%d")}'}
    if max_results is not None:
        params["limit"] = max_results

    ss_search_result = semantic_scholar_search.search_papers(params=params)
    return post_process_ss_search_results(ss_search_result)


def post_process_ss_search_results(ss_search_result: list[dict]) -> list[SSPaper]:
    papers = []
    for paper in ss_search_result:
        arxiv_id = paper['externalIds'].get('ArXiv', None)
        if arxiv_id is None:
            continue
        try:
            plain_text = arxiv_get_full_text(arxiv_id=arxiv_id)
        except Exception as e:
            print(f"Error fetching full text for ArXiv ID {arxiv_id}: {e}")
            continue
        
        citations = []
        raw_citations = paper.get('citations', [])
        for cited_paper in raw_citations:
            try:
                citations.append(RelatedSSPaper(
                    ss_id=cited_paper['paperId'],
                    title=cited_paper['title'],
                    abstract=cited_paper['abstract'],
                ))
            except AssertionError:
                # print(f"Skipping citation {cited_paper}")
                continue

        references = []
        raw_references = paper.get('references', [])
        for reference in raw_references:
            try:
                references.append(RelatedSSPaper(
                    ss_id=reference['paperId'],
                    title=reference['title'],
                    abstract=reference['abstract'],
                ))
            except AssertionError:
                # print(f"Skipping reference {reference}")
                continue

        paper = SSPaper(
            ss_id=paper.get('paperId', ''),
            title=paper.get('title', ''),
            abstract=paper.get('abstract', ''),
            full_text=plain_text,
            year=paper.get('year', 0),
            arxiv_id=arxiv_id,
            citations=citations,
            references=references
        )

        papers.append(paper)
    if not papers:
        raise NoPapersFoundInPostProcessingError("No papers found with required fields after post-processing.")
    return papers



if __name__ == "__main__":
    test_query = "PlanT 2.0: Exposing Biases and Structural Flaws in Closed-Loop Driving"
    test_query = "plant2: Explainable Planning Transformers"
    papers = ss_search_for_paper_titles_abstracts(test_query)
    print(papers)
