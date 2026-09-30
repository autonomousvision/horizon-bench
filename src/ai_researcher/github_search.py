from ai_researcher.Config import GITHUB_API_KEY
import requests
from urllib.parse import quote
from datetime import date
from horizon_bench.Config import GLOBAL_KNOWLEDGE_CUTOFF

from joblib import Memory
memory = Memory(".cache", verbose=0)  # Cached files stored in ./.cache/

EXCLUDED_USERS = ["lucidrains"]

def search_github_batched(queries: list) -> list[dict]:
    max_date = GLOBAL_KNOWLEDGE_CUTOFF
    results = []
    for query in queries:
        result = search_github(query, max_date)
        results.append(result)
    return results

@memory.cache
def search_github(query: str, max_date: date, n_results: int = 1) -> list[dict]:
    """
    Search GitHub for repositories matching the query created before max_date.
    Args:
        query (str): The search query.
        max_date (date): The maximum creation date for repositories.
        n_results (int): The number of results to return. 
    Returns:
        list[dict]: A list of dictionaries containing repository information.
        dict_keys = {
                "name":
                "author": 
                "description": 
                "link": 
                "stars": 
                "created_at": 
                "language": 
            }
    """
    query = quote(f"{query} -user:{' '.join(EXCLUDED_USERS)} created:<{max_date.strftime('%Y-%m-%d')}")

    repos = []
    PER_PAGE = 10
    MAX_PAGES_SEARCHED = 1
    for page in range(1, MAX_PAGES_SEARCHED + 1):
        url = f'https://api.github.com/search/repositories?q={query}&per_page={PER_PAGE}&page={page}'

        headers = {
            'Authorization': f'token {GITHUB_API_KEY}',
            'Accept': 'application/vnd.github.v3+json'
        }

        response = requests.get(url, headers=headers)

        if not response.status_code == 200:
            raise Exception(f"GitHub API request failed with status code {response.status_code}: {response.text}")
        
        items = response.json().get('items', [])
        for item in items:
            extracted_info = {
                "name": f"{item['owner']['login']}/{item['name']}",
                "author": item['owner']['login'],
                "description": item['description'],
                "link": item['html_url'], 
                "stars": item['stargazers_count'], 
                "created_at": item['created_at'], 
                "language": item['language']
            }
            repos.append(extracted_info)

            if len(repos) >= n_results:
                break

        if len(repos) >= n_results:
                break
    return repos