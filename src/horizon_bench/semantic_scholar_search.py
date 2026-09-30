import requests
from tenacity import retry, wait_fixed, retry_if_exception_type, stop_after_attempt
import datetime
import time
from horizon_bench.Config import MEMCACHE_PATH

from joblib import Memory
memory = Memory(MEMCACHE_PATH, verbose=0)

class NoSearchResultsError(Exception):
    """
    If Semantic scholar doesnt find anything relating to the query
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


request_time_queue = []


API_URL = "https://api.semanticscholar.org/graph/v1"
AUTH_HEADER = {"x-api-key": 'qZWKkOKyzP5g9fgjyMmBt1MN2NTC6aT61UklAiyw'}
MAX_REQUESTS_PER_SECOND = 50
TIMEOUT = 120


@memory.cache
def paper(id: str, include_unknown_refs: bool = False, params={}) -> dict:
    """Paper lookup
    :param str id: S2PaperId, DOI or ArXivId.
    :param float timeout: an exception is raised
        if the server has not issued a response for timeout seconds.
    :param bool include_unknown_refs:
        (optional) include non referenced paper.
    :returns: paper data or empty :class:`dict` if not found.
    :rtype: :class:`dict`
    """
    data = __get_data("paper", id, include_unknown_refs, params=params)
    if data == []:
        raise NoSearchResultsError("No paper found for the given id.")
    return data

@memory.cache
def author(id: str, params={}) -> dict:
    """Author lookup
    :param str id: S2AuthorId.
    :returns: author data or empty :class:`dict` if not found.
    :rtype: :class:`dict`
    """
    data = __get_data("author", id, False, params=params)
    return data

@retry(
    wait=wait_fixed(30),
    retry=retry_if_exception_type(ConnectionRefusedError),
    stop=stop_after_attempt(10),
)
def __get_data(
    method: str, id: str, include_unknown_refs: bool, params={}
) -> dict:
    """Get data from Semantic Scholar API
    :param str method: 'paper' or 'author'.
    :param str id: id of the corresponding method.
    :returns: data or empty :class:`dict` if not found.
    :rtype: :class:`dict`
    """
    # We are going to make an API request. Let's add it to our queue of request times
    data = {}
    method_types = ["paper", "author"]
    if method not in method_types:
        raise ValueError(f"Invalid method type. Expected one of: {method_types}")

    url = f"{API_URL}/{method}/{id}"
    if include_unknown_refs:
        params["include_unknown_references"] = "true"

    while len(request_time_queue) >= MAX_REQUESTS_PER_SECOND:
        time.sleep(0.05)
        try_to_remove_from_timing_queue()

    request_time_queue.append(datetime.datetime.now())

    # If you really want to use retries of this get request, do it outside of this function
    r = requests.get(
        url, timeout=TIMEOUT, headers=AUTH_HEADER, params=params
    )
    data = check_response_status(r)
    return data

@memory.cache
def search_authors(params: dict) -> dict:
    """
    Search for authors data
    :param params: dictionary with search parameters such as:
    query: string search query, containing the author name
    fields: name, etc
    :return res: dictionary with results
    """
    data = __get_data("author", "search", False, params=params)
    return data

@memory.cache
def search_papers(params: dict) -> dict:
    """
    Search for papers data
    :param query: string search query
    :param params: search parameters
    :return res: dictionary with results
    """
    data = __get_data("paper", "search", False, params=params)
    if not "data" in data:
        raise NoSearchResultsError("No papers found for the given query.")
    return data['data']

def try_to_remove_from_timing_queue():
    """
    Checks if any old requests can be removed from our request time queue.
    The request time queue makes sure that we don't exceed our API request quota of 100 requests per second.
    """
    if len(request_time_queue) >= MAX_REQUESTS_PER_SECOND:
        # Delete any requests older than a second from the queue
        # Our quota is n requests per second, so anything older than a second is irrelevant
        # The len(request_time_queue) == 0 case is only relevant for max_requests_per_second == 1 (so practically never, only for my unittest)
        while len(request_time_queue) > 0 and request_time_queue[
            0
        ] < datetime.datetime.now() - datetime.timedelta(seconds=1):
            request_time_queue.pop(0)


def check_response_status(r):
    data = {}
    if r.status_code == 200:
        data = r.json()
        if (len(data) == 1 and "error" in data) \
            or ("data" in data and len(data["data"]) == 0):
            raise NoSearchResultsError(f"No data found\n{r}")
    elif r.status_code == 403:
        raise PermissionError("HTTP status 403 Forbidden.")
    elif r.status_code == 429:
        raise ConnectionRefusedError("HTTP status 429 Too Many Requests.")
    elif r.status_code == 504:
        raise TimeoutError("HTTP status 504 Request Timeout")
    return data


if __name__ == "__main__":
    params = {"fields": "title,venue,year,publicationTypes", "arXiv": "1906.08237"}
    params =  {"query":"'Diffusion-based generation of street networks", "fields": "title,abstract,citationCount,year,externalIds,citations.title,citations.abstract,references.title,references.abstract", "limit": 3}
    response = search_papers(params=params)
