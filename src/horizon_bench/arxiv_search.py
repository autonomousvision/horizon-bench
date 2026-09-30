from typing import List
import urllib.parse
import feedparser
import requests
import os
import re
import tarfile
from horizon_bench.Config import ARXIV_REQUESTS_PER_MINUTE, DOWNLOADED_TEX_DIR, DOWNLOADED_TXT_DIR, DOWNLOADED_TAR_DIR, MEMCACHE_PATH
from horizon_bench import Config
from horizon_bench.rerank_papers import rerank_papers

class ArxivSearchError(Exception):
    def __init__(self, message):
        super().__init__(message)

class ArxivDownloadError(Exception):
    def __init__(self, message):
        super().__init__(message)

from joblib import Memory
memory = Memory(MEMCACHE_PATH, verbose=0) 

@memory.cache
def search_download_arxiv(paper_list: List[str]):
    """
    download arxiv paper source file by title
    
    Args:
        title: paper title
        paper_dir: paper directory
    """
    res = []
    for title in paper_list:
        papers = search_arxiv(title, max_results=1)
        if len(papers) == 0:
            res.append(ArxivSearchError(title))
            continue
        paper = papers[0]

        if paper['published'] > str(Config.GLOBAL_KNOWLEDGE_CUTOFF):
            continue

        try:
            download_arxiv_tex(paper['url'])
            res.append({'title': title,
                        'path': os.path.join(DOWNLOADED_TEX_DIR, f"{paper['arxiv_id']}.tex")
                        })
        except Exception as e:
            res.append(ArxivDownloadError(str(e)))
    return res

def string_format_arxiv_path_title(results: list[dict]) -> str:
    formatted_results = []
    for res in results:
        if isinstance(res, ArxivSearchError) or isinstance(res, ArxivDownloadError):
            continue
        else:
            formatted_results.append(f"Title: {res['title']}, Path: {res['path']}")
    return "\n".join(formatted_results)


from datetime import datetime, timedelta
from time import sleep
MAX_ARXIV_SEARCH_RESULTS_PER_SECOND = ARXIV_REQUESTS_PER_MINUTE / 60.0
last_arxiv_search = datetime.now() - timedelta(seconds=1/MAX_ARXIV_SEARCH_RESULTS_PER_SECOND)

@memory.cache
def search_arxiv(query, max_results=10, search_field='ti'):
    """
    search arxiv papers
    
    Args:
        query (str): search keyword
        max_results (int): max return results
        
    Returns:
        list: list of papers info
    """
    global last_arxiv_search
    if last_arxiv_search + timedelta(seconds=1/MAX_ARXIV_SEARCH_RESULTS_PER_SECOND) > datetime.now():
        sleep(1/MAX_ARXIV_SEARCH_RESULTS_PER_SECOND)

    # 构建API URL
    base_url = 'http://export.arxiv.org/api/query?'
    # search_query = urllib.parse.quote(query)

    # 设置API参数
    params = {
        'search_query': f'{search_field}:{query}',
        'start': 0,
        'max_results': max(10, max_results), # At least fetch 10 to allow reranking
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
    
    query_url = base_url + urllib.parse.urlencode(params)
    
    response = feedparser.parse(query_url)
    last_arxiv_search = datetime.now()
    
    papers = []
    for entry in response.entries:
        paper = {
            'title': entry.title,
            'author': [author.name for author in entry.authors],
            'published': entry.published,
            'abstract': entry.summary,
            'url': entry.link,
            'arxiv_id': entry.id.split('/abs/')[-1],
        }
        papers.append(paper)

    papers = rerank_papers(papers, query)
        
    return papers[:max_results]


def arxiv_get(arxiv_id: str):
    """
    get arxiv paper info by arxiv id
    
    Args:
        arxiv_id: arxiv paper id, e.g. '2006.11239v2'
        
    Returns:
        dict: paper info
    """
    papers = search_arxiv(arxiv_id, max_results=1, search_field='id')
    if len(papers) == 0:
        raise ArxivSearchError(f"Arxiv paper not found: {arxiv_id}")
    return papers[0]

from datetime import datetime, timedelta
from time import sleep
MAX_ARXIV_REQUESTS_PER_SECOND = ARXIV_REQUESTS_PER_MINUTE / 60.0
last_arxiv_request = datetime.now() - timedelta(seconds=1/MAX_ARXIV_REQUESTS_PER_SECOND)


def wait_for_rate_limit():
    global last_arxiv_request
    # Rate limiting to respect arXiv's request limits
    time_since_last_request = (datetime.now() - last_arxiv_request).total_seconds()
    min_interval = 1 / MAX_ARXIV_REQUESTS_PER_SECOND
    if time_since_last_request < min_interval:
        sleep(min_interval - time_since_last_request)
    last_arxiv_request = datetime.now()


@memory.cache
def download_arxiv_tex(arxiv_id: str):
    """
    download arxiv paper source file
    
    Args:
        arxiv_url: arxiv paper url, e.g. 'http://export.arxiv.org/abs/2006.11239v2'
        local_root: local root directory
        workplace_name: workplace name
    """
    assert isinstance(arxiv_id, str), "arxiv_id should be str, like '2006.11239v2'"
    tar_filepath = os.path.join(DOWNLOADED_TAR_DIR, f"{arxiv_id}.tar.gz")
    if os.path.exists(os.path.join(DOWNLOADED_TEX_DIR, f"{arxiv_id}.tex")):
        return  # already downloaded
    
    wait_for_rate_limit()

    arxiv_url = f'http://export.arxiv.org/abs/{arxiv_id}'
    paper_id = re.search(r'abs/([^/]+)', arxiv_url).group(1)
    source_url = f'http://export.arxiv.org/src/{paper_id}'
    response = requests.get(source_url)
    
    if response.status_code == 200:
        with open(tar_filepath, 'wb') as f:
            f.write(response.content)
        tex_content = _extract_arxiv_tar(tar_filepath)
        # TODO: refactor, to rename this folder. need to check ai_researcher
        os.makedirs(DOWNLOADED_TEX_DIR, exist_ok=True)
        with open(os.path.join(DOWNLOADED_TEX_DIR, f"{arxiv_id}.tex"), 'w') as f:
            f.write(tex_content)


def _extract_arxiv_tar(tar_path):
    try:
        all_content = []
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tex_files = [f for f in tar.getmembers() if f.name.endswith('.tex')]
            
            for tex_file in tex_files:
                f = tar.extractfile(tex_file)
                if f is not None:
                    try:
                        content = f.read().decode('utf-8')
                    except UnicodeDecodeError:
                        f.seek(0)
                        content = f.read().decode('latin-1')

                    all_content.append(f"\n{'='*50}\nFilename: {tex_file.name}\n{'='*50}\n")
                    all_content.append(content)
                    all_content.append("\n\n")
        return "".join(all_content)
    
    except Exception as e:
        return f"Extract failed with error: {str(e)}"
    
def convert_tex_to_txt(arxiv_id: str):
    """
    convert arxiv tex file to txt file
    
    Args:
        arxiv_id: arxiv paper id, e.g. '2006.11239v2'
    """
    tex_file_path = os.path.join(DOWNLOADED_TEX_DIR, f"{arxiv_id}.tex")
    txt_file_path = os.path.join(DOWNLOADED_TXT_DIR, f"{arxiv_id}.txt")

    # if size of tex file too large, skip conversion
    if os.path.getsize(tex_file_path) > 10 * 1024 * 1024:
        with open(txt_file_path, 'w') as f:
            f.write("Tex file too large to convert.")
        return

    if os.path.exists(txt_file_path):
        return  # already converted
    if not os.path.exists(tex_file_path):
        raise FileNotFoundError(f"Tex file not found: {tex_file_path}")

    command = f"timeout 20 pandoc -s {tex_file_path} -t plain -o {txt_file_path}"
    result = os.system(command)
    if result != 0:
        raise RuntimeError(f"Pandoc conversion failed for {tex_file_path}")

if __name__ == "__main__":
    print(arxiv_get('2006.11239v2'))