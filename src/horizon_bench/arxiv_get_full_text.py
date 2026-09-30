import os
from horizon_bench.Config import DOWNLOADED_TXT_DIR, MEMCACHE_PATH
from horizon_bench.arxiv_search import download_arxiv_tex, convert_tex_to_txt

from joblib import Memory
memory = Memory(MEMCACHE_PATH, verbose=0) 

class ArxivFullTextFetchError(Exception):
    pass

@memory.cache
def arxiv_get_full_text(arxiv_id: str) -> str:
    """
    Given an arXiv ID, fetch the full text of the paper as a string.
    
    Args:
        arxiv_id: The arXiv identifier of the paper (e.g., '2006.11239v2').
    """
    try:
        download_arxiv_tex(arxiv_id)
        convert_tex_to_txt(arxiv_id)
        with open(os.path.join(DOWNLOADED_TXT_DIR, f"{arxiv_id}.txt"), 'r') as f:
            return f.read()
    except Exception as e:
        raise ArxivFullTextFetchError(f"Failed to fetch full text for arXiv ID {arxiv_id}: {e}")
        


if __name__ == "__main__":
    arxiv_id = "2"
    full_text = arxiv_get_full_text(arxiv_id)
    print(full_text[:1500])  # Print the first 500 characters of the full text