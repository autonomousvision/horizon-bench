import os
import re
import threading
import time

import openreview
import requests

from horizon_bench.Config import DOWNLOADED_TXT_DIR, DOWNLOADED_PDF_DIR
from horizon_bench.pdf_full_text import extract_pdf_full_text
from horizon_bench.arxiv_search import wait_for_rate_limit


_download_timestamps: list[float] = []
_download_lock = threading.Lock()
MAX_DOWNLOADS_PER_SECOND = 3


def _rate_limit_download():
    """Block until we can make another download without exceeding 3 requests/second."""
    with _download_lock:
        now = time.monotonic()
        # Remove timestamps older than 1 second
        while _download_timestamps and now - _download_timestamps[0] >= 1.0:
            _download_timestamps.pop(0)
        if len(_download_timestamps) >= MAX_DOWNLOADS_PER_SECOND:
            sleep_time = 1.0 - (now - _download_timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            _download_timestamps.pop(0)
        _download_timestamps.append(time.monotonic())


def get_openreview_client():
    username = os.getenv("OPENREVIEW_USERNAME")
    password = os.getenv("OPENREVIEW_PASSWORD")
    if not username or not password:
        raise ValueError(
            "OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD environment variables must be set."
        )
    return openreview.api.OpenReviewClient(
        baseurl='https://api2.openreview.net',
        username=username,
        password=password,
    )

client = get_openreview_client()

def download_openreview_pdf(openreview_paper_id: str) -> str:
    """Download the PDF of an OpenReview paper given its ID."""
    pdf_path = os.path.join(DOWNLOADED_PDF_DIR, f'{openreview_paper_id}.pdf')
    if os.path.exists(pdf_path):
        return pdf_path
    _rate_limit_download()
    try:
        pdf_binary = client.get_attachment(id=openreview_paper_id, field_name='pdf')
        with open(pdf_path, 'wb') as f:
            f.write(pdf_binary)
        return pdf_path
    except Exception as e:
        raise RuntimeError(f"Failed to download PDF for OpenReview paper ID {openreview_paper_id}: {e}")


# New-style arXiv ids, e.g. 2006.11239 or 2006.11239v2. OpenReview ids never contain a dot.
ARXIV_ID_PATTERN = re.compile(r'^\d{4}\.\d{4,5}(v\d+)?$')


def is_arxiv_id(paper_id: str) -> bool:
    return bool(ARXIV_ID_PATTERN.match(paper_id))


def download_arxiv_pdf(arxiv_id: str) -> str:
    """Download the PDF of an arXiv paper from the arXiv export server."""
    pdf_path = os.path.join(DOWNLOADED_PDF_DIR, f'{arxiv_id}.pdf')
    if os.path.exists(pdf_path):
        return pdf_path
    wait_for_rate_limit()
    pdf_url = f'https://export.arxiv.org/pdf/{arxiv_id}'
    try:
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        if not response.content.startswith(b'%PDF'):
            raise ValueError(f"response from {pdf_url} is not a PDF")
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        return pdf_path
    except Exception as e:
        raise RuntimeError(f"Failed to download PDF for arXiv ID {arxiv_id}: {e}")


def openreview_get_full_text(openreview_paper_id: str) -> str:
    """Download and extract full text from an OpenReview paper.

    Returns the full text. Caches the result as a text file so subsequent
    calls with the same openreview_paper_id skip downloading and parsing.
    """
    txt_path = os.path.join(DOWNLOADED_TXT_DIR, f'{openreview_paper_id}.txt')

    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            return f.read()

    # Download PDF, from the arXiv export server if the id is an arXiv id
    if is_arxiv_id(openreview_paper_id):
        pdf_path = download_arxiv_pdf(openreview_paper_id)
    else:
        pdf_path = download_openreview_pdf(openreview_paper_id)

    # Extract full text using scipdf (requires GROBID running at localhost:8070)
    full_text = extract_pdf_full_text(pdf_path)

    # Cache to txt file
    with open(txt_path, 'w') as f:
        f.write(full_text)

    return full_text


if __name__ == '__main__':
    # client = get_openreview_client()
    # print(client)
    from horizon_bench.pipeline.A_preprocess_iclr_dataset import load_papers
    papers = load_papers()
    for _, row in papers.iterrows():
        openreview_id = row['id']
        try:
            text = download_openreview_pdf(openreview_id)
            print(f"Successfully extracted text for OpenReview ID {openreview_id}")
        except Exception as e:
            print(f"Error processing OpenReview ID {openreview_id}: {e}")
    print(text)
