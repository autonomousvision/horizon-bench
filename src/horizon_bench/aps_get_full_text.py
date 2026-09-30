"""Full-text extraction for APS papers.

Unlike OpenReview (which serves PDFs through an API), APS papers are fetched by
downloading the article PDF from the paper's URL. Each APS paper row in the
dataset parquet carries an abstract URL such as

    https://journals.aps.org/prl/abstract/10.1103/rqkg-dw31

whose PDF lives at the sibling ``/pdf/`` path

    https://journals.aps.org/prl/pdf/10.1103/rqkg-dw31

We download that PDF and run it through the same scipdf + GROBID parser used for
OpenReview PDFs.
"""
import os
import threading
import time

import pandas as pd
from curl_cffi import requests

from horizon_bench.Config import DOWNLOADED_TXT_DIR, DOWNLOADED_PDF_DIR
from horizon_bench import Config
from horizon_bench.pdf_full_text import extract_pdf_full_text


class APSFullTextFetchError(Exception):
    pass


_download_timestamps: list[float] = []
_download_lock = threading.Lock()
MAX_DOWNLOADS_PER_SECOND = 1

# APS sits behind Cloudflare bot management, which blocks by TLS/HTTP2
# fingerprint — a browser User-Agent alone is not enough. We use curl_cffi to
# impersonate a recent Chrome's TLS fingerprint; the impersonate profile also
# supplies a matching User-Agent, so no manual headers are needed.
_IMPERSONATE = "chrome136"


def _rate_limit_download():
    """Block until we can make another download without exceeding the rate limit."""
    with _download_lock:
        now = time.monotonic()
        while _download_timestamps and now - _download_timestamps[0] >= 1.0:
            _download_timestamps.pop(0)
        if len(_download_timestamps) >= MAX_DOWNLOADS_PER_SECOND:
            sleep_time = 1.0 - (now - _download_timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            _download_timestamps.pop(0)
        _download_timestamps.append(time.monotonic())


_url_by_id: dict[str, str] | None = None


def _get_paper_url(paper_id: str) -> str:
    """Look up the APS paper URL for a paper id from the dataset parquet."""
    global _url_by_id
    if _url_by_id is None:
        df = pd.read_parquet(Config.DATASET_PARQUET)
        _url_by_id = {
            str(row_id): url
            for row_id, url in zip(df["id"], df["url"])
            if url
        }
    url = _url_by_id.get(str(paper_id))
    if not url:
        raise APSFullTextFetchError(
            f"No APS URL found for paper id {paper_id!r} in {Config.DATASET_PARQUET}"
        )
    return url


def _abstract_url_to_pdf_url(abstract_url: str) -> str:
    """Convert an APS abstract URL to its PDF URL (.../abstract/... -> .../pdf/...)."""
    if "/abstract/" in abstract_url:
        return abstract_url.replace("/abstract/", "/pdf/", 1)
    if "/pdf/" in abstract_url:
        return abstract_url
    raise APSFullTextFetchError(
        f"Unexpected APS URL format, cannot derive PDF URL: {abstract_url!r}"
    )


def download_aps_pdf(paper_id: str) -> str:
    """Download the PDF of an APS paper given its id. Returns the local path."""
    pdf_path = os.path.join(DOWNLOADED_PDF_DIR, f"{paper_id}.pdf")
    if os.path.exists(pdf_path):
        return pdf_path

    pdf_url = _abstract_url_to_pdf_url(_get_paper_url(paper_id))
    _rate_limit_download()
    try:
        resp = requests.get(pdf_url, impersonate=_IMPERSONATE, timeout=60)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower():
            raise APSFullTextFetchError(
                f"URL {pdf_url} did not return a PDF (Content-Type: {content_type!r})"
            )
        with open(pdf_path, "wb") as f:
            f.write(resp.content)
        return pdf_path
    except APSFullTextFetchError:
        raise
    except Exception as e:
        raise APSFullTextFetchError(
            f"Failed to download PDF for APS paper id {paper_id} from {pdf_url}: {e}"
        )


def aps_get_full_text(paper_id: str) -> str:
    """Download and extract full text from an APS paper.

    Caches the result as a text file so subsequent calls with the same paper_id
    skip downloading and parsing.
    """
    txt_path = os.path.join(DOWNLOADED_TXT_DIR, f"{paper_id}.txt")

    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            return f.read()

    pdf_path = download_aps_pdf(paper_id)

    # Extract full text using scipdf (requires GROBID running at localhost:8070)
    full_text = extract_pdf_full_text(pdf_path)

    with open(txt_path, "w") as f:
        f.write(full_text)

    return full_text


if __name__ == "__main__":
    df = pd.read_parquet(Config.DATASET_PARQUET)
    for paper_id in df["id"].head():
        try:
            text = aps_get_full_text(paper_id)
            print(f"Extracted {len(text)} chars for APS paper {paper_id}")
        except Exception as e:
            print(f"Error processing APS paper {paper_id}: {e}")
