"""Preprocess an APS journal issue into an iclr-like papers parquet.

Downloads the title, abstract and publication date for every paper listed in a
Physical Review Letters (or other APS journal) issue, e.g.

    https://journals.aps.org/prl/issues/135/26

The individual paper/abstract pages (e.g.
https://journals.aps.org/prl/abstract/10.1103/rqkg-dw31) are protected by
Cloudflare and cannot be scraped directly. Instead we pull the exact same
metadata (title, abstract, authors, publication date) from OpenAlex, which
indexes the full issue by volume/issue and reconstructs abstracts. The APS
abstract URL is preserved on each row for traceability.

Output columns mirror ``data/iclr.parquet`` as closely as possible so the rest
of the pipeline can consume APS papers the same way it consumes ICLR papers.
"""
import re
import html

import pandas as pd
import requests

from horizon_bench.Config import WORKSPACE_PATH

OPENALEX_URL = "https://api.openalex.org/works"
MAILTO = None
PRL_ISSN = "0031-9007"  # Physical Review Letters (print ISSN)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_text(text):
    """Strip embedded (MathML/HTML) markup and collapse whitespace."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _reconstruct_abstract(inverted_index):
    """Rebuild plain-text abstract from OpenAlex's inverted index."""
    if not inverted_index:
        return ""
    positions = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return _clean_text(" ".join(word for _, word in positions))


def _doi_suffix(doi):
    """'https://doi.org/10.1103/rqkg-dw31' -> 'rqkg-dw31' (used as paper id)."""
    if not doi:
        return None
    return doi.rstrip("/").split("/")[-1]


def _fetch_issue_works(issn, volume, issue):
    """Page through every OpenAlex work in the given journal volume/issue."""
    results = []
    cursor = "*"
    select = ",".join([
        "id", "doi", "title", "publication_date",
        "abstract_inverted_index", "authorships",
    ])
    while cursor:
        params = {
            "filter": f"locations.source.issn:{issn},"
                      f"biblio.volume:{volume},biblio.issue:{issue}",
            "select": select,
            "per-page": 200,
            "cursor": cursor,
            "mailto": MAILTO,
        }
        resp = requests.get(OPENALEX_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        results.extend(payload["results"])
        cursor = payload["meta"].get("next_cursor")
        if not payload["results"]:
            break
    return results


def preprocess_APS_dataset(volume=135, issue=26, journal="prl", issn=PRL_ISSN, n=None):
    works = _fetch_issue_works(issn, volume, issue)

    rows = []
    for w in works:
        doi_url = w.get("doi")
        doi = doi_url.replace("https://doi.org/", "") if doi_url else None
        paper_id = _doi_suffix(doi_url)
        pub_date = w.get("publication_date")  # 'YYYY-MM-DD'
        year = int(pub_date[:4]) if pub_date else None
        authors = [a["author"]["display_name"] for a in w.get("authorships", [])]
        author_ids = [a["author"].get("id") for a in w.get("authorships", [])]
        abstract_url = (
            f"https://journals.aps.org/{journal}/abstract/{doi}" if doi else None
        )

        rows.append({
            "year": year,
            "id": paper_id,
            "title": _clean_text((w.get("title") or "")),
            "abstract": _reconstruct_abstract(w.get("abstract_inverted_index")),
            "authors": ", ".join(authors),
            "author_ids": ", ".join(x for x in author_ids if x),
            "decision": None,     # not applicable for a published APS issue
            "scores": None,       # APS has no reviewer scores
            "keywords": None,
            "labels": None,
            # extra APS-specific provenance columns:
            "publication_date": pub_date,
            "doi": doi,
            "url": abstract_url,
        })

    df = pd.DataFrame(rows)
    # Drop anything without an abstract (nothing useful downstream) and sort by
    # publication date, newest first.
    df = df[df["abstract"].str.len() > 0]
    df = df.sort_values(by="publication_date", ascending=False).reset_index(drop=True)
    if n is not None:
        df = df.head(n)

    filepath = f"{WORKSPACE_PATH}/data/aps_{journal}_{volume}_{issue}_preprocessed.parquet"
    df.to_parquet(filepath)

    print(f"{len(df)} papers -> {filepath}")
    print(df[["id", "title", "publication_date"]].head())
    return filepath


def load_papers(volume=135, issue=26, journal="prl"):
    filepath = f"{WORKSPACE_PATH}/data/aps_{journal}_{volume}_{issue}_preprocessed.parquet"
    return pd.read_parquet(filepath)


if __name__ == "__main__":
    preprocess_APS_dataset(volume=135, issue=26)
