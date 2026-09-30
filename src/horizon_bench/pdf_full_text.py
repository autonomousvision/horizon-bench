"""Shared helper to extract full text from a PDF via scipdf + GROBID.

Requires a GROBID server running at localhost:8070 (see project README).
Used by both the OpenReview (ICLR) and APS full-text fetchers so the parsing
logic lives in one place.
"""
import scipdf


def extract_pdf_full_text(pdf_path: str) -> str:
    """Parse a PDF into a single full-text string (title + abstract + body)."""
    parsed = scipdf.parse_pdf(pdf_path, fulltext=True, soup=True)

    sections = []
    if parsed.find('title'):
        sections.append(parsed.find('title').get_text())
    if parsed.find('abstract'):
        sections.append(parsed.find('abstract').get_text())
    for div in parsed.find_all('div'):
        text = div.get_text(separator=' ', strip=True)
        if text:
            sections.append(text)

    return '\n\n'.join(sections)
