def format_papers(papers: list[dict]) -> str:
    """Format Semantic Scholar paper results into a readable string."""
    paper_strings = []
    for i, paper in enumerate(papers):
        authors = paper.get("authors", [])
        if isinstance(authors, list) and authors and isinstance(authors[0], dict):
            authors_str = ", ".join(a.get("name", "Unknown") for a in authors)
        elif isinstance(authors, list):
            authors_str = ", ".join(str(a) for a in authors)
        else:
            authors_str = str(authors)

        paper_strings.append(
            f"{i + 1}: {paper.get('title', 'Unknown Title')}. {authors_str}. "
            f"{paper.get('venue', 'Unknown Venue')}, {paper.get('year', 'Unknown Year')}.\n"
            f"Number of citations: {paper.get('citationCount', 'N/A')}\n"
            f"Abstract: {paper.get('abstract', 'No abstract available.')}"
        )
    return "\n\n".join(paper_strings)


def idea_json_to_markdown(idea: dict) -> str:
    """Convert an idea JSON dict to a formatted markdown string."""
    sections = [
        ("Name", idea.get("Name", "")),
        ("Title", idea.get("Title", "")),
        ("Short Hypothesis", idea.get("Short Hypothesis", "")),
        ("Related Work", idea.get("Related Work", "")),
        ("Abstract", idea.get("Abstract", "")),
        ("Experiments", idea.get("Experiments", "")),
        ("Risk Factors and Limitations", idea.get("Risk Factors and Limitations", "")),
    ]
    parts = []
    for heading, content in sections:
        if content:
            parts.append(f"## {heading}\n\n{content}")
    return "\n\n".join(parts)
