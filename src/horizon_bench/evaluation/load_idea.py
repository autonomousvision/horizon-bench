import os

def load_idea(filepath: str) -> str:
    """
    Load the research idea from a text file.
    filepath: Path to the text file containing the research idea.
    """
    with open(filepath, 'r') as f:
        idea_text = f.read()
    return idea_text