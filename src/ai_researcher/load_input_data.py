from pydantic import BaseModel, Field
import json
from typing import List, Dict
import re
from datetime import date


class EvalMetadata(BaseModel):
    source_papers: list = Field(description="the list of source papers")
    task_instructions: str = Field(description="the task instructions")
    date: str = Field(description="the date", pattern=r"^\d{4}-\d{2}-\d{2}$")  # YYYY-MM-DD format
    date_limit: str = Field(description="the date limit", pattern=r"^\d{4}-\d{2}-\d{2}$")  # YYYY-MM-DD format


def load_input_data(instance_path: str) -> Dict[str, str]:
    with open(instance_path, "r", encoding="utf-8") as f:
        eval_instance = json.load(f)
    source_papers = eval_instance["source_papers"]  
    task_instructions = eval_instance['task1']   
    arxiv_url = eval_instance["url"]

    date = arxiv_id_to_date(arxiv_url.split('/')[-1]).isoformat()

    # model_dump converts the Pydantic model to a dictionary
    # Interesting that date and date_limit are the same
    res = EvalMetadata(source_papers=source_papers, task_instructions=task_instructions, date=date, date_limit=date).model_dump()
    return res

def arxiv_id_to_date(arxiv_id: str) -> date:
    """
    Convert an arXiv ID to a publication date (year-month-day as datetime.date).
    The day is assumed to be the first day of the month, since arXiv IDs don't encode the exact day.
    """
    
    # New format: yymm.number (from 2007 onwards)
    new_format = re.match(r"^(\d{2})(\d{2})\.\d{4,5}(v\d+)?$", arxiv_id)
    if new_format:
        yy, mm = new_format.groups()[:2]
        year = 2000 + int(yy)
        month = int(mm)
        return date(year, month, 1)
    
    # Old format: archive/YYMMNNN or archive/YYMMNNNN
    old_format = re.match(r"^[a-z\-]+/(\d{2})(\d{2})\d{3,4}(v\d+)?$", arxiv_id)
    if old_format:
        yy, mm = old_format.groups()[:2]
        yy = int(yy)
        year = 1900 + yy if yy >= 91 else 2000 + yy
        month = int(mm)
        return date(year, month, 1)
    raise ValueError(f"Invalid arXiv ID format: {arxiv_id}")

def get_references(task_metadata: Dict[str, List[Dict[str, str]]]) -> List[str]:
    references: List[str] = []
    for paper in task_metadata["source_papers"]:
        title = paper.get("reference")
        if title:
            references.append(title.strip())
    return references