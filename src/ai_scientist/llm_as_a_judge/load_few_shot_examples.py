import os
import pymupdf
import json
import pymupdf4llm
from pypdf import PdfReader
from horizon_bench.Config import WORKSPACE_PATH


# get directory of this file
few_shot_example_path = os.path.join(WORKSPACE_PATH, "src/ai_scientist/llm_as_a_judge/fewshot_examples")


fewshot_papers = [
    os.path.join(few_shot_example_path, "132_automated_relational.pdf"),
    os.path.join(few_shot_example_path, "attention.pdf"),
    os.path.join(few_shot_example_path, "2_carpe_diem.pdf"),
]

fewshot_reviews = [
    os.path.join(few_shot_example_path, "132_automated_relational.json"),
    os.path.join(few_shot_example_path, "attention.json"),
    os.path.join(few_shot_example_path, "2_carpe_diem.json"),
]


def get_review_fewshot_examples(num_fs_examples=1):
    fewshot_prompt = """
Below are some sample reviews, copied from previous machine learning conferences.
Note that while each review is formatted differently according to each reviewer's style, the reviews are well-structured and therefore easy to navigate.
"""
    for paper, review in zip(
        fewshot_papers[:num_fs_examples], fewshot_reviews[:num_fs_examples]
    ):
        txt_path = paper.replace(".pdf", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                paper_text = f.read()
        else:
            paper_text = load_paper(paper)
        review_text = load_review(review)
        fewshot_prompt += f"""
Paper:

```
{paper_text}
```

Review:

```
{review_text}
```
"""

    return fewshot_prompt


def load_review(path):
    with open(path, "r") as json_file:
        loaded = json.load(json_file)
    return loaded["review"]


def load_paper(pdf_path, num_pages=None, min_size=100):
    try:
        if num_pages is None:
            text = pymupdf4llm.to_markdown(pdf_path)
        else:
            reader = PdfReader(pdf_path)
            min_pages = min(len(reader.pages), num_pages)
            text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
        if len(text) < min_size:
            raise Exception("Text too short")
    except Exception as e:
        print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")
        try:
            doc = pymupdf.open(pdf_path)  # open a document
            if num_pages:
                doc = doc[:num_pages]
            text = ""
            for page in doc:  # iterate the document pages
                text = text + page.get_text()  # get plain text encoded as UTF-8
            if len(text) < min_size:
                raise Exception("Text too short")
        except Exception as e:
            print(f"Error with pymupdf, falling back to pypdf: {e}")
            reader = PdfReader(pdf_path)
            if num_pages is None:
                text = "".join(page.extract_text() for page in reader.pages)
            else:
                text = "".join(page.extract_text() for page in reader.pages[:num_pages])
            if len(text) < min_size:
                raise Exception("Text too short")
    return text



if __name__ == "__main__":
    prompt = get_review_fewshot_examples(num_fs_examples=2)
    print(prompt)