import pandas as pd
from horizon_bench.llm_api import chat
from horizon_bench import gemini_api
from tqdm import tqdm
import os
from horizon_bench import Config
from horizon_bench.Config import INPUT_DIR, TargetType, WORKSPACE_PATH, PAPERS_PARQUET, APS_PARQUET, GOAL_AND_NORMALISATION_MODEL_NAME


def generate_input_prompt(title_abstract: str, target_type: TargetType, invalidate_cache: bool = False):
    if target_type == TargetType.CHALLENGE:
        goal_ex_challenge_prompt = f"""Write a sentence that captures the research goal of the following abstract. Do not mention the challenges that this paper addresses. Do not mention the name of the paper. Begin your sentence with To...

        {title_abstract}
        """
        prompt = goal_ex_challenge_prompt
    elif target_type == TargetType.METHOD:
        goal_ex_method_prompt = f"""Write a sentence that captures the research goal of the following abstract. Do not mention the exact method that this paper uses. Do not mention the name of the paper. Begin your sentence with To...

        {title_abstract}
        """
        prompt = goal_ex_method_prompt
    else:
        raise ValueError(f"Unsupported target type: {target_type}")
    
    response = gemini_api.chat(user_prompt=prompt, invalidate_cache=invalidate_cache, model_name=GOAL_AND_NORMALISATION_MODEL_NAME)
    return response


def save_input_prompt_to_file(arxiv_id: str, input_prompt: str):
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)

    file_path = os.path.join(INPUT_DIR, f"{arxiv_id}.txt")
    with open(file_path, 'w') as f:
        f.write(input_prompt)


def check_if_input_prompt_exists(arxiv_id: str, generation_index: int) -> bool:
    file_path = os.path.join(INPUT_DIR, f"{arxiv_id}_{generation_index}.txt")
    return os.path.exists(file_path)


DATASET_PARQUETS = {
    "iclr": PAPERS_PARQUET,
    "aps": APS_PARQUET,
}


def make_input_prompts(target_type: TargetType, dataset: str = None, k_papers: int = None, paper_parquet_path: str = None, N_GENERATIONS = 1):
    # Default to the dataset selected at import time via the DATASET env var, so the
    # prompts are written into the matching dataset-specific INPUT_DIR.
    if dataset is None:
        dataset = Config.DATASET
    # INPUT_DIR is bound at import time from Config.DATASET; a mismatch would write the
    # prompts into the wrong dataset folder, so require them to agree (unless the caller
    # passes an explicit parquet path, in which case they own where papers come from).
    if paper_parquet_path is None and dataset != Config.DATASET:
        raise ValueError(
            f"dataset={dataset!r} does not match the process dataset {Config.DATASET!r}. "
            f"Select the dataset via the DATASET env var, e.g. `DATASET={dataset} python -m horizon_bench.pipeline.B_make_input_prompts`."
        )
    if paper_parquet_path is None:
        if dataset not in DATASET_PARQUETS:
            raise ValueError(f"Unknown dataset {dataset!r}, expected one of {list(DATASET_PARQUETS)}")
        paper_parquet_path = DATASET_PARQUETS[dataset]
    papers_df = pd.read_parquet(paper_parquet_path)
    if k_papers is not None:
        papers_df = papers_df.head(k_papers)
    print(papers_df.head())
    # return
    # papers_df = papers_df[[not check_if_input_prompt_exists(row["id"], 0) for _, row in papers_df.iterrows()]]

    for _, row in tqdm(papers_df.iterrows(), total=len(papers_df), desc="Generating input prompts"):
        id = row["id"]
        title_abstract = f"Title: {row['title']}\nAbstract: {row['abstract']}"
        for i in range(N_GENERATIONS):
            if check_if_input_prompt_exists(id, i):
                continue

            core_idea = generate_input_prompt(title_abstract, target_type=target_type, invalidate_cache=True)
            save_input_prompt_to_file(f"{id}_{i}", core_idea)
            

if __name__ == "__main__":
    from horizon_bench.Config import TARGET_TYPE, DATASET
    make_input_prompts(target_type=TARGET_TYPE, N_GENERATIONS=1, dataset=DATASET)