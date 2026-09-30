# binary decision if two ideas are the same or not

import os
import csv
import json
from pydantic import BaseModel
from horizon_bench.llm_api import get_formatted_chat_response
from horizon_bench.prompts.ideas_same_prompt import ideas_same_prompt
from horizon_bench.Config import OUTPUT_DIR, TARGET_TYPE, DATA_ROOT
from horizon_bench.pipeline.D_extract_targets import get_gt
from horizon_bench.evaluation.load_idea import load_idea
from horizon_bench.evaluation.cosine_distance import get_cosine_similarity


class IdeaComparisonResult(BaseModel):
    are_same: bool
    similarity: float

def check_ideas_are_same(idea_filepath: str, verbose=False) -> IdeaComparisonResult:
    idea_filename = idea_filepath.split(os.sep)[-1].replace(".txt", "")
    idea_a = load_idea(idea_filepath)

    arxiv_id = idea_filename.split('_')[0]
    idea_b = get_gt(arxiv_id, target_type=TARGET_TYPE)

    user_prompt = ideas_same_prompt.format(idea_a=idea_a, idea_b=idea_b)
    response = get_formatted_chat_response(
        user_prompt=user_prompt,
        system_prompt="You are an expert evaluator of research ideas.",
        response_format=IdeaComparisonResult,
    )
    if verbose:
        print(idea_a)
        print(idea_b)
    save(idea_filepath, idea_filename, response)
    return response

def save(idea_filepath: str, idea_filename: str, idea_comparison_result: IdeaComparisonResult):
    output_dir = os.path.join(*os.path.split(idea_filepath)[:-1], 'idea_comparison')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{idea_filename}.json")
    with open(output_path, 'w') as f:
        json.dump(idea_comparison_result.dict(), f, indent=4)
    return idea_comparison_result


# Folder name -> human-readable model type label
MODEL_TYPES = {
    "naive_baseline": "naive baseline",
    "ai_scientist_v2": "ai scientist v2",
    "research_agent": "research agent",
    "chain_of_ideas": "chain of ideas",
    "ai_scientist": "ai scientist",
}


def _get_gt_for_task(arxiv_id: str, task_type: str) -> str:
    """Read ground truth for a given task type, independent of the module-level
    TARGET_TYPE (so both 'challenge' and 'method' can be evaluated in one run)."""
    gt_path = os.path.join(DATA_ROOT, "ground_truth", task_type, f"{arxiv_id}.txt")
    with open(gt_path, "r") as f:
        return f.read()


def _llm_similarity(idea_a: str, idea_b: str) -> IdeaComparisonResult:
    user_prompt = ideas_same_prompt.format(idea_a=idea_a, idea_b=idea_b)
    return get_formatted_chat_response(
        user_prompt=user_prompt,
        system_prompt="You are an expert evaluator of research ideas.",
        response_format=IdeaComparisonResult,
    )


def evaluate_predictions(
    base_dir: str = os.path.join(DATA_ROOT, "ex_7_gemini_flash_preview_oscar"),
    output_csv: str = os.path.join(DATA_ROOT, "ex_7_gemini_flash_preview_oscar", "similarity_results.csv"),
    task_types=("challenge", "method"),
    model_dirs=tuple(MODEL_TYPES.keys()),
    flush_every: int = 10,
):
    """Compute LLM-based similarity and cosine similarity between each model
    prediction and its ground truth, writing results to a CSV.

    The CSV records the task type (challenge/method) and model type for each row.
    Results are buffered and appended to the CSV every `flush_every` rows.
    """
    fieldnames = [
        "task_type",
        "model_type",
        "model_dir",
        "arxiv_id",
        "prediction_file",
        "llm_are_same",
        "llm_similarity",
        "cosine_similarity",
    ]

    # Write header once (only if the file does not already exist).
    if not os.path.exists(output_csv):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def flush(rows):
        if not rows:
            return
        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(rows)
        print(f"Appended {len(rows)} rows to {output_csv}")

    buffer = []
    for task_type in task_types:
        for model_dir in model_dirs:
            preds_dir = os.path.join(base_dir, task_type, "predictions", model_dir)
            if not os.path.isdir(preds_dir):
                print(f"Skipping missing folder: {preds_dir}")
                continue

            for fname in sorted(os.listdir(preds_dir)):
                if not fname.endswith(".txt"):
                    continue
                idea_filepath = os.path.join(preds_dir, fname)
                idea_filename = fname.replace(".txt", "")
                arxiv_id = idea_filename.split("_")[0]

                try:
                    idea_a = load_idea(idea_filepath)
                    idea_b = _get_gt_for_task(arxiv_id, task_type)
                    llm = _llm_similarity(idea_a, idea_b)
                    cosine = get_cosine_similarity(idea_a, idea_b)
                except Exception as e:
                    print(f"Error on {idea_filepath}: {e}")
                    continue

                buffer.append({
                    "task_type": task_type,
                    "model_type": MODEL_TYPES.get(model_dir, model_dir),
                    "model_dir": model_dir,
                    "arxiv_id": arxiv_id,
                    "prediction_file": fname,
                    "llm_are_same": llm.are_same,
                    "llm_similarity": llm.similarity,
                    "cosine_similarity": cosine,
                })
                print(f"{task_type}/{model_dir}/{fname}: "
                      f"same={llm.are_same} llm={llm.similarity:.3f} cosine={cosine:.3f}")

                if len(buffer) >= flush_every:
                    flush(buffer)
                    buffer = []

    flush(buffer)
    print(f"Done. Results written to {output_csv}")



def plot_correlation(
    input_csv: str = os.path.join(DATA_ROOT, "ex_7_gemini_flash_preview_oscar", "similarity_results.csv"),
    output_plot: str = os.path.join(DATA_ROOT, "ex_7_gemini_flash_preview_oscar", "similarity_correlation.pdf"),
    output_table: str = os.path.join(DATA_ROOT, "ex_7_gemini_flash_preview_oscar", "similarity_correlation.csv"),
):
    """Plot the correlation between LLM similarity and cosine similarity.

    Aggregates over all models, but reports/plots each task type separately.
    Produces:
      - a scatter plot (one panel per task type) with a regression line
      - a correlation table (Pearson & Spearman + n) saved to CSV
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    df = pd.read_csv(input_csv)
    task_types = sorted(df["task_type"].unique())

    # --- correlation table (aggregated over all models, per task type) ---
    rows = []
    for task_type in task_types:
        sub = df[df["task_type"] == task_type].dropna(
            subset=["llm_similarity", "cosine_similarity"])
        x = sub["cosine_similarity"]
        y = sub["llm_similarity"]
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        rows.append({
            "task_type": task_type,
            "n": len(sub),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        })
    table = pd.DataFrame(rows)
    table.to_csv(output_table, index=False)
    print(table.to_string(index=False))
    print(f"Saved correlation table to {output_table}")

    # --- scatter plot, one panel per task type ---
    n = len(task_types)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, task_type in zip(axes[0], task_types):
        sub = df[df["task_type"] == task_type].dropna(
            subset=["llm_similarity", "cosine_similarity"])
        x = sub["cosine_similarity"]
        y = sub["llm_similarity"]
        ax.scatter(x, y, alpha=0.5, edgecolors="none")

        # regression line
        if len(sub) >= 2:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, m * xs + b, color="red", linewidth=1.5)

        r = table.loc[table["task_type"] == task_type, "pearson_r"].iloc[0]
        rho = table.loc[table["task_type"] == task_type, "spearman_r"].iloc[0]
        ax.set_title(f"{task_type} (n={len(sub)})\nPearson r={r:.3f}, Spearman ρ={rho:.3f}")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("LLM similarity")

    fig.tight_layout()
    fig.savefig(output_plot)
    print(f"Saved correlation plot to {output_plot}")
    return table


if __name__ == "__main__":
    evaluate_predictions(flush_every=10)
    plot_correlation()

    # --- original single-file example (kept for reference) ---
    # path = 'data/ex_7_gemini_flash_preview_oscar/challenge/predictions/ai_scientist/1PIfB5w05x_0.txt'
    # result = check_ideas_are_same(path, verbose=True)
    # idea_filename = path.split(os.sep)[-1].replace(".txt", "")
    # idea_a = load_idea(path)
    # arxiv_id = idea_filename.split('_')[0]
    # idea_b = get_gt(arxiv_id, target_type=TARGET_TYPE)
    # print(f'Cosine Similarity (direct): {get_cosine_similarity(idea_a, idea_b)}')
    # print(result)
