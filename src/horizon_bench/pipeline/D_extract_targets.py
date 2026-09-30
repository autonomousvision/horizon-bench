import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from horizon_bench.Config import AI_RESEARCHER_OUTPUT_DIR, AI_SCIENTIST_OUTPUT_DIR, CHAIN_OF_IDEAS_OUTPUT_DIR, NAIVE_BASELINE_OUTPUT_DIR, AUTODS_OUTPUT_DIR, RESEARCH_AGENT_OUTPUT_DIR, AI_SCIENTIST_V2_OUTPUT_DIR, OSCAR_OUTPUT_DIR, OSCAR_V2_OUTPUT_DIR, OSCAR_V3_OUTPUT_DIR, OSCAR_V4_OUTPUT_DIR, OSCAR_V5_OUTPUT_DIR, OSCAR_V6_OUTPUT_DIR, TargetType, GROUND_TRUTH_DIR, AI_RESEARCHER_PREDICTIONS_DIR, AI_SCIENTIST_PREDICTIONS_DIR, CHAIN_OF_IDEAS_PREDICTIONS_DIR, NAIVE_BASELINE_PREDICTIONS_DIR, AUTODS_PREDICTIONS_DIR, RESEARCH_AGENT_PREDICTIONS_DIR, AI_SCIENTIST_V2_PREDICTIONS_DIR, OSCAR_PREDICTIONS_DIR, OSCAR_V2_PREDICTIONS_DIR, OSCAR_V3_PREDICTIONS_DIR, OSCAR_V4_PREDICTIONS_DIR, OSCAR_V5_PREDICTIONS_DIR, OSCAR_V6_PREDICTIONS_DIR, GOAL_AND_NORMALISATION_MODEL_NAME
from horizon_bench.llm_api import get_formatted_chat_response
from pydantic import BaseModel
from tqdm import tqdm
from horizon_bench.arxiv_get_full_text import ArxivFullTextFetchError
from horizon_bench.Config import INPUT_DIR
from horizon_bench.openreview_get_full_text import openreview_get_full_text
from horizon_bench.aps_get_full_text import aps_get_full_text
from horizon_bench import Config
from horizon_bench import gemini_api
from google.genai import types as genai_types


def get_full_text(paper_id: str) -> str:
    """Fetch a paper's full text using the fetcher appropriate for the dataset.

    For the APS dataset the paper's PDF is downloaded from its URL; otherwise the
    OpenReview PDF is fetched by paper id.
    """
    if Config.DATASET == "aps":
        return aps_get_full_text(paper_id)
    return openreview_get_full_text(paper_id)

class Prediction(BaseModel):
    pass

class Challenge(Prediction):
    challenge: str

class Method(Prediction):
    method: str

class ResearchQuestion(Prediction):
    research_question: str

class Predictions(BaseModel):
    # Each prediction in this list is from a separate inference run of the respective agent.
    ai_researcher: list[Prediction] = []
    ai_scientist: list[Prediction] = []
    coi_agent: list[Prediction] = []
    naive_baseline: list[Prediction] = []
    autods: list[Prediction] = []
    research_agent: list[Prediction] = []
    ai_scientist_v2: list[Prediction] = []
    oscar: list[Prediction] = []
    oscar_v2: list[Prediction] = []
    oscar_v3: list[Prediction] = []
    oscar_v4: list[Prediction] = []
    oscar_v5: list[Prediction] = []
    oscar_v6: list[Prediction] = []


def _to_internal_key(agent_name: str) -> str:
    """Map run_all agent names to internal keys (only 'chain_of_ideas' differs)."""
    return 'coi_agent' if agent_name == 'chain_of_ideas' else agent_name


def _get_active_output_dirs() -> list:
    """Return output dirs filtered by AGENTS_TO_RUN env var, or all if unset."""
    agents_env = os.environ.get("AGENTS_TO_RUN")
    if not agents_env:
        return list(OUTPUT_TO_PREDICTIONS_DIR.keys())
    active_keys = {_to_internal_key(a) for a in agents_env.split(",")}
    return [d for d, (_, key) in OUTPUT_TO_PREDICTIONS_DIR.items() if key in active_keys]


OUTPUT_TO_PREDICTIONS_DIR = {
    AI_RESEARCHER_OUTPUT_DIR: (AI_RESEARCHER_PREDICTIONS_DIR, 'ai_researcher'),
    AI_SCIENTIST_OUTPUT_DIR: (AI_SCIENTIST_PREDICTIONS_DIR, 'ai_scientist'),
    CHAIN_OF_IDEAS_OUTPUT_DIR: (CHAIN_OF_IDEAS_PREDICTIONS_DIR, 'coi_agent'),
    NAIVE_BASELINE_OUTPUT_DIR: (NAIVE_BASELINE_PREDICTIONS_DIR, 'naive_baseline'),
    AUTODS_OUTPUT_DIR: (AUTODS_PREDICTIONS_DIR, 'autods'),
    RESEARCH_AGENT_OUTPUT_DIR: (RESEARCH_AGENT_PREDICTIONS_DIR, 'research_agent'),
    AI_SCIENTIST_V2_OUTPUT_DIR: (AI_SCIENTIST_V2_PREDICTIONS_DIR, 'ai_scientist_v2'),
    OSCAR_OUTPUT_DIR: (OSCAR_PREDICTIONS_DIR, 'oscar'),
    OSCAR_V2_OUTPUT_DIR: (OSCAR_V2_PREDICTIONS_DIR, 'oscar_v2'),
    OSCAR_V3_OUTPUT_DIR: (OSCAR_V3_PREDICTIONS_DIR, 'oscar_v3'),
    OSCAR_V4_OUTPUT_DIR: (OSCAR_V4_PREDICTIONS_DIR, 'oscar_v4'),
    OSCAR_V5_OUTPUT_DIR: (OSCAR_V5_PREDICTIONS_DIR, 'oscar_v5'),
    OSCAR_V6_OUTPUT_DIR: (OSCAR_V6_PREDICTIONS_DIR, 'oscar_v6'),
}


def prediction_to_text(prediction: Prediction) -> str:
    if isinstance(prediction, Challenge):
        return prediction.challenge
    elif isinstance(prediction, Method):
        return prediction.method
    elif isinstance(prediction, ResearchQuestion):
        return prediction.research_question


def save_predictions(predictions: list[Prediction], predictions_dir: str, arxiv_id: str):
    for i, prediction in enumerate(predictions):
        filepath = os.path.join(predictions_dir, f"{arxiv_id}_{i}.txt")
        with open(filepath, "w") as f:
            f.write(prediction_to_text(prediction))


def load_existing_predictions(preds_dir: str, arxiv_id: str, target_type: TargetType) -> list[Prediction] | None:
    """Load already-extracted predictions from disk. Returns None if none exist."""
    files = [f for f in os.listdir(preds_dir) if f.startswith(arxiv_id + "_") and f.endswith(".txt")]
    if not files:
        return None
    files.sort()
    predictions = []
    for filename in files:
        with open(os.path.join(preds_dir, filename), "r") as f:
            text = f.read()
        if target_type == TargetType.CHALLENGE:
            predictions.append(Challenge(challenge=text))
        elif target_type == TargetType.METHOD:
            predictions.append(Method(method=text))
        elif target_type == TargetType.RESEARCH_QUESTION:
            predictions.append(ResearchQuestion(research_question=text))
    return predictions


def get_predictions(arxiv_id: str, target_type: TargetType):
    prediction_dirs = _get_active_output_dirs()
    res = dict()
    for prediction_dir in prediction_dirs:
        preds_dir, key = OUTPUT_TO_PREDICTIONS_DIR[prediction_dir]
        predicted_ideas = load_predicted_ideas(prediction_dir, arxiv_id)
        existing = load_existing_predictions(preds_dir, arxiv_id, target_type)
        if existing is not None and len(existing) >= len(predicted_ideas):
            res[key] = existing
            continue

        predictions = [extract_prediction(idea, target_type=target_type) for idea in predicted_ideas]

        save_predictions(predictions, preds_dir, arxiv_id)
        res[key] = predictions
    return Predictions(**res)

def load_predicted_ideas(prediction_dir: str, arxiv_id: str) -> list[str]:
    import os
    predicted_ideas = []
    for filename in os.listdir(prediction_dir):
        if filename.startswith(arxiv_id + "_") and filename.endswith(".txt"):
            filepath = os.path.join(prediction_dir, filename)
            with open(filepath, "r") as f:
                idea = f.read()
                predicted_ideas.append(idea)
    return predicted_ideas


def extract_prediction(predicted_idea: str, target_type: TargetType) -> Prediction:
    if target_type == TargetType.CHALLENGE:
        prompt = f"What is the main challenge outlined in the following idea? Answer one sentence, only.\n\n{predicted_idea}"
        response = gemini_api.get_formatted_chat_response(
            response_format=Challenge,
            user_prompt=prompt,
            model_name=GOAL_AND_NORMALISATION_MODEL_NAME,
        )
    elif target_type == TargetType.METHOD:
        prompt = f"What is the main method outlined in the following idea? Answer one sentence, only.\n\n{predicted_idea}"
        response = gemini_api.get_formatted_chat_response(
            response_format=Method,
            user_prompt=prompt,
            model_name=GOAL_AND_NORMALISATION_MODEL_NAME,
        )
    elif target_type == TargetType.RESEARCH_QUESTION:
        prompt = f"What is the main research question addressed in the following idea? Answer one sentence, only.\n\n{predicted_idea}"
        response = gemini_api.get_formatted_chat_response(
            response_format=ResearchQuestion,
            user_prompt=prompt,
            model_name=GOAL_AND_NORMALISATION_MODEL_NAME,
        )
    return response


# ---------------------------------------------------
# Shared batch helpers
# ---------------------------------------------------

def _get_response_format(target_type: TargetType) -> type[Prediction]:
    return {
        TargetType.CHALLENGE: Challenge,
        TargetType.METHOD: Method,
        TargetType.RESEARCH_QUESTION: ResearchQuestion,
    }[target_type]


def _build_extraction_prompt(text: str, target_type: TargetType) -> str:
    if target_type == TargetType.CHALLENGE:
        return f"What is the main challenge outlined in the following idea? Answer one sentence, only.\n\n{text}"
    elif target_type == TargetType.METHOD:
        return f"What is the main method outlined in the following idea? Answer one sentence, only.\n\n{text}"
    elif target_type == TargetType.RESEARCH_QUESTION:
        return f"What is the main research question addressed in the following idea? Answer one sentence, only.\n\n{text}"


BATCH_POLL_INTERVAL_SECONDS = 30
BATCH_MAX_WAIT_SECONDS = 3600


def _submit_and_poll_batch(requests: list[genai_types.InlinedRequest]) -> genai_types.BatchJob:
    batch_job = gemini_api.client.batches.create(
        model=GOAL_AND_NORMALISATION_MODEL_NAME,
        src=requests,
    )
    print(f"Batch job submitted: {batch_job.name} ({len(requests)} requests)")

    elapsed = 0
    while True:
        batch_job = gemini_api.client.batches.get(name=batch_job.name)
        state = batch_job.state
        print(f"  Batch state: {state} (elapsed {elapsed}s)")

        if state == genai_types.JobState.JOB_STATE_SUCCEEDED:
            return batch_job
        if state == genai_types.JobState.JOB_STATE_PARTIALLY_SUCCEEDED:
            print(f"  WARNING: Batch partially succeeded. Stats: {batch_job.completion_stats}")
            return batch_job
        if state in (
            genai_types.JobState.JOB_STATE_FAILED,
            genai_types.JobState.JOB_STATE_CANCELLED,
            genai_types.JobState.JOB_STATE_EXPIRED,
        ):
            raise RuntimeError(f"Batch job {batch_job.name} ended with state {state}: {batch_job.error}")

        if elapsed >= BATCH_MAX_WAIT_SECONDS:
            raise TimeoutError(f"Batch job {batch_job.name} did not complete within {BATCH_MAX_WAIT_SECONDS}s")

        time.sleep(BATCH_POLL_INTERVAL_SECONDS)
        elapsed += BATCH_POLL_INTERVAL_SECONDS


# ---------------------------------------------------

def get_gt(arxiv_id: str, target_type: TargetType) -> str:
    gt_path = os.path.join(GROUND_TRUTH_DIR, f"{arxiv_id}.txt")
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            return f.read()

    full_paper_text = get_full_text(arxiv_id)
    gt = extract_prediction(full_paper_text, target_type)
    if target_type == TargetType.CHALLENGE:
        gt_text = gt.challenge
    elif target_type == TargetType.METHOD:
        gt_text = gt.method
    elif target_type == TargetType.RESEARCH_QUESTION:
        gt_text = gt.research_question

    with open(gt_path, "w") as f:
        f.write(gt_text)
    return gt_text

def get_input_ids() -> list[str]:
    import os
    arxiv_ids = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            arxiv_id = filename.split('_')[0]
            arxiv_ids.append(arxiv_id)
    return list(set(arxiv_ids))


def extract_predictions(target_type: TargetType, k_papers: int = None):
    arxiv_ids = get_input_ids()[::-1]
    if k_papers is not None:
        arxiv_ids = arxiv_ids[:k_papers]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_predictions, arxiv_id, target_type): arxiv_id for arxiv_id in arxiv_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Extracting Predictions'):
            future.result()


def make_ground_truth(target_type: TargetType, k_papers: int = None):
    arxiv_ids = get_input_ids()
    if k_papers is not None:
        arxiv_ids = arxiv_ids[:k_papers]
    for arxiv_id in tqdm(arxiv_ids, 'Ground Truth Predictions'):
        try:
            get_gt(arxiv_id, target_type=target_type)
        except ArxivFullTextFetchError as e:
            print(f"Skipping arXiv ID {arxiv_id} due to error: {e}")


# ---------------------------------------------------
# Batch ground truth extraction
# ---------------------------------------------------

def _download_all_texts(arxiv_ids: list[str]) -> dict[str, str]:
    texts = {}
    for arxiv_id in tqdm(arxiv_ids, desc='Downloading paper texts'):
        gt_path = os.path.join(GROUND_TRUTH_DIR, f"{arxiv_id}.txt")
        if os.path.exists(gt_path):
            continue
        try:
            texts[arxiv_id] = get_full_text(arxiv_id)
        except Exception as e:
            print(f"Skipping {arxiv_id} due to download error: {e}")
    return texts


def make_ground_truth_batch(target_type: TargetType, k_papers: int = None):
    arxiv_ids = get_input_ids()
    if k_papers is not None:
        arxiv_ids = arxiv_ids[:k_papers]

    # Phase 1: Download all texts
    texts = _download_all_texts(arxiv_ids)

    if not texts:
        print("All ground truths already exist on disk. Nothing to do.")
        return

    # Fall back to non-batch extraction for small runs
    if len(texts) < 10000:
        print(f"Only {len(texts)} papers — using non-batch extraction.")
        for arxiv_id, full_text in tqdm(texts.items(), desc='Extracting GT'):
            gt = extract_prediction(full_text, target_type)
            gt_text = prediction_to_text(gt)
            gt_path = os.path.join(GROUND_TRUTH_DIR, f"{arxiv_id}.txt")
            with open(gt_path, "w") as f:
                f.write(gt_text)
        return

    # Phase 2: Batch extract predictions
    response_format = _get_response_format(target_type)
    requests = []
    for arxiv_id, full_text in texts.items():
        prompt = _build_extraction_prompt(full_text, target_type)
        req = genai_types.InlinedRequest(
            model=GOAL_AND_NORMALISATION_MODEL_NAME,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_format,
            ),
            metadata={"arxiv_id": arxiv_id},
        )
        requests.append(req)

    print(f"Submitting {len(requests)} GT papers for batch extraction...")
    batch_job = _submit_and_poll_batch(requests)

    # Parse responses and save
    success_count = 0
    for resp in batch_job.dest.inlined_responses:
        arxiv_id = resp.metadata.get("arxiv_id")
        if not arxiv_id:
            print("  WARNING: Response missing arxiv_id in metadata, skipping.")
            continue
        if resp.error:
            print(f"  ERROR for {arxiv_id}: {resp.error}")
            continue
        try:
            parsed = response_format.model_validate_json(resp.response.text)
            gt_text = prediction_to_text(parsed)
            gt_path = os.path.join(GROUND_TRUTH_DIR, f"{arxiv_id}.txt")
            with open(gt_path, "w") as f:
                f.write(gt_text)
            success_count += 1
        except Exception as e:
            print(f"  ERROR parsing response for {arxiv_id}: {e}")

    print(f"Batch GT extraction complete: {success_count}/{len(requests)} succeeded.")


# ---------------------------------------------------
# Batch agent prediction extraction
# ---------------------------------------------------

def extract_predictions_batch(target_type: TargetType, k_papers: int = None):
    arxiv_ids = get_input_ids()[::-1]
    if k_papers is not None:
        arxiv_ids = arxiv_ids[:k_papers]

    response_format = _get_response_format(target_type)

    # Phase 1: Collect all work items
    work_items = []  # (arxiv_id, prediction_dir, idea_index, idea_text)

    prediction_dirs = _get_active_output_dirs()

    for arxiv_id in tqdm(arxiv_ids, desc='Collecting predictions'):
        for prediction_dir in prediction_dirs:
            predicted_ideas = load_predicted_ideas(prediction_dir, arxiv_id)
            for i, idea in enumerate(predicted_ideas):
                work_items.append((arxiv_id, prediction_dir, i, idea))

    # Fall back to non-batch extraction for small runs
    if len(work_items) < 10000:
        print(f"Only {len(work_items)} work items — using non-batch extraction.")
        extract_predictions(target_type, k_papers)
        return

    # Phase 2: Batch extract
    if work_items:
        requests = []
        for arxiv_id, prediction_dir, i, idea in work_items:
            prompt = _build_extraction_prompt(idea, target_type)
            req = genai_types.InlinedRequest(
                model=GOAL_AND_NORMALISATION_MODEL_NAME,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_format,
                ),
                metadata={
                    "arxiv_id": arxiv_id,
                    "prediction_dir": prediction_dir,
                    "idea_index": str(i),
                },
            )
            requests.append(req)

        print(f"Submitting {len(requests)} agent predictions for batch extraction...")
        batch_job = _submit_and_poll_batch(requests)

        # Group parsed results by (arxiv_id, prediction_dir)
        grouped = {}  # (arxiv_id, prediction_dir) -> dict[int, Prediction]
        for resp in batch_job.dest.inlined_responses:
            arxiv_id = resp.metadata.get("arxiv_id")
            pred_dir = resp.metadata.get("prediction_dir")
            idx = int(resp.metadata.get("idea_index", -1))
            if resp.error:
                print(f"  ERROR for {arxiv_id} dir={pred_dir} idx={idx}: {resp.error}")
                continue
            try:
                parsed = response_format.model_validate_json(resp.response.text)
                grouped.setdefault((arxiv_id, pred_dir), {})[idx] = parsed
            except Exception as e:
                print(f"  ERROR parsing {arxiv_id} dir={pred_dir} idx={idx}: {e}")

        # Save predictions in order
        for (arxiv_id, pred_dir), idx_map in grouped.items():
            preds_dir, _ = OUTPUT_TO_PREDICTIONS_DIR[pred_dir]
            ordered = [idx_map[i] for i in sorted(idx_map.keys())]
            save_predictions(ordered, preds_dir, arxiv_id)

    print("Batch prediction extraction complete.")


def _resolve_dataset(dataset: str = None) -> str:
    """Validate the dataset selection against the one bound at import time.

    The input/ground-truth/prediction folders are bound at import time from
    Config.DATASET (INPUT_DIR, GROUND_TRUTH_DIR, the *_PREDICTIONS_DIRs above all
    `from Config import <DIR>`). A dataset passed here that differs from that would
    read/write the wrong folders, so the dataset MUST be selected via the DATASET
    env var and any explicit value must match it.
    """
    from horizon_bench import Config
    if dataset is None:
        dataset = Config.DATASET
    if dataset != Config.DATASET:
        raise ValueError(
            f"dataset={dataset!r} does not match the process dataset {Config.DATASET!r}. "
            f"Select the dataset via the DATASET env var, e.g. "
            f"`DATASET={dataset} python -m horizon_bench.pipeline.D_extract_targets`."
        )
    if dataset not in Config.DATASET_PARQUETS:
        raise ValueError(f"Unknown dataset {dataset!r}, expected one of {list(Config.DATASET_PARQUETS)}")
    return dataset


if __name__ == "__main__":
    from horizon_bench.Config import TARGET_TYPE, DATASET
    dataset = _resolve_dataset(DATASET)
    print(f"Extracting targets for dataset={dataset!r}, target_type={TARGET_TYPE.value!r}")
    extract_predictions_batch(TARGET_TYPE)
    make_ground_truth_batch(TARGET_TYPE)