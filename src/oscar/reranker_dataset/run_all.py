import os
import json
from tqdm import tqdm

# --- Patch Config paths: redirect 'data' -> 'training_data' BEFORE importing pipeline modules ---
import horizon_bench.Config as _Config

_OLD_DATA = os.path.join(_Config.WORKSPACE_PATH, 'data')
_NEW_DATA = os.path.join(_Config.WORKSPACE_PATH, 'training_data')

_KEEP_OLD = _Config.DOWNLOADED_PAPERS_DIR  # keep downloaded papers in original data/
for _attr in dir(_Config):
    _val = getattr(_Config, _attr)
    if isinstance(_val, str) and _val.startswith(_OLD_DATA) and _val != _OLD_DATA:
        if _val.startswith(_KEEP_OLD):
            continue
        setattr(_Config, _attr, _val.replace(_OLD_DATA, _NEW_DATA, 1))
# Patch DATA_ROOT itself (exact match)
if _Config.DATA_ROOT == _OLD_DATA:
    _Config.DATA_ROOT = _NEW_DATA

# Recreate directories for the new paths
_Config.make_dirs([
    _Config.INPUT_DIR, _Config.EXAMPLE_CODE_DIR,
    _Config.OUTPUT_DIR, _Config.OSCAR_V5_OUTPUT_DIR,
    _Config.LOGS_DIR, _Config.OSCAR_V5_LOGS_DIR,
    _Config.GROUND_TRUTH_DIR, _Config.PREDICTIONS_DIR,
    _Config.OSCAR_V5_PREDICTIONS_DIR,
    os.path.join(_Config.TASK_TYPE_ROOT, 'oscar_insights'),
])

WORKSPACE_PATH = _Config.WORKSPACE_PATH
TASK_TYPE_ROOT = _Config.TASK_TYPE_ROOT

# --- Now import pipeline modules (they pick up patched Config values via `from Config import ...`) ---
from horizon_bench.Config import TargetType
from horizon_bench.pipeline.A_preprocess_iclr_dataset import preprocess_iclr_dataset
from horizon_bench.pipeline.B_make_input_prompts import make_input_prompts
from horizon_bench.pipeline.C_run_baselines import run_baselines
from horizon_bench.pipeline.D_extract_targets import extract_prediction, get_gt
from horizon_bench.evaluation.cosine_distance import get_cosine_similarity


def add_extracted_predictions_to_insights():
    # Extract predicted method for each oscar_insights JSON file
    oscar_insights_dir = os.path.join(TASK_TYPE_ROOT, 'oscar_insights')
    for filename in tqdm(sorted(os.listdir(oscar_insights_dir)), desc='Extracting methods'):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(oscar_insights_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        if 'predicted_method' in data:
            continue
        prediction = extract_prediction(data['predicted_paper'], target_type=TargetType.METHOD)
        data['predicted_method'] = prediction.method
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)


def add_scores_to_insights():
    # Compute cosine similarity between predicted_method and ground truth for each file
    oscar_insights_dir = os.path.join(TASK_TYPE_ROOT, 'oscar_insights')
    for filename in tqdm(sorted(os.listdir(oscar_insights_dir)), desc='Scoring'):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(oscar_insights_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        if 'score' in data:
            continue
        goal_id = data['goal_id']
        gt = get_gt(goal_id, target_type=TargetType.METHOD)
        score = get_cosine_similarity(data['predicted_method'], gt)
        data['score'] = float(score)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)


def collate_insights_to_csv():
    import pandas as pd
    oscar_insights_dir = os.path.join(TASK_TYPE_ROOT, 'oscar_insights')
    records = []
    for filename in sorted(os.listdir(oscar_insights_dir)):
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(oscar_insights_dir, filename), 'r') as f:
            data = json.load(f)
        data['filename'] = filename
        records.append(data)
    df = pd.DataFrame(records)
    csv_path = os.path.join(TASK_TYPE_ROOT, f'reranker_train_{len(df)}.parquet')
    df.to_parquet(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}")
    print(df.head())
    return df


if __name__=="__main__":
    START_IDX = 100
    END_IDX = 300
    iclr_dataset_path = preprocess_iclr_dataset(best_papers=True, start_idx=100, end_idx=300)
    # make_input_prompts(target_type=TargetType.METHOD, paper_parquet_path=iclr_dataset_path, N_GENERATIONS=1)
    # run_baselines(target_type=TargetType.METHOD, MODELS_TO_RUN = ['oscar_v5'])
    # add_extracted_predictions_to_insights()
    add_scores_to_insights()
    collate_insights_to_csv()
