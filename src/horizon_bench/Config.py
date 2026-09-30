import os
from pathlib import Path
from enum import Enum

class LLMAPIType(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    VLLM = "vllm"

_llmapi_env = os.getenv("LLMAPINAME", LLMAPIType.GEMINI.value).lower()
LLMAPINAME = LLMAPIType(_llmapi_env)
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "gemini-3-flash-preview")


GOAL_AND_NORMALISATION_MODEL_NAME = "gemini-3-flash-preview"


import datetime
GLOBAL_KNOWLEDGE_CUTOFF = datetime.datetime(2024, 4, 1)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on system environment variables


class TargetType(Enum):
    CHALLENGE = "challenge"
    METHOD = "method"
    RESEARCH_QUESTION = "research_question"


WORKSPACE_PATH = str(Path(__file__).resolve().parents[2])  # repo root
PAPERS_PARQUET = os.path.join(WORKSPACE_PATH, "data", "iclr_preprocessed_0_100.parquet")
# PAPERS_PARQUET = os.path.join(WORKSPACE_PATH, "data", "iclr_worst_preprocessed.parquet")
APS_PARQUET = os.path.join(WORKSPACE_PATH, "data", "aps_prl_135_26_preprocessed.parquet")

# --- Dataset selection ---
# Which source dataset to run the pipeline against. Can be overridden via the
# DATASET environment variable (e.g. DATASET=aps). Because the output/input/log
# paths below are bound at import time (save_idea.py, logger.py, load_goal_prompt
# all `from Config import <DIR>`), the dataset MUST be chosen via this env var so
# that every consumer resolves the same dataset-specific folders.
DATASET = os.getenv("DATASET", "aps").lower()
DATASET_PARQUETS = {
    "iclr": PAPERS_PARQUET,
    "aps": APS_PARQUET,
}
if DATASET not in DATASET_PARQUETS:
    raise ValueError(f"Unknown DATASET {DATASET!r}, expected one of {list(DATASET_PARQUETS)}")
# Parquet backing the currently selected dataset (used e.g. for the k_papers filter).
DATASET_PARQUET = DATASET_PARQUETS[DATASET]
# iclr is the historical default -> no extra path segment, so existing data paths
# (data/{target_type}/...) stay valid. Other datasets get their own subfolder.
_dataset_segment = "" if DATASET == "iclr" else DATASET

# --- Shared paths (not target-type specific) ---
DOWNLOADED_PAPERS_DIR = os.path.join(WORKSPACE_PATH, 'data', 'downloaded_papers')
DOWNLOADED_TEX_DIR = os.path.join(DOWNLOADED_PAPERS_DIR, 'tex')
DOWNLOADED_TXT_DIR = os.path.join(DOWNLOADED_PAPERS_DIR, 'txt')
DOWNLOADED_PDF_DIR = os.path.join(DOWNLOADED_PAPERS_DIR, 'pdf')
DOWNLOADED_TAR_DIR = os.path.join(DOWNLOADED_PAPERS_DIR, 'tar')

# --- Target type: change this to switch between method/challenge prediction ---
# Can be overridden via the TARGET_TYPE environment variable (e.g. TARGET_TYPE=method)
_target_type_env = os.getenv("TARGET_TYPE", "method").lower()
TARGET_TYPE = TargetType(_target_type_env)

# --- Experiment directory: set by run_all_target_types.py to isolate each run ---
# When EXPERIMENT_DIR is set (e.g. "ex_1"), paths are rooted under data/ex_1/{target_type}/
# When unset, falls back to data/{target_type}/ for backward compatibility
_experiment_dir = os.getenv("EXPERIMENT_DIR", "")

# --- Dataset + target-type-specific paths ---
# Rooted under data/{experiment}/{dataset}/{target_type}/ where the experiment and
# dataset segments collapse away when empty (no experiment / legacy iclr dataset),
# preserving the historical data/{target_type}/... layout.
DATA_ROOT = os.path.join(WORKSPACE_PATH, 'data')
_task_type_segments = [s for s in (_experiment_dir, _dataset_segment) if s] + [TARGET_TYPE.value]
TASK_TYPE_ROOT = os.path.join(DATA_ROOT, *_task_type_segments)

OUTPUT_DIR = os.path.join(TASK_TYPE_ROOT, 'outputs')
AI_RESEARCHER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'ai_researcher')
AI_SCIENTIST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'ai_scientist')
CHAIN_OF_IDEAS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'chain_of_ideas')
NAIVE_BASELINE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'naive_baseline')
AUTODS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'autods')
RESEARCH_AGENT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'research_agent')
AI_SCIENTIST_V2_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'ai_scientist_v2')
OSCAR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'oscar')
OSCAR_V2_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'oscar_v2')
OSCAR_V3_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'oscar_v3')
OSCAR_V4_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'oscar_v4')
OSCAR_V5_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'oscar_v5')
OSCAR_V6_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'oscar_v6')

AI_RESEARCHER_OUTPUT_JUDGE_DIR = os.path.join(AI_RESEARCHER_OUTPUT_DIR, 'llm_as_a_judge')
AI_SCIENTIST_OUTPUT_JUDGE_DIR = os.path.join(AI_SCIENTIST_OUTPUT_DIR, 'llm_as_a_judge')
CHAIN_OF_IDEAS_OUTPUT_JUDGE_DIR = os.path.join(CHAIN_OF_IDEAS_OUTPUT_DIR, 'llm_as_a_judge')
AUTODS_OUTPUT_JUDGE_DIR = os.path.join(AUTODS_OUTPUT_DIR, 'llm_as_a_judge')
RESEARCH_AGENT_OUTPUT_JUDGE_DIR = os.path.join(RESEARCH_AGENT_OUTPUT_DIR, 'llm_as_a_judge')
AI_SCIENTIST_V2_OUTPUT_JUDGE_DIR = os.path.join(AI_SCIENTIST_V2_OUTPUT_DIR, 'llm_as_a_judge')
OSCAR_OUTPUT_JUDGE_DIR = os.path.join(OSCAR_OUTPUT_DIR, 'llm_as_a_judge')
OSCAR_V2_OUTPUT_JUDGE_DIR = os.path.join(OSCAR_V2_OUTPUT_DIR, 'llm_as_a_judge')
OSCAR_V3_OUTPUT_JUDGE_DIR = os.path.join(OSCAR_V3_OUTPUT_DIR, 'llm_as_a_judge')
OSCAR_V4_OUTPUT_JUDGE_DIR = os.path.join(OSCAR_V4_OUTPUT_DIR, 'llm_as_a_judge')
OSCAR_V5_OUTPUT_JUDGE_DIR = os.path.join(OSCAR_V5_OUTPUT_DIR, 'llm_as_a_judge')
OSCAR_V6_OUTPUT_JUDGE_DIR = os.path.join(OSCAR_V6_OUTPUT_DIR, 'llm_as_a_judge')

PREDICTIONS_DIR = os.path.join(TASK_TYPE_ROOT, 'predictions')
AI_RESEARCHER_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'ai_researcher')
AI_SCIENTIST_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'ai_scientist')
CHAIN_OF_IDEAS_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'chain_of_ideas')
NAIVE_BASELINE_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'naive_baseline')
AUTODS_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'autods')
RESEARCH_AGENT_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'research_agent')
AI_SCIENTIST_V2_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'ai_scientist_v2')
OSCAR_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'oscar')
OSCAR_V2_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'oscar_v2')
OSCAR_V3_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'oscar_v3')
OSCAR_V4_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'oscar_v4')
OSCAR_V5_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'oscar_v5')
OSCAR_V6_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, 'oscar_v6')

_gt_segments = ([_dataset_segment] if _dataset_segment else []) + [TARGET_TYPE.value]
GROUND_TRUTH_DIR = os.path.join(DATA_ROOT, 'ground_truth', *_gt_segments)

_input_segments = ([_dataset_segment] if _dataset_segment else []) + [TARGET_TYPE.value]
INPUT_DIR = os.path.join(DATA_ROOT, 'inputs', *_input_segments)
EXAMPLE_CODE_DIR = os.path.join(INPUT_DIR, 'generated_example_code')

LOGS_DIR = os.path.join(TASK_TYPE_ROOT, 'logs')
AI_SCIENTIST_LOGS_DIR = os.path.join(LOGS_DIR, 'ai_scientist')
AI_RESEARCHER_LOGS_DIR = os.path.join(LOGS_DIR, 'ai_researcher')
CHAIN_OF_IDEAS_LOGS_DIR = os.path.join(LOGS_DIR, 'chain_of_ideas')
AUTODS_LOGS_DIR = os.path.join(LOGS_DIR, 'autods')
RESEARCH_AGENT_LOGS_DIR = os.path.join(LOGS_DIR, 'research_agent')
AI_SCIENTIST_V2_LOGS_DIR = os.path.join(LOGS_DIR, 'ai_scientist_v2')
NAIVE_BASELINE_LOGS_DIR = os.path.join(LOGS_DIR, 'naive_baseline')
OSCAR_LOGS_DIR = os.path.join(LOGS_DIR, 'oscar')
OSCAR_V2_LOGS_DIR = os.path.join(LOGS_DIR, 'oscar_v2')
OSCAR_V3_LOGS_DIR = os.path.join(LOGS_DIR, 'oscar_v3')
OSCAR_V4_LOGS_DIR = os.path.join(LOGS_DIR, 'oscar_v4')
OSCAR_V5_LOGS_DIR = os.path.join(LOGS_DIR, 'oscar_v5')
OSCAR_V6_LOGS_DIR = os.path.join(LOGS_DIR, 'oscar_v6')

# --- Non-path configuration ---


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your environment or create a .env file.")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MEMCACHE_PATH = os.path.join(WORKSPACE_PATH, '.cache')

ARXIV_REQUESTS_PER_MINUTE = 120


def make_dirs(dirs_to_make: list | str):
    if not isinstance(dirs_to_make, list):
        dirs_to_make = [dirs_to_make]

    import os
    for dir_path in dirs_to_make:
        os.makedirs(dir_path, exist_ok=True)

make_dirs([DOWNLOADED_PAPERS_DIR,
           DOWNLOADED_TEX_DIR,
           DOWNLOADED_TAR_DIR,
           DOWNLOADED_TXT_DIR,
           DOWNLOADED_PDF_DIR,
           AI_RESEARCHER_OUTPUT_DIR,
           AI_SCIENTIST_OUTPUT_DIR,
           CHAIN_OF_IDEAS_OUTPUT_DIR,
           EXAMPLE_CODE_DIR,
           AI_RESEARCHER_OUTPUT_JUDGE_DIR,
           AI_SCIENTIST_OUTPUT_JUDGE_DIR,
           CHAIN_OF_IDEAS_OUTPUT_JUDGE_DIR,
           GROUND_TRUTH_DIR,
           NAIVE_BASELINE_OUTPUT_DIR,
           AUTODS_OUTPUT_DIR,
           AUTODS_OUTPUT_JUDGE_DIR,
           RESEARCH_AGENT_OUTPUT_DIR,
           RESEARCH_AGENT_OUTPUT_JUDGE_DIR,
           AI_SCIENTIST_V2_OUTPUT_DIR,
           AI_SCIENTIST_V2_OUTPUT_JUDGE_DIR,
           LOGS_DIR,
           AI_SCIENTIST_LOGS_DIR,
           AI_RESEARCHER_LOGS_DIR,
           CHAIN_OF_IDEAS_LOGS_DIR,
           AUTODS_LOGS_DIR,
           RESEARCH_AGENT_LOGS_DIR,
           AI_SCIENTIST_V2_LOGS_DIR,
           NAIVE_BASELINE_LOGS_DIR,
           AI_RESEARCHER_PREDICTIONS_DIR,
           AI_SCIENTIST_PREDICTIONS_DIR,
           CHAIN_OF_IDEAS_PREDICTIONS_DIR,
           NAIVE_BASELINE_PREDICTIONS_DIR,
           AUTODS_PREDICTIONS_DIR,
           RESEARCH_AGENT_PREDICTIONS_DIR,
           AI_SCIENTIST_V2_PREDICTIONS_DIR,
           OSCAR_OUTPUT_DIR,
           OSCAR_OUTPUT_JUDGE_DIR,
           OSCAR_LOGS_DIR,
           OSCAR_PREDICTIONS_DIR,
           OSCAR_V2_OUTPUT_DIR,
           OSCAR_V2_OUTPUT_JUDGE_DIR,
           OSCAR_V2_LOGS_DIR,
           OSCAR_V2_PREDICTIONS_DIR,
           OSCAR_V3_OUTPUT_DIR,
           OSCAR_V3_OUTPUT_JUDGE_DIR,
           OSCAR_V3_LOGS_DIR,
           OSCAR_V3_PREDICTIONS_DIR,
           OSCAR_V4_OUTPUT_DIR,
           OSCAR_V4_OUTPUT_JUDGE_DIR,
           OSCAR_V4_LOGS_DIR,
           OSCAR_V4_PREDICTIONS_DIR,
           OSCAR_V5_OUTPUT_DIR,
           OSCAR_V5_OUTPUT_JUDGE_DIR,
           OSCAR_V5_LOGS_DIR,
           OSCAR_V5_PREDICTIONS_DIR,
           OSCAR_V6_OUTPUT_DIR,
           OSCAR_V6_OUTPUT_JUDGE_DIR,
           OSCAR_V6_LOGS_DIR,
           OSCAR_V6_PREDICTIONS_DIR,
           ])
