"""Run the full pipeline (A → B → C) sequentially for each TargetType."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TARGET_TYPES = ["method"] # "","method"

# Which agents to run in C_run_baselines, D_extract_targets, E_generate_histogram.
AGENTS_TO_RUN = [
    # 'oscar_v5',
    # 'oscar_v6',
    'naive_baseline',
    # 'chain_of_ideas',
    # 'ai_scientist',
    # 'research_agent',
    # 'ai_scientist_v2',
    # 'oscar',
    # 'oscar_v2',
    # 'oscar_v3',
    # 'oscar_v4',
]

# List of (LLMAPINAME, DEFAULT_MODEL_NAME) pairs to cycle through.
# Each combination runs the full pipeline independently.
MODEL_CONFIGS = [
    # ("vllm", "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
    # ("vllm", "Qwen/Qwen2.5-14B-Instruct"),
    # ("vllm", "Qwen/Qwen2.5-7B-Instruct"),
    ("gemini", "gemini-3-flash-preview"),
    # ("openai", "gpt-5"),
]

PIPELINE_SCRIPTS = [
    # "horizon_bench/pipeline/A_preprocess_iclr_dataset.py",
    # "horizon_bench/pipeline/B_make_input_prompts.py",
    # "horizon_bench/pipeline/C_run_baselines.py",
    "horizon_bench/pipeline/D_extract_targets.py",
    "horizon_bench/pipeline/E_generate_histogram.py",
]

# Set to an existing experiment directory name (e.g. "ex_3") to resume,
# or None to create a new experiment.
RESUME_EXPERIMENT = "ex_7_gemini_flash_preview_oscar"

CONFIG_SRC = Path(__file__).parent.parent / "Config.py"


def get_next_experiment_dir(data_dir: str) -> str:
    """Return the next available ex_{i} directory name under data_dir."""
    existing = [
        int(m.group(1))
        for p in Path(data_dir).iterdir()
        if p.is_dir() and (m := re.fullmatch(r"ex_(\d+)", p.name))
    ]
    next_i = max(existing, default=0) + 1
    return f"ex_{next_i}"


def save_config_snapshot(experiment_dir: str):
    from horizon_bench.Config import WORKSPACE_PATH
    exp_path = os.path.join(WORKSPACE_PATH, "data", experiment_dir)
    os.makedirs(exp_path, exist_ok=True)
    shutil.copy2(CONFIG_SRC, os.path.join(exp_path, "Config.py"))
    print(f"Saved Config.py snapshot to {exp_path}/Config.py")


if __name__ == "__main__":
    from horizon_bench.Config import WORKSPACE_PATH
    data_dir = os.path.join(WORKSPACE_PATH, "data")

    if RESUME_EXPERIMENT:
        experiment_dir = RESUME_EXPERIMENT
        exp_path = os.path.join(data_dir, experiment_dir)
        if not os.path.isdir(exp_path):
            print(f"ERROR: experiment directory does not exist: {exp_path}")
            sys.exit(1)
        print(f"Resuming experiment: data/{experiment_dir}")
    else:
        experiment_dir = get_next_experiment_dir(data_dir)
        os.makedirs(os.path.join(data_dir, experiment_dir), exist_ok=True)
        print(f"New experiment directory: data/{experiment_dir}")

    save_config_snapshot(experiment_dir)
    for llmapiname, model_name in MODEL_CONFIGS:
        if llmapiname == "vllm":
            from horizon_bench.vllm_api import restart_vllm_server
            # restart_vllm_server(model_name)

        for target_type in TARGET_TYPES:
            for script in PIPELINE_SCRIPTS:
                print(f"\n{'='*60}")
                print(f"Running {script} with TARGET_TYPE={target_type} LLMAPINAME={llmapiname} DEFAULT_MODEL_NAME={model_name} EXPERIMENT_DIR={experiment_dir}")
                print(f"{'='*60}\n")
                env = {
                    **os.environ,
                    "TARGET_TYPE": target_type,
                    "LLMAPINAME": llmapiname,
                    "DEFAULT_MODEL_NAME": model_name,
                    "EXPERIMENT_DIR": experiment_dir,
                    "AGENTS_TO_RUN": ",".join(AGENTS_TO_RUN),
                }
                result = subprocess.run(
                    [sys.executable, "-m", script.replace("/", ".").removesuffix(".py")],
                    env=env,
                )
                if result.returncode != 0:
                    print(f"FAILED: {script} with TARGET_TYPE={target_type} LLMAPINAME={llmapiname} DEFAULT_MODEL_NAME={model_name} (exit code {result.returncode})")
                    sys.exit(1)

    print("\nAll pipeline runs completed successfully.")
