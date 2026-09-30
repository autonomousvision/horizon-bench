from horizon_bench.Config import INPUT_DIR, AI_SCIENTIST_OUTPUT_DIR, AI_RESEARCHER_OUTPUT_DIR, CHAIN_OF_IDEAS_OUTPUT_DIR, NAIVE_BASELINE_OUTPUT_DIR, AUTODS_OUTPUT_DIR, RESEARCH_AGENT_OUTPUT_DIR, AI_SCIENTIST_V2_OUTPUT_DIR, TargetType
import os
from ai_scientist.main import run_ai_scientist
from ai_researcher.main import run_ai_researcher
from coi_agent.main import run_chain_of_ideas_agent
from autods.main import run_autods
from research_agent.main import run_research_agent
from ai_scientist_v2.main import run_ai_scientist_v2
from oscar.main import run_oscar, run_oscar_v2, run_oscar_v3, run_oscar_v4, run_oscar_v5, run_oscar_v6
from multiprocessing import Pool, cpu_count
from functools import partial
from horizon_bench.naive_baseline import run_naive_baseline
import contextlib
from horizon_bench import Config




def check_task_attempted(question_filename: str, model_output_dir: str) -> bool:
    completed_tasks = os.listdir(model_output_dir)
    completed_tasks = [f for f in completed_tasks if f.endswith('.txt')]

    model_name = os.path.basename(model_output_dir)
    failed_tasks = os.listdir(os.path.join(Config.LOGS_DIR, model_name))
    failed_input_filenames = [f.replace('.json', '.txt') for f in failed_tasks if f.endswith('.json')]
    return question_filename in completed_tasks or question_filename in failed_input_filenames

def run_model_for_task(goal_prompt_filename: str, model_name: str, target_type: TargetType):
    """Run a specific model for a given task"""
    try:
        if model_name == "ai_scientist":
            print(f"Running AI Scientist for task {goal_prompt_filename}...")
            run_ai_scientist(question_filename=goal_prompt_filename)
        elif model_name == "ai_researcher":
            print(f"Running AI Researcher for task {goal_prompt_filename}...")
            run_ai_researcher(question_filename=goal_prompt_filename)
        elif model_name == "chain_of_ideas":
            print(f"Running Chain of Ideas Agent for task {goal_prompt_filename}...")
            run_chain_of_ideas_agent(question_filename=goal_prompt_filename)
        elif model_name == "naive_baseline":
            print(f"Running Naive Baseline for task {goal_prompt_filename}...")
            run_naive_baseline(goal_prompt_filename, target_type=target_type)
        elif model_name == "autods":
            print(f"Running AutoDS for task {goal_prompt_filename}...")
            run_autods(question_filename=goal_prompt_filename)
        elif model_name == "research_agent":
            print(f"Running ResearchAgent for task {goal_prompt_filename}...")
            run_research_agent(question_filename=goal_prompt_filename)
        elif model_name == "ai_scientist_v2":
            print(f"Running AI Scientist v2 for task {goal_prompt_filename}...")
            run_ai_scientist_v2(question_filename=goal_prompt_filename)
        elif model_name == "oscar":
            print(f"Running Oscar for task {goal_prompt_filename}...")
            run_oscar(question_filename=goal_prompt_filename)
        elif model_name == "oscar_v2":
            print(f"Running Oscar v2 for task {goal_prompt_filename}...")
            run_oscar_v2(question_filename=goal_prompt_filename)
        elif model_name == "oscar_v3":
            print(f"Running Oscar v3 for task {goal_prompt_filename}...")
            run_oscar_v3(question_filename=goal_prompt_filename)
        elif model_name == "oscar_v4":
            print(f"Running Oscar v4 for task {goal_prompt_filename}...")
            run_oscar_v4(question_filename=goal_prompt_filename)
        elif model_name == "oscar_v5":
            print(f"Running Oscar v5 for task {goal_prompt_filename}...")
            run_oscar_v5(question_filename=goal_prompt_filename)
        elif model_name == "oscar_v6":
            print(f"Running Oscar v6 for task {goal_prompt_filename}...")
            run_oscar_v6(question_filename=goal_prompt_filename)
    except Exception as e:
        print(f"Error occurred while running {model_name} for task {goal_prompt_filename}: {e}")

def process_task(question_filename: str, target_type: TargetType, models_to_run: list = None):
    """Process a single task file with all models"""
    if not question_filename.endswith('.txt'):
        return

    # Run all models for this task
    for model in models_to_run:
        if not check_task_attempted(question_filename, os.path.join(Config.OUTPUT_DIR, model)):
            print(f"Starting {model} for task {question_filename}...")
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                run_model_for_task(question_filename, model, target_type=target_type)


def get_n_pending_runs(models_to_run: list):
    task_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    pending_runs = {}
    for model in models_to_run:
        model_output_dir = os.path.join(Config.OUTPUT_DIR, model)
        completed_tasks = os.listdir(model_output_dir)
        completed_tasks = [f for f in completed_tasks if f.endswith('.txt')]
        pending_runs[model] = len([f for f in task_files if f not in completed_tasks])
    return pending_runs



# MODELS_TO_RUN = ['oscar_v5'] #, 'naive_baseline', "ai_scientist", "research_agent", 'ai_scientist_v2', "oscar"] #"ai_researcher", "chain_of_ideas", ] 
# dont run autods for now, because its expensive. 'autods', 


def run_baselines(target_type: TargetType, MODELS_TO_RUN: list, k_papers: int = None, dataset: str = None):
    # Which source dataset to run against (iclr or aps). Defaults to the dataset
    # selected at import time via the DATASET env var.
    if dataset is None:
        dataset = Config.DATASET
    # The input/output/log folders are bound at import time from Config.DATASET
    # (see save_idea.py / logger.py / load_goal_prompt). A dataset passed here that
    # differs from that would write to the wrong folders, so require them to match.
    if dataset != Config.DATASET:
        raise ValueError(
            f"dataset={dataset!r} does not match the process dataset {Config.DATASET!r}. "
            f"Select the dataset via the DATASET env var, e.g. `DATASET={dataset} python -m horizon_bench.pipeline.C_run_baselines`."
        )
    if dataset not in Config.DATASET_PARQUETS:
        raise ValueError(f"Unknown dataset {dataset!r}, expected one of {list(Config.DATASET_PARQUETS)}")

    # Determine number of workers (use fewer than CPU count to avoid overload)
    num_workers = 40

    task_files = os.listdir(INPUT_DIR)
    # task_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_0.txt')]
    if k_papers is not None:
        import pandas as pd
        df = pd.read_parquet(Config.DATASET_PARQUETS[dataset])
        top_k_ids = set(df['id'].iloc[:k_papers])
        task_files = [f for f in task_files if f.rsplit('_', 1)[0] in top_k_ids]

    
    print(f"Found pending tasks:")
    print('\n'.join(f"{model}: {count}" for model, count in get_n_pending_runs(MODELS_TO_RUN).items()))

    # Process tasks in parallel (imap_unordered so tasks complete independently)
    with Pool(processes=num_workers) as pool:
        for _ in pool.imap_unordered(partial(process_task, target_type=target_type, models_to_run=MODELS_TO_RUN), task_files):
            pass
    

if __name__ == "__main__":
    from horizon_bench.Config import TARGET_TYPE, TargetType, DATASET
    agents_env = os.environ.get("AGENTS_TO_RUN")
    if agents_env:
        MODELS_TO_RUN = agents_env.split(",")
    else:
        MODELS_TO_RUN = ["naive_baseline", "chain_of_ideas", "ai_scientist", "research_agent", 'ai_scientist_v2']
    run_baselines(target_type=TARGET_TYPE, MODELS_TO_RUN=MODELS_TO_RUN, dataset=DATASET, k_papers=20)