from horizon_bench.Config import INPUT_DIR
import os


def load_goal_prompt(filename: str) -> str:
    task_file_path = os.path.join(INPUT_DIR, filename)
    with open(task_file_path, 'r') as f:
        task_description = f.read().strip()
    return task_description