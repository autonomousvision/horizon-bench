from ai_scientist.prompts.example_experiment_prompt import get_example_experiment_prompt
from horizon_bench.Config import EXAMPLE_CODE_DIR
from pydantic import BaseModel

class ExampleCodeResult(BaseModel):
        code: str

def task_to_example_code(question_filename: str, task_description: str) -> str:
    from horizon_bench.llm_api import get_formatted_chat_response

    user_prompt = get_example_experiment_prompt(task_description=task_description)

    response = get_formatted_chat_response(
        user_prompt=user_prompt,
        response_format=ExampleCodeResult
    )
    save_example_code(question_filename, response.code)
    return response.code


def save_example_code(question_filename: str, example_code: str):
    from horizon_bench.Config import EXAMPLE_CODE_DIR
    import os

    os.makedirs(EXAMPLE_CODE_DIR, exist_ok=True)
    file_path = os.path.join(EXAMPLE_CODE_DIR, f"{question_filename.split('.')[0]}.py")
    with open(file_path, "w") as f:
        f.write(example_code)

import os
if __name__ == "__main__":
    print(os.path.basename('some/dir/somefile.txt'))