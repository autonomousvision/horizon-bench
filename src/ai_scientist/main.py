import os
from horizon_bench.Config import WORKSPACE_PATH
from ai_scientist.prompts.idea_refinement_prompt import idea_refinement_prompt
from ai_scientist.extract_json import extract_json
from ai_scientist.extract_json import extract_json
from horizon_bench.load_goal_prompt import load_goal_prompt
from horizon_bench.llm_api import chat
from ai_scientist.extract_json import json_to_markdown
from horizon_bench.save_idea import save_idea
from ai_scientist.task_to_example_code import task_to_example_code
from horizon_bench.logger import initialize_logger


def run_ai_scientist(question_filename: str):
    initialize_logger("ai_scientist", question_filename)
    task_description = load_goal_prompt(question_filename)

    STARTING_IDEAS = []
    # {
    #     "Name": "learning_rate_schedule",
    #     "Title": "Adaptive Learning Rate Schedules: Comparing different learning rate schedules for diffusion models.",
    #     "Experiment": "In this experiment, we compare the performance of different learning rate schedules on diffusion model performance. We use the final estimated KL as the evaluation metric.",
    #     "Interestingness": 4,
    #     "Feasibility": 10,
    #     "Novelty": 3
    # }
    # ]

    EXPERIMENT_CODE = task_to_example_code(question_filename, task_description)
    # CODE_FILE_PATH = os.path.join(WORKSPACE_PATH, 'src/ai_scientist', "example_experiment.py")
    # EXPERIMENT_CODE = ""
    # if CODE_FILE_PATH and os.path.exists(CODE_FILE_PATH):
    #     # They define a code experiment for context. Major limitation, as writing such example code is time-consuming.
    #     with open(CODE_FILE_PATH, "r") as f:
    #         EXPERIMENT_CODE = f.read()

    MAX_ROUNDS = 5
    # can be empty list
    
    SYSTEM_PROMPT = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field."""

    from ai_scientist.prompts.idea_prompt import user_idea_prompt
    import json

    user_prompt = user_idea_prompt.format(
        task_description=task_description, 
        code=EXPERIMENT_CODE if EXPERIMENT_CODE else "",
        prev_ideas_string="\n\n".join(json.dumps(x) for x in STARTING_IDEAS), 
        num_reflections=MAX_ROUNDS
    )

    response = chat(user_prompt=user_prompt, system_prompt=SYSTEM_PROMPT)
    json_output = extract_json(response)
    message = {"role": "user", "content": user_prompt}
    message_history = [message]

    for current_round_number in range(2, MAX_ROUNDS + 1):
        user_prompt = idea_refinement_prompt.format(
            current_round=current_round_number,
            num_reflections=MAX_ROUNDS,
        )

        response = chat(user_prompt=user_prompt, system_prompt=SYSTEM_PROMPT, message_history=message_history)
        message = {"role": "user", "content": user_prompt}
        message_history.append(message)

        # They use a brittle regex-based JSON extraction method.
        # I could easily replace this with OpenAI's inbuilt output parser.
        json_output = extract_json(response)
        assert json_output is not None, "Failed to extract JSON from LLM output"

        if "I am done" in response:
            # print(f"Idea generation finished after {current_round_number - 1} rounds of refinement.")
            break

    str_idea = json_to_markdown(json_output)
    # print("Final idea:\n", str_idea)
    save_idea(question_filename, idea=str_idea, model_name='ai_scientist')

if __name__ == "__main__":
    run_ai_scientist(question_filename="0JLUFJMo5p_0.txt" )
    pass