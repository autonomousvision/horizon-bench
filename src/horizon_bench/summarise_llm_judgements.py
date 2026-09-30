import src.horizon_bench.Config as Config
import os
import json
import pydantic

class NumericJudgement(pydantic.BaseModel):
    soundness: int
    presentation: int
    contribution: int
    overall: int
    confidence: int

if __name__ == "__main__":
    pass
    # all_judged_task_filenames = []
    # for source_path in [Config.AI_RESEARCHER_OUTPUT_JUDGE_DIR, Config.AI_SCIENTIST_OUTPUT_JUDGE_DIR, Config.CHAIN_OF_IDEAS_OUTPUT_JUDGE_DIR]:
    #     task_filenames = [f for f in os.listdir(source_path) if f.endswith(".json")]
    #     all_judged_task_filenames += task_filenames

    # max_judged_task_id = max([int(f.split(".")[0]) for f in all_judged_task_filenames])

    # model_task_matrix = []
    # for source_path, judge_output_dir in [
    #     (Config.AI_RESEARCHER_OUTPUT_DIR, Config.AI_RESEARCHER_OUTPUT_JUDGE_DIR),
    #     (Config.AI_SCIENTIST_OUTPUT_DIR, Config.AI_SCIENTIST_OUTPUT_JUDGE_DIR),
    #     (Config.CHAIN_OF_IDEAS_OUTPUT_DIR, Config.CHAIN_OF_IDEAS_OUTPUT_JUDGE_DIR)
    # ]:
    #     # One row for every model
    #     model_task_matrix.append([None for _ in range(max_judged_task_id)])
        
    #     judgement_filenames = [f for f in os.listdir(judge_output_dir) if f.endswith(".json")]
    #     summaries = []
        
    #     for judgement_filename in judgement_filenames:
    #         judgement_filepath = os.path.join(judge_output_dir, judgement_filename)
    #         with open(judgement_filepath, "r") as f:
    #             judgement_data = json.load(f)
    #             numeric_judgement = NumericJudgement(**judgement_data)

    #             model_task_matrix[-1][int(judgement_filename.split(".")[0]) - 1] = numeric_judgement.confidence

    # for model_row in model_task_matrix:
    #     print(model_row)