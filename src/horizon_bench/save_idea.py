from horizon_bench.Config import AI_RESEARCHER_OUTPUT_DIR, AI_SCIENTIST_OUTPUT_DIR, CHAIN_OF_IDEAS_OUTPUT_DIR, NAIVE_BASELINE_OUTPUT_DIR, AUTODS_OUTPUT_DIR, RESEARCH_AGENT_OUTPUT_DIR, AI_SCIENTIST_V2_OUTPUT_DIR, OSCAR_OUTPUT_DIR, OSCAR_V2_OUTPUT_DIR, OSCAR_V3_OUTPUT_DIR, OSCAR_V4_OUTPUT_DIR, OSCAR_V5_OUTPUT_DIR, OSCAR_V6_OUTPUT_DIR


def save_idea(filename: str, idea: str, model_name: str):
    assert model_name in ['ai_researcher', 'ai_scientist', 'chain_of_ideas', 'naive_baseline', 'autods', 'research_agent', 'ai_scientist_v2', 'oscar', 'oscar_v2', 'oscar_v3', 'oscar_v4', 'oscar_v5', 'oscar_v6'], "Unsupported model name."
    import os
    if model_name == 'ai_researcher':
        output_dir = AI_RESEARCHER_OUTPUT_DIR
    elif model_name == 'ai_scientist':
        output_dir = AI_SCIENTIST_OUTPUT_DIR
    elif model_name == 'naive_baseline':
        output_dir = NAIVE_BASELINE_OUTPUT_DIR
    elif model_name == 'autods':
        output_dir = AUTODS_OUTPUT_DIR
    elif model_name == 'research_agent':
        output_dir = RESEARCH_AGENT_OUTPUT_DIR
    elif model_name == 'ai_scientist_v2':
        output_dir = AI_SCIENTIST_V2_OUTPUT_DIR
    elif model_name == 'oscar':
        output_dir = OSCAR_OUTPUT_DIR
    elif model_name == 'oscar_v2':
        output_dir = OSCAR_V2_OUTPUT_DIR
    elif model_name == 'oscar_v3':
        output_dir = OSCAR_V3_OUTPUT_DIR
    elif model_name == 'oscar_v4':
        output_dir = OSCAR_V4_OUTPUT_DIR
    elif model_name == 'oscar_v5':
        output_dir = OSCAR_V5_OUTPUT_DIR
    elif model_name == 'oscar_v6':
        output_dir = OSCAR_V6_OUTPUT_DIR
    else:
        output_dir = CHAIN_OF_IDEAS_OUTPUT_DIR

    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        f.write(idea + '\n')
    print(f"Idea saved to {filepath}")