import os
from horizon_bench.Config import WORKSPACE_PATH, make_dirs


GITHUB_API_KEY = None


# ------------------------------------------------------- Directories ------------------------------------------------------- #
AI_RESEARCHER_AGENT_WORKING_PATH = os.path.join(WORKSPACE_PATH, 'src/ai_researcher/data/codebases')
AI_RESEARCHER_AGENT_DATA_PATH = os.path.join(WORKSPACE_PATH, 'src/ai_researcher/data')

make_dirs([AI_RESEARCHER_AGENT_WORKING_PATH, AI_RESEARCHER_AGENT_DATA_PATH])
