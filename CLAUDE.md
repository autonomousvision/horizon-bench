# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Horizon Bench is a research framework for comparing different AI Scientist implementations. It provides reproducible implementations of multiple AI research agents (AI Scientist, AI Researcher, Chain of Ideas Agent, AutoDS, Research Agent, AI Scientist v2) plus a naive baseline, all running against the same evaluation framework.

## Setup and Installation

```bash
conda env create --name horizon_bench python=3.13 -c conda-forge
conda activate horizon_bench
pip install -r requirements.txt
sudo apt install pandoc  # Required for COI Agent's paper text extraction
cp .env.example .env     # Add OPENAI_API_KEY and GOOGLE_API_KEY
```

**GROBID** (required for OpenReview PDF text extraction):
```bash
sudo apt install -y openjdk-17-jdk
pip install scipdf-parser
wget https://github.com/kermitt2/grobid/archive/0.8.2.zip && unzip 0.8.2.zip
cd grobid-0.8.2 && ./gradlew clean install
# Start before use: cd grobid-0.8.2 && ./gradlew run (runs on localhost:8070)
```

`WORKSPACE_PATH` in `src/horizon_bench/Config.py` is derived from the file location (the repo root), so no path setup is needed. Directories are auto-created on Config import.

## Running the Pipeline

All scripts must be run from the repo root with PYTHONPATH set:

```bash
cd /path/to/horizon_bench
export PYTHONPATH=$(pwd)/src

# Individual pipeline steps (run sequentially):
python -m horizon_bench.pipeline.A_preprocess_iclr_dataset
python -m horizon_bench.pipeline.B_make_input_prompts
python -m horizon_bench.pipeline.C_run_baselines
python -m horizon_bench.pipeline.D_extract_targets
python -m horizon_bench.pipeline.E_generate_histogram

# Or run the full pipeline for all target types:
python -m horizon_bench.pipeline.run_all_target_types
```

**Target type** is controlled by the `TARGET_TYPE` environment variable (default: `"challenge"`). Options: `challenge`, `method`, `research_question`. The full pipeline re-runs independently for each target type.

Individual agent notebooks for manual testing:
- `src/ai_researcher/reproducing_ai_researcher.ipynb`
- `src/ai_scientist/reproducing_ai_scientist.ipynb`
- `src/coi_agent/reproducing_coi_agent.ipynb`

**No test suite or linting tools are configured.**

## Architecture

### Multi-Provider LLM Abstraction

The LLM layer uses a dispatcher pattern:
- [llm_api.py](src/horizon_bench/llm_api.py) routes calls to either `openai_api.py` or `gemini_api.py` based on `Config.LLMAPINAME` (set in Config.py, currently `LLMAPIType.GEMINI`)
- Three main functions: `chat()`, `get_formatted_chat_response()` (Pydantic structured output), `chat_with_agentic_tool_use()` (OpenAI only; Gemini raises NotImplementedError)
- All code uses OpenAI-style message dicts (`{"role": ..., "content": ...}`) everywhere; Gemini adapter converts internally
- Both providers have independent joblib caches; pass `invalidate_cache=True` to force refresh

### Pipeline Flow

1. **A: Preprocess** — Loads `data/iclr.parquet`, filters year 2026 papers, sorts by reviewer scores, saves top/bottom 100
2. **B: Make Input Prompts** — Uses LLM to generate one-sentence goal prompts per paper, excluding the target type from the prompt
3. **C: Run Baselines** — Runs all agents in parallel (multiprocessing Pool, 40 workers default). Checks task completion to skip already-processed files. Logs errors to JSON
4. **D: Extract Targets** — Uses LLM to extract specific predictions from agent outputs; generates ground truth from OpenReview PDF text via GROBID
5. **E: Generate Histograms** — Creates evaluation visualizations

### Agent Convention

All agents follow the same interface pattern:
- Entry point: `run_<agent_name>(question_filename)` in each agent's `main.py`
- Load input via `load_goal_prompt(filename)` from `INPUT_DIR`
- Initialize logger via `initialize_logger()` (auto-captures uncaught exceptions to JSON logs)
- Save output via `save_idea(filename, idea_text, model_name)` to agent-specific `OUTPUT_DIR`
- Output files named `{arxiv_id}_{generation_index}.txt`

Agents live at: `src/{ai_scientist,ai_researcher,coi_agent,autods,research_agent,ai_scientist_v2}/main.py`

### Core Components

- [Config.py](src/horizon_bench/Config.py): Central config — paths, model names, API keys, target type, directory setup. All data paths are target-type-specific (`data/{target_type}/...`)
- [logger.py](src/horizon_bench/logger.py): Global exception hook that logs failures to `{LOGS_DIR}/{model}/{filename}.json` — individual task failures don't stop the pipeline
- [save_idea.py](src/horizon_bench/save_idea.py) / `load_goal_prompt()`: Shared I/O for all agents

### Key Implementation Details

1. **Caching**: `joblib.Memory` at `.cache/` reduces API costs. Both OpenAI and Gemini have separate caches
2. **Rate Limiting**: arXiv API limited to 120 requests/minute (Config.py)
3. **Knowledge Cutoff**: Global cutoff date `2024-04-01` — agents should not reference papers after this date
4. **Structured Outputs**: All LLM structured responses use Pydantic `BaseModel` classes, parsed via `get_formatted_chat_response()`
5. **Error Isolation**: Pipeline uses `imap_unordered` — individual task failures are logged and skipped, not propagated

## Model Configuration

Set in [Config.py](src/horizon_bench/Config.py):
- `LLMAPINAME`: Switch between `LLMAPIType.OPENAI` and `LLMAPIType.GEMINI`
- `DEFAULT_MODEL_NAME`: Currently `"gemini-3-flash-preview"` (OpenAI options: `gpt-5`, `gpt-5-nano`)

## Known Limitations

- COI Agent only works with arXiv papers (uses LaTeX source + pandoc, not general PDFs)
- AI Researcher allows arbitrary code execution via LLM tool use (security risk)
- Agentic tool use (`chat_with_agentic_tool_use`) only implemented for OpenAI, not Gemini
