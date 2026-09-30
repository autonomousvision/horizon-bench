"""
Configuration for ResearchAgent.

This module defines ResearchAgent-specific constants while importing
shared configuration from Horizon Bench's Config.py.
"""

# Import shared configuration from Horizon Bench
from horizon_bench.Config import (
    DEFAULT_MODEL_NAME,
    OPENAI_API_KEY,
    MEMCACHE_PATH,
    GLOBAL_KNOWLEDGE_CUTOFF,
)

# ===== ResearchAgent Configuration =====

# Pipeline Configuration
MAX_ITERATIONS_PER_PHASE = 2  # Number of refinement iterations per phase
MIN_VALIDATION_SCORE = 7.0     # Minimum average score to proceed to next phase
MAX_PAPERS = 5                 # Number of papers to retrieve for each topic

# Model Configuration
GENERATOR_MODEL = "gpt-5-nano"  # Model for generator agents
VALIDATOR_MODEL = "gpt-5-nano"  # Model for validator agents

# Timeout Configuration (in seconds)
PAPER_RETRIEVAL_TIMEOUT = 120   # Timeout for paper retrieval
PHASE_TIMEOUT = 300             # Timeout per phase execution

# Knowledge Entity Configuration
USE_KNOWLEDGE_ENTITIES = True   # Enable/disable entity co-occurrence scoring
ENTITY_EXTRACTION_METHOD = "llm"  # Method for entity extraction ("llm" or "simple")
MAX_ENTITIES_PER_PAPER = 10    # Maximum entities to extract per paper
TOP_K_RELEVANT_ENTITIES = 10   # Top-k entities to use for context

# Validation Metrics Configuration
VALIDATION_METRICS_PROBLEM = [
    "Clarity",
    "Relevance",
    "Originality",
    "Feasibility",
    "Significance"
]

VALIDATION_METRICS_METHOD = [
    "Clarity",
    "Relevance",
    "Originality",
    "Feasibility",
    "Significance"
]

VALIDATION_METRICS_EXPERIMENT = [
    "Clarity",
    "Validity",
    "Robustness",
    "Feasibility",
    "Reproducibility"
]

# Temperature settings for creativity vs consistency
GENERATOR_TEMPERATURE = 0.7  # Higher for more creative generation
VALIDATOR_TEMPERATURE = 0.3  # Lower for more consistent validation
