"""
ResearchAgent main entry point for Horizon Bench.

Implements the three-phase iterative research idea generation pipeline.
"""

from horizon_bench.load_goal_prompt import load_goal_prompt
from horizon_bench.save_idea import save_idea
from horizon_bench.logger import initialize_logger
from research_agent.paper_retrieval import topic_to_papers
from research_agent.knowledge_builder import extract_entities_from_papers
from research_agent.pipeline.research_pipeline import run_research_pipeline
from research_agent.output_formatter import format_research_agent_output
from research_agent.config import MAX_PAPERS


def run_research_agent(question_filename: str) -> None:
    """
    Main entry point for ResearchAgent.

    Follows Horizon Bench agent interface pattern:
    1. Load topic from input file
    2. Retrieve relevant papers via Semantic Scholar + arXiv
    3. Extract knowledge entities from papers
    4. Run three-phase pipeline (Problem → Method → Experiment)
    5. Format output as markdown
    6. Save to output directory

    Args:
        question_filename: Input file name (e.g., "question_4.txt")
    """
    initialize_logger("research_agent", question_filename)
    print("\n" + "="*80)
    print("RESEARCHAGENT - Starting Idea Generation")
    print("="*80 + "\n")

    # ===== Step 1: Load task description =====
    print("Loading task description...")
    task_description = load_goal_prompt(question_filename)
    print(f"Task: {task_description[:100]}...")
    print()

    # ===== Step 2: Retrieve relevant papers =====
    print("Retrieving relevant papers from Semantic Scholar and arXiv...")
    try:
        papers = topic_to_papers(task_description, max_papers=MAX_PAPERS)
        print(f"✓ Retrieved {len(papers)} papers")
    except Exception as e:
        print(f"✗ Error retrieving papers: {e}")
        print("Proceeding with empty paper list (degraded mode)")
        papers = []

    print()

    # ===== Step 3: Extract knowledge entities =====
    print("Extracting knowledge entities from papers...")
    try:
        entities = extract_entities_from_papers(papers)
        print(f"✓ Extracted {len(entities)} unique entities")
    except Exception as e:
        print(f"✗ Error extracting entities: {e}")
        print("Proceeding without entities (degraded mode)")
        entities = []

    print()

    # ===== Step 4: Run three-phase research pipeline =====
    print("Running three-phase research pipeline...")
    try:
        research_output = run_research_pipeline(
            papers=papers,
            entities=entities,
            task_description=task_description
        )
        print("✓ Pipeline completed successfully")
    except Exception as e:
        print(f"✗ Error in research pipeline: {e}")
        raise  # Re-raise to signal failure

    print()

    # ===== Step 5: Format output as markdown =====
    print("Formatting output...")
    markdown_output = format_research_agent_output(
        problem=research_output.problem,
        method=research_output.method,
        experiment=research_output.experiment,
        problem_validation=research_output.problem_validation,
        method_validation=research_output.method_validation,
        experiment_validation=research_output.experiment_validation,
    )
    print("✓ Output formatted as markdown")
    print()

    # ===== Step 6: Save output =====
    print("Saving idea to output directory...")
    save_idea(
        filename=question_filename,
        idea=markdown_output,
        model_name='research_agent'
    )
    print(f"✓ Saved to research_agent output directory")

    print("\n" + "="*80)
    print("RESEARCHAGENT - Idea Generation Complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_research_agent("2408.03314_0.txt")