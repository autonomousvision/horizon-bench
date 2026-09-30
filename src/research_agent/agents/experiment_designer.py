"""Experiment Design Generator Agent."""

from research_agent.agents.base_agent import GeneratorAgent
from research_agent.models.schemas import ExperimentDesign
from research_agent.prompts.experiment_design import get_experiment_design_prompt


class ExperimentDesigner(GeneratorAgent):
    """Designs experiments to validate methods."""

    def generate(
        self,
        problem_statement: str,
        method_description: str,
        papers_context: str,
        feedback: str = None
    ) -> ExperimentDesign:
        """Generate an experiment design."""
        prompt = get_experiment_design_prompt(
            problem_statement=problem_statement,
            method_description=method_description,
            papers_context=papers_context,
            feedback=feedback
        )

        return self.call_llm(prompt, ExperimentDesign)
