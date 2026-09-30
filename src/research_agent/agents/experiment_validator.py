"""Experiment Design Validator Agent."""

from research_agent.agents.base_agent import ValidatorAgent
from research_agent.models.schemas import ValidationScores
from research_agent.prompts.validation import get_experiment_validation_prompt


class ExperimentValidator(ValidatorAgent):
    """Validates experiment designs on 5 metrics."""

    def generate(
        self,
        experiment: str,
        problem: str,
        method: str
    ) -> ValidationScores:
        """Validate an experiment design."""
        prompt = get_experiment_validation_prompt(experiment, problem, method)
        return self.call_llm(prompt, ValidationScores)
