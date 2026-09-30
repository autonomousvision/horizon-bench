"""Problem Identification Validator Agent."""

from research_agent.agents.base_agent import ValidatorAgent
from research_agent.models.schemas import ValidationScores
from research_agent.prompts.validation import get_problem_validation_prompt


class ProblemValidator(ValidatorAgent):
    """Validates research problems on 5 metrics."""

    def generate(
        self,
        problem: str,
        task_context: str
    ) -> ValidationScores:
        """Validate a research problem."""
        prompt = get_problem_validation_prompt(problem, task_context)
        return self.call_llm(prompt, ValidationScores)
