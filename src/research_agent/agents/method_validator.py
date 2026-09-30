"""Method Development Validator Agent."""

from research_agent.agents.base_agent import ValidatorAgent
from research_agent.models.schemas import ValidationScores
from research_agent.prompts.validation import get_method_validation_prompt


class MethodValidator(ValidatorAgent):
    """Validates research methods on 5 metrics."""

    def generate(
        self,
        method: str,
        problem: str
    ) -> ValidationScores:
        """Validate a research method."""
        prompt = get_method_validation_prompt(method, problem)
        return self.call_llm(prompt, ValidationScores)
