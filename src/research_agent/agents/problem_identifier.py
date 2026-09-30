"""Problem Identification Generator Agent."""

from research_agent.agents.base_agent import GeneratorAgent
from research_agent.models.schemas import ProblemIdentification
from research_agent.prompts.problem_identification import get_problem_identification_prompt


class ProblemIdentifier(GeneratorAgent):
    """Generates research problems from task description and papers."""

    def generate(
        self,
        task_description: str,
        papers_context: str,
        entities_context: str,
        feedback: str = None
    ) -> ProblemIdentification:
        """Generate a research problem."""
        prompt = get_problem_identification_prompt(
            task_description=task_description,
            papers_context=papers_context,
            entities_context=entities_context,
            feedback=feedback
        )

        return self.call_llm(prompt, ProblemIdentification)
