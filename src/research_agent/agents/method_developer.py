"""Method Development Generator Agent."""

from research_agent.agents.base_agent import GeneratorAgent
from research_agent.models.schemas import MethodDevelopment
from research_agent.prompts.method_development import get_method_development_prompt


class MethodDeveloper(GeneratorAgent):
    """Develops methods to address research problems."""

    def generate(
        self,
        problem_statement: str,
        papers_context: str,
        entities_context: str,
        feedback: str = None
    ) -> MethodDevelopment:
        """Generate a research method."""
        prompt = get_method_development_prompt(
            problem_statement=problem_statement,
            papers_context=papers_context,
            entities_context=entities_context,
            feedback=feedback
        )

        return self.call_llm(prompt, MethodDevelopment)
