"""
Abstract base class for ResearchAgent agents.

Defines common interface and shared utilities for all generator and validator agents.
"""

from abc import ABC, abstractmethod
from typing import Any
from horizon_bench.llm_api import get_formatted_chat_response
from research_agent.config import GENERATOR_TEMPERATURE, VALIDATOR_TEMPERATURE


class BaseAgent(ABC):
    """Abstract base class for all ResearchAgent agents."""

    def __init__(self, model: str, temperature: float = None):
        """
        Initialize agent.

        Args:
            model: Model name to use (e.g., "gpt-5-nano")
            temperature: Sampling temperature (None uses default)
        """
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def generate(self, **kwargs) -> Any:
        """
        Generate output based on inputs.

        Args:
            **kwargs: Agent-specific arguments

        Returns:
            Generated output (type varies by agent)
        """
        pass

    def call_llm(self, prompt: str, response_format: type, temperature: float = None):
        """
        Call LLM with structured output.

        Note: Currently uses Horizon Bench's get_formatted_chat_response which doesn't support
        model or temperature parameters. Uses DEFAULT_MODEL_NAME from Config.

        Args:
            prompt: Prompt string
            response_format: Pydantic model for response
            temperature: Override temperature (currently not used due to API limitations)

        Returns:
            Pydantic model instance
        """
        # Note: Horizon Bench's get_formatted_chat_response doesn't accept model/temperature
        # It uses DEFAULT_MODEL_NAME from Config
        return get_formatted_chat_response(
            response_format=response_format,
            user_prompt=prompt,
        )


class GeneratorAgent(BaseAgent):
    """Base class for generator agents."""

    def __init__(self, model: str):
        super().__init__(model, temperature=GENERATOR_TEMPERATURE)


class ValidatorAgent(BaseAgent):
    """Base class for validator agents."""

    def __init__(self, model: str):
        super().__init__(model, temperature=VALIDATOR_TEMPERATURE)
