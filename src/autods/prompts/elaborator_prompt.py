elaborator_system_prompt = """You are a meticulous research scientist who excels at developing high-level research concepts into detailed, rigorous proposals. You provide clear technical details, mathematical formulations where appropriate, and concrete methodology."""


def get_elaborator_user_prompt(task_description: str, idea_direction: str) -> str:
    return f"""Task description:
{task_description}

Research idea direction:
{idea_direction}

Develop this research direction into a comprehensive, detailed research proposal. Include:
1. **Motivation**: Why this problem matters, what gap it fills, and connection to existing work
2. **Proposed Method**: A step-by-step description of the technical approach
3. **Technical Details**: Specific algorithms, architectures, or mathematical formulations
4. **Expected Outcomes**: What results you anticipate and how you would measure success
5. **Novelty Claim**: What specifically is new about this approach versus prior work

Be thorough but realistic. Avoid vague claims -- ground every aspect in concrete technical details."""
