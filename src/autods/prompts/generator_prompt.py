generator_system_prompt = """You are a creative research scientist specializing in generating novel and diverse research ideas. Your goal is to propose seed ideas that explore different corners of the research space. Each idea should be distinct from the others and represent a genuinely different approach or angle."""


def get_generator_user_prompt(task_description: str, existing_ideas: list[str], idea_index: int, total_ideas: int) -> str:
    existing_str = "\n".join(f"- {idea}" for idea in existing_ideas) if existing_ideas else "None yet."
    return f"""Task description:
{task_description}

You are generating research idea {idea_index} of {total_ideas}. Each idea should explore a DIFFERENT research direction.

Ideas already generated:
{existing_str}

Generate a new research idea that is DISTINCT from the ones above. Focus on a different angle, method, or problem formulation. The idea should be:
1. Novel -- not a trivial extension of existing work
2. Feasible -- could realistically be investigated
3. Significant -- addresses an important problem or gap

Provide the idea as a concise seed that can be elaborated later."""


def get_branch_directions_prompt(task_description: str, parent_direction: str, parent_elaboration: str = None) -> str:
    elaboration_ctx = f"\n\nThe parent idea was elaborated as:\n{parent_elaboration}" if parent_elaboration else ""
    return f"""Task description:
{task_description}

Parent research direction:
{parent_direction}{elaboration_ctx}

Generate 2-3 distinct child research directions that branch from the parent direction. Each child should:
1. Specialize or extend the parent idea in a meaningfully different way
2. Explore a different technical approach or application within the same theme
3. Be concrete enough to develop into a full research proposal

These directions will be used to expand a search tree of research ideas."""
