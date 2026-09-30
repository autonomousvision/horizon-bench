"""
Validation criteria definitions for ResearchAgent.

Defines the 5-metric validation framework for each phase:
- Problem Identification: Clarity, Relevance, Originality, Feasibility, Significance
- Method Development: Clarity, Relevance, Originality, Feasibility, Significance
- Experiment Design: Clarity, Validity, Robustness, Feasibility, Reproducibility
"""

from dataclasses import dataclass


@dataclass
class ValidationMetric:
    """Definition of a single validation metric."""
    name: str
    description: str
    scoring_guide: str


# ===== Problem Identification Validation Metrics =====

PROBLEM_CLARITY = ValidationMetric(
    name="Clarity",
    description="How clearly is the research problem stated?",
    scoring_guide="""
    1-3: Vague or confusing problem statement
    4-6: Problem is stated but lacks precision or specificity
    7-8: Clear problem statement with most details specified
    9-10: Crystal clear, well-defined problem that is immediately understandable
    """
)

PROBLEM_RELEVANCE = ValidationMetric(
    name="Relevance",
    description="How relevant is this problem to current research and applications?",
    scoring_guide="""
    1-3: Minimal relevance to current research or applications
    4-6: Somewhat relevant but not a pressing concern
    7-8: Highly relevant to active research areas
    9-10: Addresses a critical gap in current research with broad implications
    """
)

PROBLEM_ORIGINALITY = ValidationMetric(
    name="Originality",
    description="How novel or original is this problem?",
    scoring_guide="""
    1-3: Well-studied problem with no new angle
    4-6: Some novel aspects but largely familiar
    7-8: Novel formulation or perspective on the problem
    9-10: Highly original problem that hasn't been addressed before
    """
)

PROBLEM_FEASIBILITY = ValidationMetric(
    name="Feasibility",
    description="How feasible is it to make progress on this problem?",
    scoring_guide="""
    1-3: Currently infeasible with available methods or resources
    4-6: Challenging but potentially addressable
    7-8: Feasible with reasonable effort and resources
    9-10: Highly feasible with clear path to investigation
    """
)

PROBLEM_SIGNIFICANCE = ValidationMetric(
    name="Significance",
    description="How significant would solving this problem be?",
    scoring_guide="""
    1-3: Minimal impact on the field
    4-6: Modest contribution to specific subarea
    7-8: Significant contribution with broader implications
    9-10: Transformative impact on the field or related areas
    """
)

PROBLEM_METRICS = [
    PROBLEM_CLARITY,
    PROBLEM_RELEVANCE,
    PROBLEM_ORIGINALITY,
    PROBLEM_FEASIBILITY,
    PROBLEM_SIGNIFICANCE
]


# ===== Method Development Validation Metrics =====

METHOD_CLARITY = ValidationMetric(
    name="Clarity",
    description="How clearly is the proposed method described?",
    scoring_guide="""
    1-3: Vague or incomplete method description
    4-6: Method described but missing key technical details
    7-8: Clear description with most technical details specified
    9-10: Comprehensive, crystal clear method description
    """
)

METHOD_RELEVANCE = ValidationMetric(
    name="Relevance",
    description="How well does the method address the identified problem?",
    scoring_guide="""
    1-3: Weak or unclear connection to the problem
    4-6: Addresses some aspects of the problem
    7-8: Directly addresses the core problem
    9-10: Perfect alignment with problem, addresses all key aspects
    """
)

METHOD_ORIGINALITY = ValidationMetric(
    name="Originality",
    description="How novel or innovative is the proposed method?",
    scoring_guide="""
    1-3: Straightforward application of existing methods
    4-6: Minor variations on existing approaches
    7-8: Novel combination or significant modification of existing methods
    9-10: Highly innovative approach with new ideas
    """
)

METHOD_FEASIBILITY = ValidationMetric(
    name="Feasibility",
    description="How feasible is it to implement this method?",
    scoring_guide="""
    1-3: Impractical or overly complex to implement
    4-6: Implementable but requires significant resources
    7-8: Feasible with reasonable computational/data resources
    9-10: Highly practical and straightforward to implement
    """
)

METHOD_SIGNIFICANCE = ValidationMetric(
    name="Significance",
    description="How significant would this method's contribution be?",
    scoring_guide="""
    1-3: Minimal expected improvement over existing methods
    4-6: Modest incremental improvement
    7-8: Significant advancement over current approaches
    9-10: Breakthrough method with transformative potential
    """
)

METHOD_METRICS = [
    METHOD_CLARITY,
    METHOD_RELEVANCE,
    METHOD_ORIGINALITY,
    METHOD_FEASIBILITY,
    METHOD_SIGNIFICANCE
]


# ===== Experiment Design Validation Metrics =====

EXPERIMENT_CLARITY = ValidationMetric(
    name="Clarity",
    description="How clearly is the experimental setup described?",
    scoring_guide="""
    1-3: Vague or incomplete experimental description
    4-6: Basic setup described but missing important details
    7-8: Clear description with most details specified
    9-10: Comprehensive, reproducible experimental description
    """
)

EXPERIMENT_VALIDITY = ValidationMetric(
    name="Validity",
    description="How well does the experiment validate the proposed method?",
    scoring_guide="""
    1-3: Weak or inappropriate validation approach
    4-6: Validates some aspects but has gaps
    7-8: Solid validation covering main claims
    9-10: Comprehensive validation of all key claims
    """
)

EXPERIMENT_ROBUSTNESS = ValidationMetric(
    name="Robustness",
    description="How robust is the experimental design?",
    scoring_guide="""
    1-3: Limited scope, single setting only
    4-6: Tests a few settings but narrow scope
    7-8: Tests multiple settings and conditions
    9-10: Comprehensive testing across diverse scenarios
    """
)

EXPERIMENT_FEASIBILITY = ValidationMetric(
    name="Feasibility",
    description="How feasible is it to conduct this experiment?",
    scoring_guide="""
    1-3: Requires excessive resources or unavailable data
    4-6: Demanding but potentially achievable
    7-8: Feasible with reasonable resources
    9-10: Highly practical and efficient to execute
    """
)

EXPERIMENT_REPRODUCIBILITY = ValidationMetric(
    name="Reproducibility",
    description="How reproducible is the proposed experiment?",
    scoring_guide="""
    1-3: Insufficient details for reproduction
    4-6: Some details provided but gaps remain
    7-8: Most details specified for reproduction
    9-10: Complete specification enabling exact reproduction
    """
)

EXPERIMENT_METRICS = [
    EXPERIMENT_CLARITY,
    EXPERIMENT_VALIDITY,
    EXPERIMENT_ROBUSTNESS,
    EXPERIMENT_FEASIBILITY,
    EXPERIMENT_REPRODUCIBILITY
]


def get_metrics_for_phase(phase: str) -> list[ValidationMetric]:
    """Get validation metrics for a specific phase."""
    if phase == "problem":
        return PROBLEM_METRICS
    elif phase == "method":
        return METHOD_METRICS
    elif phase == "experiment":
        return EXPERIMENT_METRICS
    else:
        raise ValueError(f"Unknown phase: {phase}")


def format_metrics_for_prompt(metrics: list[ValidationMetric]) -> str:
    """Format validation metrics for inclusion in prompts."""
    formatted = []
    for metric in metrics:
        formatted.append(f"**{metric.name}**: {metric.description}")
        formatted.append(f"Scoring Guide:\n{metric.scoring_guide}")
        formatted.append("")
    return "\n".join(formatted)
