from dataclasses import dataclass
from math import lgamma
from scipy.special import digamma


@dataclass
class BetaBelief:
    """
    Beta distribution belief model for idea quality.

    Models P(idea is high quality) as Beta(alpha, beta_param).
    Prior is Beta(1, 1) = Uniform[0,1] (maximum ignorance).
    """
    alpha: float = 1.0
    beta_param: float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta_param)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta_param
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def update(self, score: float) -> None:
        """Update belief given an evaluation score in [0, 1]."""
        self.alpha += score
        self.beta_param += (1.0 - score)

    def kl_divergence_from(self, prior: 'BetaBelief') -> float:
        """
        Compute KL(self || prior) — the Bayesian surprise.

        Uses the closed-form KL divergence for Beta distributions:
        KL(Beta(a1,b1) || Beta(a2,b2)) =
            log(B(a2,b2)/B(a1,b1)) + (a1-a2)*psi(a1) + (b1-b2)*psi(b1)
            + (a2-a1+b2-b1)*psi(a1+b1)
        """
        a1, b1 = self.alpha, self.beta_param
        a2, b2 = prior.alpha, prior.beta_param

        log_beta_ratio = (lgamma(a2) + lgamma(b2) - lgamma(a2 + b2)) - \
                         (lgamma(a1) + lgamma(b1) - lgamma(a1 + b1))

        kl = log_beta_ratio + \
             (a1 - a2) * digamma(a1) + \
             (b1 - b2) * digamma(b1) + \
             (a2 - a1 + b2 - b1) * digamma(a1 + b1)

        return max(0.0, kl)

    def copy(self) -> 'BetaBelief':
        return BetaBelief(alpha=self.alpha, beta_param=self.beta_param)
