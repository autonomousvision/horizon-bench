import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

COLOR_METHOD = "#387FF0"
COLOR_CHALLENGE = "#FFB366"
if __name__ == "__main__":
    agents =     ["OSCAR", "Naïve Baseline", "AI Scientist V2", "Research Agent", "Chain of Ideas", "AI Researcher", "AI Scientist"]
    method =     [0.822, 0.802, 0.780, 0.759, 0.755, 0.743, 0.701]
    challenge =  [0.855, 0.853, 0.815, 0.810, 0.799, 0.826, 0.723]

    x = np.arange(len(agents))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - width / 2, method, width, label="Method", color=COLOR_METHOD)
    ax.bar(x + width / 2, challenge, width, label="Challenge", color=COLOR_CHALLENGE)

    ax.set_ylabel("Soft Recall", fontsize=15)
    ax.set_ylim(0.65, 0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=40, ha="right", fontsize=15)
    ax.legend(fontsize=15)
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gpt5_barchart.svg", bbox_inches="tight")
    plt.show()
