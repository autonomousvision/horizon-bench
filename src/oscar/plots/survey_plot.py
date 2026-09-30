import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from oscar.plots.barplot import COLOR_CHALLENGE, COLOR_METHOD, REPO_ROOT, FIGURES_DIR

df = pd.read_csv(REPO_ROOT / "data" / "combined_survey.csv")



rank_cols = [c for c in df.columns if c.endswith("_rank")]
num_ranked = df[rank_cols].notna().sum(axis=1)

for col in rank_cols:
    relative_col = col.replace("_rank", "_relative_rank")
    df[relative_col] = np.where(df[col].isna(), np.nan, (num_ranked - df[col]) / num_ranked * 100)

# Prepare data for plotting
# relative_cols = [c.replace("_rank", "_relative_rank") for c in rank_cols]
model_names = {
    "naive_baseline_relative_rank": "Naive Baseline",
    "ai_researcher_relative_rank": "AI Researcher",
    "ai_scientist_relative_rank": "AI Scientist",
    "ai_scientist_v2_relative_rank": "AI Scientist v2",
    "chain_of_ideas_relative_rank": "Chain of Ideas",
    "research_agent_relative_rank": "Research Agent",
    "OSCAR_relative_rank": "OSCAR",
}

task_type_order = ["challenge", "method"]#, "research_question"]
task_type_labels = {"method": "Method", "challenge": "Challenge", "research_question": "Research Question"}

# Compute mean and SEM per model per task_type
grouped = df.groupby("task_type")

task_type_colors = {"challenge": COLOR_CHALLENGE, "method": COLOR_METHOD}

# Sort models by descending mean performance on the challenge task
challenge_group = grouped.get_group("method")
challenge_means = {col: challenge_group[col].dropna().mean() for col in model_names}
model_order = sorted(challenge_means, key=challenge_means.get, reverse=True)
model_labels = [model_names[col] for col in model_order]

fig, ax = plt.subplots(figsize=(8, 5))

for tt in task_type_order:
    means = []
    sems = []
    for col in model_order:
        group = grouped.get_group(tt)[col].dropna()
        means.append(group.mean())
        sems.append(group.sem())
    ax.errorbar(
        model_labels,
        means,
        yerr=sems,
        marker="o",
        label=task_type_labels[tt],
        color=task_type_colors[tt],
        linewidth=2,
        capsize=3,
    )

# ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Percentile Rank", fontsize=15)
ax.set_yticklabels([f"{int(t)}%" for t in ax.get_yticks()], fontsize=12)
ax.set_xticks(range(len(model_order)))
ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=15)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc="upper right", fontsize=15)
# ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "human_study.svg", bbox_inches="tight")
plt.show()