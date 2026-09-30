import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import kendalltau, spearmanr
import krippendorff

from oscar.plots.barplot import COLOR_CHALLENGE, COLOR_METHOD, REPO_ROOT

df = pd.read_csv(REPO_ROOT / "data" / "combined_survey.csv")

RANK_COLS = [c for c in df.columns if c.endswith("_rank")]
TASK_TYPES = ["challenge", "method", "research_question"]


def kendalls_w(rankings: np.ndarray) -> float:
    """Kendall's W (coefficient of concordance) for m judges ranking n items.

    Parameters
    ----------
    rankings : np.ndarray, shape (m, n)
        Each row is one judge's ranking of the n items.
    """
    m, n = rankings.shape
    if m < 2 or n < 2:
        return np.nan
    rank_sums = rankings.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    ss = np.sum((rank_sums - mean_rank_sum) ** 2)
    w = 12 * ss / (m**2 * (n**3 - n))
    return w


def compute_pairwise_tau(rankings: np.ndarray) -> list[float]:
    """Kendall's tau for every pair of judges."""
    taus = []
    for i, j in combinations(range(len(rankings)), 2):
        tau, _ = kendalltau(rankings[i], rankings[j])
        if not np.isnan(tau):
            taus.append(tau)
    return taus


def compute_pairwise_spearman(group: pd.DataFrame) -> list[float]:
    """Spearman's rho for every pair of judges, using only agents both ranked."""
    from scipy.stats import rankdata

    rows = group[RANK_COLS]
    indices = list(range(len(rows)))
    rhos = []
    for i, j in combinations(indices, 2):
        r_i = rows.iloc[i]
        r_j = rows.iloc[j]
        # Keep only agents that both annotators ranked
        mask = r_i.notna() & r_j.notna()
        if mask.sum() < 3:
            continue
        # Re-rank within the shared agents
        a = rankdata(r_i[mask].values, method="average")
        b = rankdata(r_j[mask].values, method="average")
        rho, _ = spearmanr(a, b)
        if not np.isnan(rho):
            rhos.append(rho)
    return rhos


def get_rankings_matrix(group: pd.DataFrame) -> np.ndarray | None:
    """Extract a (n_annotators, n_agents) matrix from a group, dropping agents
    with any missing ranks. Re-ranks within each row so values are 1..n."""
    ranks = group[RANK_COLS].dropna(axis=1)  # drop agents not ranked by all
    if ranks.shape[1] < 2:
        return None
    # Re-rank each row to get contiguous ranks 1..n_agents
    from scipy.stats import rankdata
    mat = ranks.values.astype(float)
    for i in range(len(mat)):
        mat[i] = rankdata(mat[i], method="average")
    return mat


def compute_iaa_per_item(df: pd.DataFrame):
    """Compute IAA metrics for each (paper, task_type) with >=2 annotators."""
    results = []
    for (paper, task), group in df.groupby(["target_paper_id", "task_type"]):
        if group["anonymised_user_id"].nunique() < 2:
            continue
        mat = get_rankings_matrix(group)
        if mat is None:
            continue

        w = kendalls_w(mat)
        taus = compute_pairwise_tau(mat)
        rhos = compute_pairwise_spearman(group)

        results.append({
            "paper": paper,
            "task_type": task,
            "n_annotators": len(mat),
            "n_agents": mat.shape[1],
            "kendalls_w": w,
            "mean_tau": np.mean(taus) if taus else np.nan,
            "mean_spearman": np.mean(rhos) if rhos else np.nan,
        })
    return pd.DataFrame(results)


def compute_krippendorff_alpha(df: pd.DataFrame, task_type: str | None = None) -> float:
    """Compute Krippendorff's alpha (ordinal) across all items for a task type.

    Each 'unit' is an (agent, paper) pair; each coder is an annotator.
    """
    subset = df if task_type is None else df[df["task_type"] == task_type]

    # Only keep (paper, task_type) groups with >=2 annotators
    multi = subset.groupby(["target_paper_id", "task_type"]).filter(
        lambda x: x["anonymised_user_id"].nunique() >= 2
    )
    if multi.empty:
        return np.nan

    # Build reliability data matrix: rows = coders, columns = units
    # Each unit is (paper, task_type, agent)
    annotators = sorted(multi["anonymised_user_id"].unique())
    ann_to_idx = {a: i for i, a in enumerate(annotators)}

    units = []
    for (paper, task), group in multi.groupby(["target_paper_id", "task_type"]):
        # Find agents ranked by all annotators in this group
        available_cols = group[RANK_COLS].dropna(axis=1).columns
        for col in available_cols:
            units.append((paper, task, col))

    # Build matrix: (n_annotators, n_units)
    rel_data = np.full((len(annotators), len(units)), np.nan)
    for unit_idx, (paper, task, col) in enumerate(units):
        rows = multi[(multi["target_paper_id"] == paper) & (multi["task_type"] == task)]
        for _, row in rows.iterrows():
            ann_idx = ann_to_idx[row["anonymised_user_id"]]
            val = row[col]
            if not pd.isna(val):
                rel_data[ann_idx, unit_idx] = val

    return krippendorff.alpha(reliability_data=rel_data, level_of_measurement="ordinal")


if __name__ == "__main__":
    iaa = compute_iaa_per_item(df)

    print("=" * 70)
    print("INTER-ANNOTATOR AGREEMENT SCORES")
    print("=" * 70)

    # Overall summary
    print("\n--- Overall (all task types) ---")
    print(f"  Items with >=2 annotators: {len(iaa)}")
    print(f"  Mean Kendall's W:          {iaa['kendalls_w'].mean():.3f} (±{iaa['kendalls_w'].std():.3f})")
    print(f"  Mean Kendall's tau:         {iaa['mean_tau'].mean():.3f} (±{iaa['mean_tau'].std():.3f})")
    print(f"  Mean Spearman's rho:        {iaa['mean_spearman'].mean():.3f} (±{iaa['mean_spearman'].std():.3f})")
    print(f"  Krippendorff's alpha:       {compute_krippendorff_alpha(df):.3f}")

    # Per task type
    for task in TASK_TYPES:
        subset = iaa[iaa["task_type"] == task]
        if subset.empty:
            continue
        print(f"\n--- {task} ---")
        print(f"  Items with >=2 annotators: {len(subset)}")
        print(f"  Mean Kendall's W:          {subset['kendalls_w'].mean():.3f} (±{subset['kendalls_w'].std():.3f})")
        print(f"  Mean Kendall's tau:         {subset['mean_tau'].mean():.3f} (±{subset['mean_tau'].std():.3f})")
        print(f"  Mean Spearman's rho:        {subset['mean_spearman'].mean():.3f} (±{subset['mean_spearman'].std():.3f})")
        print(f"  Krippendorff's alpha:       {compute_krippendorff_alpha(df, task):.3f}")

    print()
