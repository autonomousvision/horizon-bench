import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from horizon_bench.Config import TargetType, TASK_TYPE_ROOT, TARGET_TYPE, DATASET
from horizon_bench.evaluation.cosine_distance import get_cosine_similarity
from horizon_bench.arxiv_get_full_text import ArxivFullTextFetchError
from horizon_bench.pipeline.D_extract_targets import get_input_ids, get_gt, get_predictions, _resolve_dataset


def _dataset_title() -> str:
    """Human-readable name of the active dataset, for plot/table titles."""
    return DATASET.upper()


_ALL_AGENT_COLORS = {
    'ai_researcher': '#1f77b4',
    'ai_scientist': '#ff7f0e',
    'ai_scientist_v2': '#CC79A7',
    'coi_agent': '#E69F00',
    'autods': '#9467bd',
    'research_agent': '#D55E00',
    'naive_baseline': '#d62728',
    'oscar': '#17becf',
    'oscar_v2': '#bcbd22',
    'oscar_v3': '#e377c2',
    'oscar_v4': '#7f7f7f',
    'oscar_v5': '#aec7e8',
    'oscar_v6': '#0072B2',
}


# Display aliases for matplotlib plots
AGENT_ALIASES = {
    'oscar': r'OSCAR$_{\mathrm{LLM}}$',
    'oscar_v2': r'OSCAR$_{\mathrm{random}}$',
    'oscar_v3': r'OSCAR$_{\mathrm{gt\ insights}}$',
    'oscar_v4': r'OSCAR$_{\mathrm{cheating\ reranker}}$',
    'oscar_v5': r'OSCAR$_{\mathrm{no\ reranker}}$',
    'oscar_v6': r'OSCAR$_{\mathrm{learned\ reranker}}$',
}

# LaTeX aliases for tables
AGENT_LATEX_ALIASES = {
    'oscar': r'OSCAR\textsubscript{LLM}',
    'oscar_v2': r'OSCAR\textsubscript{random}',
    'oscar_v3': r'OSCAR\textsubscript{gt insights}',
    'oscar_v4': r'OSCAR\textsubscript{cheating reranker}',
    'oscar_v5': r'OSCAR\textsubscript{no reranker}',
    'oscar_v6': r'OSCAR\textsubscript{learned reranker}',
}


def _get_agent_label(agent_name: str, latex: bool = False) -> str:
    aliases = AGENT_LATEX_ALIASES if latex else AGENT_ALIASES
    if agent_name == 'coi_agent':
        agent_name = 'Chain of Ideas'
    return aliases.get(agent_name, agent_name.replace('_', ' ').title())


def _to_internal_key(agent_name: str) -> str:
    return 'coi_agent' if agent_name == 'chain_of_ideas' else agent_name


def _get_active_agent_colors() -> dict[str, str]:
    agents_env = os.environ.get("AGENTS_TO_RUN")
    if not agents_env:
        return _ALL_AGENT_COLORS
    active_keys = {_to_internal_key(a) for a in agents_env.split(",")}
    return {k: v for k, v in _ALL_AGENT_COLORS.items() if k in active_keys}


AGENT_COLORS = _get_active_agent_colors()


def compute_cosine_similarities(target_type: TargetType) -> dict[str, list[tuple[str, float]]]:
    input_ids = get_input_ids()
    cosine_similarities: dict[str, list[tuple[str, float]]] = {
        agent: [] for agent in AGENT_COLORS
    }

    for input_id in tqdm(input_ids, desc='Computing cosine similarities'):
        try:
            gt = get_gt(input_id, target_type=target_type)
        except ArxivFullTextFetchError as e:
            print(f"Skipping arXiv ID {input_id} due to error: {e}")
            continue

        raw_predictions = get_predictions(input_id, target_type=target_type)

        for agent_name, predictions in list(raw_predictions):
            if agent_name not in AGENT_COLORS:
                continue
            for prediction in predictions:
                if target_type == TargetType.CHALLENGE:
                    text = prediction.challenge
                elif target_type == TargetType.METHOD:
                    text = prediction.method
                elif target_type == TargetType.RESEARCH_QUESTION:
                    text = prediction.research_question
                sim = get_cosine_similarity(gt, text)
                cosine_similarities[agent_name].append((input_id, sim))

    return cosine_similarities


def save_cosine_similarities(cosine_similarities: dict[str, list[tuple[str, float]]], target_type: TargetType):
    rows = []
    for agent_name, similarities in cosine_similarities.items():
        for input_id, sim in similarities:
            rows.append({'agent': agent_name, 'input_id': input_id, 'cosine_similarity': sim})
    df = pd.DataFrame(rows)
    out_path = os.path.join(TASK_TYPE_ROOT, f'cosine_similarities_{target_type.value}.parquet')
    df.to_parquet(out_path, index=False)
    print(f"Cosine similarities saved to {out_path}")


def compute_soft_recall(cosine_similarities: dict[str, list[tuple[str, float]]]) -> dict[str, tuple[float, float]]:
    """Compute soft recall: average per-sample maximum semantic similarity to the target.

    Returns a dict mapping agent name to (mean, std) of per-sample max similarities.
    """
    soft_recall = {}
    for agent_name, similarities in cosine_similarities.items():
        if not similarities:
            continue
        # Group similarities by input_id
        per_sample: dict[str, list[float]] = {}
        for input_id, sim in similarities:
            per_sample.setdefault(input_id, []).append(sim)
        # Max similarity per sample, then average across samples
        max_sims = [max(sims) for sims in per_sample.values()]
        soft_recall[agent_name] = (np.mean(max_sims).item(), np.std(max_sims).item())
    return soft_recall


def compute_mean_similarity(cosine_similarities: dict[str, list[tuple[str, float]]]) -> dict[str, tuple[float, float]]:
    """Compute mean similarity: average per-sample mean semantic similarity to the target.

    Returns a dict mapping agent name to (mean, std) of per-sample mean similarities.
    """
    mean_sim = {}
    for agent_name, similarities in cosine_similarities.items():
        if not similarities:
            continue
        per_sample: dict[str, list[float]] = {}
        for input_id, sim in similarities:
            per_sample.setdefault(input_id, []).append(sim)
        mean_sims = [np.mean(sims).item() for sims in per_sample.values()]
        mean_sim[agent_name] = (np.mean(mean_sims).item(), np.std(mean_sims).item())
    return mean_sim


def generate_soft_recall_latex_table(soft_recall: dict[str, tuple[float, float]], mean_sim: dict[str, tuple[float, float]], target_type: TargetType):
    sorted_agents = sorted(soft_recall.items(), key=lambda x: x[1][0], reverse=True)
    best_recall = sorted_agents[0][1][0] if sorted_agents else None
    best_mean = max((v[0] for v in mean_sim.values()), default=None)
    task_name = f"{_dataset_title()} -- {target_type.name.replace('_', ' ').title()}"

    rows = []
    for agent, (recall_mean, recall_std) in sorted_agents:
        label = _get_agent_label(agent, latex=True)
        recall_fmt = f'{recall_mean:.3f} $\\pm$ {2 * recall_std:.3f}'
        if recall_mean == best_recall:
            recall_fmt = r'\textbf{' + recall_fmt + '}'
        m_mean, m_std = mean_sim.get(agent, (0, 0))
        mean_fmt = f'{m_mean:.3f} $\\pm$ {2 * m_std:.3f}'
        if m_mean == best_mean:
            mean_fmt = r'\textbf{' + mean_fmt + '}'
        rows.append(f'        {label} & {recall_fmt} & {mean_fmt} \\\\')

    table_rows = '\n'.join(rows)

    latex_src = rf"""\documentclass[border=10pt]{{standalone}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{caption}}
\begin{{document}}
\begin{{minipage}}{{\linewidth}}
\captionof{{table}}{{Task: {task_name}}}
\centering
\begin{{tabular}}{{l r r}}
    \toprule
    \textbf{{Agent}} & \textbf{{Soft Recall}} & \textbf{{Mean Similarity}} \\
    \midrule
{table_rows}
    \bottomrule
\end{{tabular}}
\end{{minipage}}
\end{{document}}
"""

    out_pdf = os.path.join(TASK_TYPE_ROOT, f'soft_recall_{target_type.value}.pdf')
    out_tex = os.path.join(TASK_TYPE_ROOT, f'soft_recall_{target_type.value}.tex')

    with open(out_tex, 'w') as f:
        f.write(latex_src)
    print(f"Soft recall LaTeX saved to {out_tex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, 'table.tex')
        with open(tex_path, 'w') as f:
            f.write(latex_src)

        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-output-directory', tmpdir, tex_path],
            capture_output=True, check=True,
        )

        pdf_path = os.path.join(tmpdir, 'table.pdf')
        os.replace(pdf_path, out_pdf)

    print(f"Soft recall LaTeX table saved to {out_pdf}")


def generate_soft_recall_color_table(soft_recall: dict[str, tuple[float, float]], mean_sim: dict[str, tuple[float, float]], target_type: TargetType):
    sorted_agents = sorted(soft_recall.items(), key=lambda x: x[1][0], reverse=True)
    task_name = f"{_dataset_title()} -- {target_type.name.replace('_', ' ').title()}"
    cell_text = [
        [
            _get_agent_label(agent),
            f'{recall_mean:.3f} \u00b1 {2 * recall_std:.3f}',
            f'{mean_sim.get(agent, (0, 0))[0]:.3f} \u00b1 {2 * mean_sim.get(agent, (0, 0))[1]:.3f}',
        ]
        for agent, (recall_mean, recall_std) in sorted_agents
    ]
    cell_colors = [
        [AGENT_COLORS.get(agent, '#cccccc') + '33'] * 3
        for agent, _ in sorted_agents
    ]

    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.4 * len(cell_text)))
    ax.axis('off')
    table = ax.table(
        cellText=cell_text,
        colLabels=['Agent', 'Soft Recall (\u00b12\u03c3)', 'Mean Similarity (\u00b12\u03c3)'],
        cellColours=cell_colors,
        colColours=['#e0e0e0', '#e0e0e0', '#e0e0e0'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
    ax.set_title(
        f'Task: {task_name}',
        fontsize=14, fontweight='bold', pad=12,
    )
    fig.tight_layout()

    out_path = os.path.join(TASK_TYPE_ROOT, f'soft_recall_{target_type.value}_color.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Soft recall color table saved to {out_path}")


def generate_histogram(cosine_similarities: dict[str, list[tuple[str, float]]], target_type: TargetType):
    sns.set_theme(style="white", palette="muted")
    plt.figure(figsize=(10, 4))

    for agent_name, similarities in cosine_similarities.items():
        sims = [sim for _, sim in similarities]
        if not sims:
            continue
        sns.kdeplot(
            sims,
            label=_get_agent_label(agent_name),
            color=AGENT_COLORS[agent_name],
            linewidth=1.5,
            fill=True,
            alpha=0.2,
        )

    plt.title(
        f'{_dataset_title()} — {target_type.name.title()}',
        fontsize=24, fontweight='bold', pad=20,
    )
    plt.xlim(0.5, 1.0)
    plt.xlabel('Cosine Similarity', fontsize=17, labelpad=13)
    plt.ylabel('Percentage (%)', fontsize=17, labelpad=13)
    sns.despine()
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], frameon=False, fontsize=15)
    plt.tight_layout()

    out_path = os.path.join(TASK_TYPE_ROOT, f'histogram_{target_type.value}.pdf')
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Histogram saved to {out_path}")


def generate_histogram_pipeline(target_type: TargetType):
    cosine_similarities = compute_cosine_similarities(target_type)
    save_cosine_similarities(cosine_similarities, target_type)
    soft_recall = compute_soft_recall(cosine_similarities)
    mean_sim = compute_mean_similarity(cosine_similarities)
    print(f"\nSoft Recall ({target_type.name}):")
    for agent_name, (mean, std) in sorted(soft_recall.items(), key=lambda x: x[1][0], reverse=True):
        print(f"  {agent_name:25s} {mean:.3f} \u00b1 {2 * std:.3f}")
    print(f"\nMean Similarity ({target_type.name}):")
    for agent_name, (mean, std) in sorted(mean_sim.items(), key=lambda x: x[1][0], reverse=True):
        print(f"  {agent_name:25s} {mean:.3f} \u00b1 {2 * std:.3f}")
    generate_soft_recall_latex_table(soft_recall, mean_sim, target_type)
    generate_soft_recall_color_table(soft_recall, mean_sim, target_type)
    generate_histogram(cosine_similarities, target_type)


if __name__ == "__main__":
    dataset = _resolve_dataset(DATASET)
    print(f"Generating histogram for dataset={dataset!r}, target_type={TARGET_TYPE.value!r}")
    generate_histogram_pipeline(TARGET_TYPE)
