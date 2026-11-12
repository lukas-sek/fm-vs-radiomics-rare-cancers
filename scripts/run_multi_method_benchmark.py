"""Multi-method AUROC benchmark across WORC cohorts.

Discovers nnFoundation MLP patient-prediction CSVs and WORC past-performance
JSONs, computes patient-level AUROC with stratified bootstrap 95% CIs, then
runs Iman-Davenport Friedman + Holm-adjusted Wilcoxon tests vs WORC.

Example:

    python scripts/run_multi_method_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "Results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Results" / "benchmark_summary"
DEFAULT_MLP_DIR = DEFAULT_RESULTS_DIR / "nnFoundation" / "MLP_Results"
DEFAULT_WORC_DIR = DEFAULT_RESULTS_DIR / "WORC" / "pastperformance"

DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")
DATASET_SET = set(DATASETS)

CONTROL_METHOD = "AutoML CASH (WORC)"
METHOD_ORDER = (
    CONTROL_METHOD,
    "CNN + mlp",
    "CNN + mlp_pca",
    "ViT + mlp",
    "ViT + mlp_pca",
)

PRED_NAME_RE = re.compile(
    r"^(?P<cohort>Lipo|Desmoid|Liver|GIST|CRLM|Melanoma)_"
    r"(?P<stem>nncnn|nnvit)_"
    r"(?P<model>mlp_pca|mlp)_cv_patient_predictions\.csv$"
)
STEM_MODEL_TO_METHOD = {
    ("nncnn", "mlp"): "CNN + mlp",
    ("nncnn", "mlp_pca"): "CNN + mlp_pca",
    ("nnvit", "mlp"): "ViT + mlp",
    ("nnvit", "mlp_pca"): "ViT + mlp_pca",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-method benchmark analysis across all cohorts."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root Results directory (default: Results/).",
    )
    parser.add_argument(
        "--mlp-dir",
        type=Path,
        default=None,
        help="MLP patient-prediction directory (default: Results/nnFoundation/MLP_Results).",
    )
    parser.add_argument(
        "--worc-dir",
        type=Path,
        default=None,
        help="WORC past-performance directory (default: Results/WORC/pastperformance).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for tables and plots.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for 95%% CIs.",
    )
    return parser.parse_args()


def parse_worc_json(json_path: Path) -> dict[str, float | int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    auc_str = data["Statistics"]["AUC 95%:"]
    mean_part, ci_part = auc_str.split("(")
    mean_auc = float(mean_part.strip())
    low_str, high_str = ci_part.replace(")", "").split(",")
    percentages = data["Rankings"]["Percentages"]
    return {
        "n_patients": len(percentages),
        "auc": mean_auc,
        "ci_low": float(low_str.strip()),
        "ci_high": float(high_str.strip()),
    }


def evaluate_patient_predictions(
    csv_path: Path, n_bootstraps: int = 2000, seed: int = 42
) -> dict[str, float | int]:
    df = pd.read_csv(csv_path)
    required = {"patient_id", "label", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")

    agg = (
        df.groupby("patient_id")
        .agg(label=("label", "first"), probability=("probability", "mean"))
        .reset_index()
    )
    y_true = agg["label"].to_numpy()
    y_prob = agg["probability"].to_numpy()
    auc = float(roc_auc_score(y_true, y_prob))

    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError(f"{csv_path.name}: need both classes for AUROC/bootstrap")

    boot_aucs: list[float] = []
    for _ in range(n_bootstraps):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sample_idx = np.concatenate([b_pos, b_neg])
        boot_aucs.append(float(roc_auc_score(y_true[sample_idx], y_prob[sample_idx])))

    return {
        "n_patients": int(len(agg)),
        "auc": auc,
        "ci_low": float(np.percentile(boot_aucs, 2.5)),
        "ci_high": float(np.percentile(boot_aucs, 97.5)),
    }


def discover_and_collect_results(
    mlp_dir: Path,
    worc_dir: Path,
    n_bootstraps: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    if not worc_dir.is_dir():
        raise SystemExit(f"WORC directory not found: {worc_dir}")
    for json_file in sorted(worc_dir.glob("WORCDatabase_performance_*_WORC371.json")):
        cohort = json_file.name.removeprefix("WORCDatabase_performance_").removesuffix(
            "_WORC371.json"
        )
        if cohort not in DATASET_SET:
            continue
        res = parse_worc_json(json_file)
        records.append(
            {
                "cohort": cohort,
                "method": CONTROL_METHOD,
                "is_baseline": True,
                "source_file": json_file.name,
                **res,
            }
        )

    if not mlp_dir.is_dir():
        raise SystemExit(f"MLP directory not found: {mlp_dir}")
    matched = 0
    for csv_file in sorted(mlp_dir.glob("*_cv_patient_predictions.csv")):
        match = PRED_NAME_RE.match(csv_file.name)
        if match is None:
            continue
        cohort = match.group("cohort")
        method = STEM_MODEL_TO_METHOD[(match.group("stem"), match.group("model"))]
        print(f"  Evaluating {csv_file.name} ...")
        res = evaluate_patient_predictions(csv_file, n_bootstraps=n_bootstraps)
        records.append(
            {
                "cohort": cohort,
                "method": method,
                "is_baseline": False,
                "source_file": csv_file.name,
                **res,
            }
        )
        matched += 1

    if matched == 0:
        raise SystemExit(f"No matching *_cv_patient_predictions.csv in {mlp_dir}")
    return records


def assert_complete_grid(summary_df: pd.DataFrame) -> None:
    pivot = summary_df.pivot(index="cohort", columns="method", values="auc")
    expected_methods = list(METHOD_ORDER)
    missing_methods = [m for m in expected_methods if m not in pivot.columns]
    if missing_methods:
        raise SystemExit(f"Missing methods in results: {missing_methods}")
    missing_cohorts = [c for c in DATASETS if c not in pivot.index]
    if missing_cohorts:
        raise SystemExit(f"Missing cohorts in results: {missing_cohorts}")
    incomplete = pivot.loc[list(DATASETS), expected_methods]
    if incomplete.isna().any().any():
        bad = [
            f"{cohort} / {method}"
            for cohort in DATASETS
            for method in expected_methods
            if pd.isna(incomplete.loc[cohort, method])
        ]
        raise SystemExit(f"Incomplete method x cohort grid: {bad}")


def compute_cross_cohort_statistics(
    summary_df: pd.DataFrame,
) -> tuple[pd.Series, float, float, dict[str, float]]:
    pivot = summary_df.pivot(index="cohort", columns="method", values="auc")
    pivot = pivot.loc[list(DATASETS), list(METHOD_ORDER)]

    if CONTROL_METHOD not in pivot.columns:
        raise ValueError(f"Control baseline '{CONTROL_METHOD}' missing from results.")

    ranks = stats.rankdata(-pivot.to_numpy(), axis=1)
    rank_df = pd.DataFrame(ranks, index=pivot.index, columns=pivot.columns)
    avg_ranks = rank_df.mean(axis=0)

    k = pivot.shape[1]
    n = pivot.shape[0]
    friedman_stat, _ = stats.friedmanchisquare(
        *[pivot[c].to_numpy(dtype=float) for c in pivot.columns]
    )

    denom = n * (k - 1) - friedman_stat
    if denom <= 0:
        f_stat = float("inf")
        p_friedman = 0.0
    else:
        f_stat = float(((n - 1) * friedman_stat) / denom)
        p_friedman = float(1.0 - stats.f.cdf(f_stat, k - 1, (k - 1) * (n - 1)))

    competitors = [m for m in METHOD_ORDER if m != CONTROL_METHOD]
    raw_pvals: list[float] = []
    for comp in competitors:
        diff = pivot[CONTROL_METHOD] - pivot[comp]
        res = stats.wilcoxon(diff.to_numpy(dtype=float), alternative="greater")
        raw_pvals.append(float(res.pvalue))

    sorted_order = np.argsort(raw_pvals)
    adj_pvals = np.empty(len(raw_pvals), dtype=float)
    m_tests = len(raw_pvals)
    for step, idx in enumerate(sorted_order):
        adj_pvals[idx] = min(1.0, raw_pvals[idx] * (m_tests - step))
    for i in range(1, m_tests):
        idx_curr = sorted_order[i]
        idx_prev = sorted_order[i - 1]
        adj_pvals[idx_curr] = max(adj_pvals[idx_curr], adj_pvals[idx_prev])

    holm_dict = {comp: float(adj_pvals[i]) for i, comp in enumerate(competitors)}
    return avg_ranks, f_stat, p_friedman, holm_dict


def format_p_value(p_val: float) -> str:
    if p_val < 0.0001:
        return f"{p_val:.1e}"
    return f"{p_val:.4f}"


def write_latex_table(
    summary_df: pd.DataFrame,
    avg_ranks: pd.Series,
    f_stat: float,
    p_friedman: float,
    holm_pvals: dict[str, float],
    out_path: Path,
) -> None:
    pivot_auc = summary_df.pivot(index="method", columns="cohort", values="auc")
    pivot_low = summary_df.pivot(index="method", columns="cohort", values="ci_low")
    pivot_high = summary_df.pivot(index="method", columns="cohort", values="ci_high")

    cohort_order = [c for c in DATASETS if c in pivot_auc.columns]
    method_order = [m for m in METHOD_ORDER if m in pivot_auc.index]

    caption = (
        r"AUROC [95\% CI] comparison across cohorts and methods. "
        r"Mean ranks are Friedman ranks (1 = best). "
        rf"Iman--Davenport omnibus $F={f_stat:.3f}$ "
        rf"($p={format_p_value(p_friedman)}$). "
        r"Holm-adjusted one-sided Wilcoxon $p$-values test whether "
        r"AutoML CASH (WORC) outperforms each competitor across cohorts."
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        r"\label{tab:all_methods_benchmark}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l "
        + "c" * len(cohort_order)
        + r" cc@{}}",
        r"\toprule",
        r"\textbf{Method} & "
        + " & ".join([f"\\textbf{{{c}}}" for c in cohort_order])
        + r" & \textbf{Mean Rank} & \textbf{Holm $p$ vs WORC} \\",
        r"\midrule",
    ]

    for method in method_order:
        cells = [method]
        for c in cohort_order:
            val = pivot_auc.loc[method, c]
            low = pivot_low.loc[method, c]
            high = pivot_high.loc[method, c]
            if pd.isna(val):
                cells.append("---")
            else:
                cells.append(f"{val:.2f} [{low:.2f}--{high:.2f}]")
        cells.append(f"{avg_ranks.get(method, np.nan):.1f}")
        if method == CONTROL_METHOD:
            cells.append("---")
        else:
            cells.append(format_p_value(holm_pvals[method]))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}"])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_multi_method_forest(summary_df: pd.DataFrame, out_path: Path) -> None:
    cohorts = [c for c in DATASETS if c in summary_df["cohort"].unique()]
    methods = [m for m in METHOD_ORDER if m in summary_df["method"].unique()]
    n_cohorts = len(cohorts)

    fig, axes = plt.subplots(
        1, n_cohorts, figsize=(3.2 * n_cohorts, 5.0), sharey=True, dpi=150
    )
    if n_cohorts == 1:
        axes = [axes]

    y_pos = np.arange(len(methods))[::-1]
    method_idx = {m: i for i, m in enumerate(reversed(methods))}

    for ax, cohort in zip(axes, cohorts):
        c_df = summary_df[summary_df["cohort"] == cohort]
        ax.axvline(0.50, color="0.4", linestyle="--", linewidth=0.9, zorder=0)

        worc_row = c_df[c_df["method"] == CONTROL_METHOD]
        if not worc_row.empty:
            w_low = float(worc_row["ci_low"].iloc[0])
            w_high = float(worc_row["ci_high"].iloc[0])
            ax.axvspan(w_low, w_high, color="tab:blue", alpha=0.12, zorder=1)

        for _, row in c_df.iterrows():
            m = str(row["method"])
            if m not in method_idx:
                continue
            y = method_idx[m]
            auc = float(row["auc"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            ax.errorbar(
                auc,
                y,
                xerr=[[auc - low], [high - auc]],
                fmt="o",
                color="0.15",
                ecolor="0.3",
                capsize=3,
                markersize=5,
                zorder=2,
            )

        ax.set_title(cohort, fontsize=11, fontweight="bold")
        ax.set_xlim(0.35, 1.0)
        ax.set_xlabel("AUROC (95% CI)")
        ax.grid(axis="x", linestyle=":", alpha=0.6)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(list(reversed(methods)))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    mlp_dir = (
        args.mlp_dir.resolve()
        if args.mlp_dir is not None
        else (results_dir / "nnFoundation" / "MLP_Results")
    )
    worc_dir = (
        args.worc_dir.resolve()
        if args.worc_dir is not None
        else (results_dir / "WORC" / "pastperformance")
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting results ...")
    records = discover_and_collect_results(mlp_dir, worc_dir, args.n_bootstraps)
    summary_df = pd.DataFrame(records)
    assert_complete_grid(summary_df)

    # Stable ordering for CSV
    summary_df["_c"] = summary_df["cohort"].map({c: i for i, c in enumerate(DATASETS)})
    summary_df["_m"] = summary_df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    summary_df = (
        summary_df.sort_values(["_c", "_m"], kind="mergesort")
        .drop(columns=["_c", "_m"])
        .reset_index(drop=True)
    )

    csv_out = output_dir / "benchmark_per_method_cohort.csv"
    summary_df.to_csv(csv_out, index=False)

    avg_ranks, f_stat, p_f, holm_pvals = compute_cross_cohort_statistics(summary_df)

    latex_out = output_dir / "benchmark_table.tex"
    write_latex_table(summary_df, avg_ranks, f_stat, p_f, holm_pvals, latex_out)

    plot_out = output_dir / "multi_method_forest_plot.png"
    plot_multi_method_forest(summary_df, plot_out)

    print(f"\nOmnibus Iman-Davenport F: {f_stat:.3f} (p = {p_f:.4e})")
    print("Mean ranks (1 = best):")
    for method in METHOD_ORDER:
        print(f"  {method}: {avg_ranks[method]:.2f}")
    print("Post-hoc Holm-adjusted p-values vs. WORC (one-sided greater):")
    for method, p_adj in holm_pvals.items():
        print(f"  {method}: p_adj = {p_adj:.4f}")
    print(f"\nWrote {csv_out}")
    print(f"Wrote {latex_out}")
    print(f"Wrote {plot_out}")


if __name__ == "__main__":
    main()
