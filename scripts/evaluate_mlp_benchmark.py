"""Demšar Friedman+Nemenyi ranking and per-dataset Welch winner tests.

Reads ``Results/all_mlp_cv_summaries.csv`` (dataset / features / model /
n_splits / metric mean+std), ranks methods across datasets, runs the Friedman
omnibus test with Nemenyi critical difference, and compares the top two
methods per dataset with Welch's t-test from summary statistics.

Example:

    python scripts/evaluate_mlp_benchmark.py
"""
from __future__ import annotations

import argparse
import string
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "Results" / "all_mlp_cv_summaries.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Results" / "nnFoundation"

# Studentized range critical values q_alpha for alpha=0.05 (Demšar 2006)
Q_ALPHA_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Friedman+Nemenyi overall ranking and per-dataset Welch "
            "best-vs-runner-up tests on MLP/WORC CV summary CSV."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Input summary CSV (default: Results/all_mlp_cv_summaries.csv).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="AUC",
        help="Primary metric prefix (default: AUC → AUC_mean / AUC_std).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for Welch verdict and Nemenyi CD (default: 0.05).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rank and per-dataset CSVs.",
    )
    return parser.parse_args()


def nemenyi_groups(mean_ranks: pd.Series, cd: float) -> dict[str, str]:
    """Assign compact letter groups for Nemenyi CD cliques.

    Two methods share a letter when their mean-rank difference is <= CD.
    Overlapping contiguous cliques get concatenated letters (e.g. AB).
    """
    ordered = mean_ranks.sort_values()
    methods = list(ordered.index)
    ranks = ordered.to_numpy(dtype=float)
    k = len(methods)
    if k == 0:
        return {}

    # Contiguous intervals [i, j] with ranks[j] - ranks[i] <= CD
    intervals: list[tuple[int, int]] = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] <= cd + 1e-12:
            j += 1
        intervals.append((i, j))

    # Keep maximal intervals only (not strictly contained in another)
    maximal: list[tuple[int, int]] = []
    for i, j in intervals:
        if any(i2 <= i and j2 >= j and (i2, j2) != (i, j) for i2, j2 in intervals):
            continue
        maximal.append((i, j))
    # Deduplicate while preserving order
    seen: set[tuple[int, int]] = set()
    unique_maximal: list[tuple[int, int]] = []
    for pair in maximal:
        if pair not in seen:
            seen.add(pair)
            unique_maximal.append(pair)

    letters = string.ascii_uppercase
    membership: dict[str, list[str]] = {m: [] for m in methods}
    for c_idx, (i, j) in enumerate(unique_maximal):
        letter = letters[c_idx] if c_idx < len(letters) else f"G{c_idx}"
        for idx in range(i, j + 1):
            membership[methods[idx]].append(letter)

    return {m: "".join(membership[m]) or "?" for m in methods}


def evaluate_benchmark(
    csv_path: Path,
    metric: str = "AUC",
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    required = {"dataset", "features", "model", "n_splits"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")

    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
    for col in (mean_col, std_col):
        if col not in df.columns:
            raise SystemExit(f"CSV missing metric column: {col}")

    df = df.copy()
    df["method"] = df["features"].astype(str) + " + " + df["model"].astype(str)

    pivot = df.pivot(index="dataset", columns="method", values=mean_col)
    if pivot.isna().any().any():
        bad = pivot.columns[pivot.isna().any()].tolist()
        raise SystemExit(
            f"Incomplete method coverage across datasets (NaNs in {bad})."
        )

    ranks = pivot.rank(ascending=False, axis=1, method="average")
    mean_ranks = ranks.mean().sort_values()

    friedman_stat, friedman_p = stats.friedmanchisquare(
        *(pivot[c].to_numpy(dtype=float) for c in pivot.columns)
    )
    k, n = len(pivot.columns), len(pivot)
    q_alpha = Q_ALPHA_05.get(k, Q_ALPHA_05[5])
    cd = float(q_alpha * np.sqrt((k * (k + 1)) / (6 * n)))

    groups = nemenyi_groups(mean_ranks, cd)
    ranks_table = pd.DataFrame(
        {
            "method": mean_ranks.index,
            "mean_rank": mean_ranks.values,
            "nemenyi_group": [groups[m] for m in mean_ranks.index],
            "friedman_stat": friedman_stat,
            "friedman_p": friedman_p,
            "nemenyi_cd": cd,
            "n_datasets": n,
            "n_methods": k,
            "alpha": alpha,
        }
    )

    dataset_order = list(dict.fromkeys(df["dataset"].tolist()))
    rows: list[dict[str, object]] = []
    for dataset in dataset_order:
        sub = (
            df[df["dataset"] == dataset]
            .sort_values(by=mean_col, ascending=False)
            .reset_index(drop=True)
        )
        if len(sub) < 2:
            raise SystemExit(f"Need >=2 methods for dataset {dataset}")

        r1, r2 = sub.iloc[0], sub.iloc[1]
        m1, s1, n1 = float(r1[mean_col]), float(r1[std_col]), float(r1["n_splits"])
        m2, s2, n2 = float(r2[mean_col]), float(r2[std_col]), float(r2["n_splits"])

        se = np.sqrt(s1**2 / n1 + s2**2 / n2)
        t_stat = (m1 - m2) / se
        df_val = (s1**2 / n1 + s2**2 / n2) ** 2 / (
            ((s1**2 / n1) ** 2 / (n1 - 1)) + ((s2**2 / n2) ** 2 / (n2 - 1))
        )
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df_val)))

        rows.append(
            {
                "Dataset": dataset,
                "Top_Method": r1["method"],
                "Top_Mean_SD": f"{m1:.3f} ± {s1:.3f}",
                "Runner_Up": r2["method"],
                "Runner_Up_Mean_SD": f"{m2:.3f} ± {s2:.3f}",
                "Diff": round(m1 - m2, 4),
                "t_stat": round(float(t_stat), 4),
                "df": round(float(df_val), 2),
                "p_value": round(p_val, 4),
                "Verdict": (
                    "Decisive Winner" if p_val < alpha else "Statistical Tie"
                ),
            }
        )

    omnibus = {
        "friedman_stat": float(friedman_stat),
        "friedman_p": float(friedman_p),
        "nemenyi_cd": cd,
        "n_datasets": float(n),
        "n_methods": float(k),
    }
    return ranks_table, omnibus, pd.DataFrame(rows)


def print_summary(
    ranks_table: pd.DataFrame,
    omnibus: dict[str, float],
    per_dataset: pd.DataFrame,
) -> None:
    print("\n=== Overall Ranking (mean Friedman rank, lower is better) ===")
    for _, row in ranks_table.iterrows():
        print(
            f"  {row['mean_rank']:.3f}  [{row['nemenyi_group']}]  {row['method']}"
        )
    print(
        f"\nFriedman chi^2={omnibus['friedman_stat']:.4f}, "
        f"p={omnibus['friedman_p']:.4g}, "
        f"Nemenyi CD={omnibus['nemenyi_cd']:.4f} "
        f"(N={int(omnibus['n_datasets'])}, K={int(omnibus['n_methods'])})"
    )
    if omnibus["friedman_p"] >= 0.05:
        print("  Friedman not significant at alpha=0.05; CD groups are descriptive.")

    print("\n=== Per-Dataset Winner Verification (Welch) ===")
    for _, row in per_dataset.iterrows():
        print(
            f"  {row['Dataset']:9s}  {row['Top_Method']:20s} "
            f"{row['Top_Mean_SD']}  vs  {row['Runner_Up']:20s} "
            f"{row['Runner_Up_Mean_SD']}  "
            f"diff={row['Diff']:+.4f}  p={row['p_value']:.4f}  -> {row['Verdict']}"
        )


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    output_dir = args.output_dir.resolve()

    if not csv_path.is_file():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    ranks_table, omnibus, per_dataset = evaluate_benchmark(
        csv_path, metric=args.metric, alpha=args.alpha
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ranks_path = output_dir / "mlp_benchmark_mean_ranks.csv"
    per_path = output_dir / "mlp_benchmark_per_dataset.csv"
    ranks_table.to_csv(ranks_path, index=False)
    per_dataset.to_csv(per_path, index=False)

    print(f"Wrote {ranks_path}")
    print(f"Wrote {per_path}")
    print_summary(ranks_table, omnibus, per_dataset)


if __name__ == "__main__":
    main()
