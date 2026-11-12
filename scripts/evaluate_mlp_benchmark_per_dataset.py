"""Demšar Friedman+Nemenyi ranking within each dataset (CV splits as blocks).

Same analysis as ``evaluate_mlp_benchmark.py``, but instead of ranking methods
across datasets it ranks them across the CV splits of a single dataset. Per-split
AUCs come from ``Results/nnFoundation/MLP_Results/*_cv_per_split.csv`` and from
``All_performances["AUC"]`` in ``Results/WORC/pastperformance``. The MLPs were
trained on the first N WORC CV splits, so WORC iterations 0..N-1 are paired with
MLP splits 0..N-1.

Example:

    python scripts/evaluate_mlp_benchmark_per_dataset.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_mlp_benchmark import Q_ALPHA_05, nemenyi_groups  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MLP_DIR = REPO_ROOT / "Results" / "nnFoundation" / "MLP_Results"
DEFAULT_WORC_DIR = REPO_ROOT / "Results" / "WORC" / "pastperformance"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Results" / "nnFoundation"

DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")

PER_SPLIT_RE = re.compile(
    r"^(?P<dataset>Lipo|Desmoid|Liver|GIST|CRLM|Melanoma)_"
    r"(?P<stem>nncnn|nnvit)_"
    r"(?P<model>mlp_pca|mlp)_cv_per_split\.csv$"
)

METRIC_COLUMNS = {"AUC": ("auroc", "auc"), "AP": ("ap",)}
WORC_METRIC_KEYS = {"AUC": "AUC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Friedman+Nemenyi ranking of MLP/WORC methods within each dataset, "
            "using CV splits as blocks."
        )
    )
    parser.add_argument("--mlp-dir", type=Path, default=DEFAULT_MLP_DIR)
    parser.add_argument("--worc-dir", type=Path, default=DEFAULT_WORC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metric",
        type=str,
        default="AUC",
        choices=sorted(METRIC_COLUMNS),
        help="Per-split metric to rank (default: AUC).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for Friedman verdict and Nemenyi CD (default: 0.05).",
    )
    parser.add_argument(
        "--exclude-worc",
        action="store_true",
        help="Rank only the MLP methods (drop WORC radiomics).",
    )
    return parser.parse_args()


def load_mlp_per_split(mlp_dir: Path, metric: str) -> dict[str, pd.DataFrame]:
    """Return {dataset: DataFrame indexed by split, one column per method}."""
    if not mlp_dir.is_dir():
        raise SystemExit(f"MLP directory not found: {mlp_dir}")

    columns: dict[str, dict[str, pd.Series]] = {name: {} for name in DATASETS}
    for path in sorted(mlp_dir.glob("*_cv_per_split.csv")):
        match = PER_SPLIT_RE.match(path.name)
        if match is None:
            continue
        frame = pd.read_csv(path)
        value_col = next(
            (c for c in METRIC_COLUMNS[metric] if c in frame.columns), None
        )
        if value_col is None:
            raise SystemExit(f"{path.name} has no column for metric {metric}")
        features = "CNN" if match.group("stem") == "nncnn" else "ViT"
        method = f"{features} + {match.group('model')}"
        columns[match.group("dataset")][method] = frame.set_index("split")[
            value_col
        ].astype(float)

    return {
        dataset: pd.DataFrame(methods).sort_index()
        for dataset, methods in columns.items()
        if methods
    }


def load_worc_per_split(worc_dir: Path, metric: str) -> dict[str, np.ndarray]:
    if metric not in WORC_METRIC_KEYS:
        raise SystemExit(f"WORC past performance has no per-split {metric}")
    if not worc_dir.is_dir():
        raise SystemExit(f"WORC directory not found: {worc_dir}")

    scores: dict[str, np.ndarray] = {}
    for path in sorted(worc_dir.glob("WORCDatabase_performance_*_WORC371.json")):
        dataset = path.name.removeprefix("WORCDatabase_performance_").removesuffix(
            "_WORC371.json"
        )
        if dataset not in DATASETS:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = (payload.get("All_performances") or {}).get(WORC_METRIC_KEYS[metric])
        if not values:
            raise SystemExit(f"No All_performances['{metric}'] in {path.name}")
        scores[dataset] = np.asarray(values, dtype=float)
    return scores


def friedman_nemenyi(
    pivot: pd.DataFrame, alpha: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Rank columns (methods) over rows (blocks); higher score = better rank."""
    if pivot.isna().any().any():
        bad = pivot.columns[pivot.isna().any()].tolist()
        raise SystemExit(f"Incomplete split coverage (NaNs in {bad}).")

    ranks = pivot.rank(ascending=False, axis=1, method="average")
    mean_ranks = ranks.mean().sort_values()

    friedman_stat, friedman_p = stats.friedmanchisquare(
        *(pivot[c].to_numpy(dtype=float) for c in pivot.columns)
    )
    k, n = len(pivot.columns), len(pivot)
    q_alpha = Q_ALPHA_05.get(k, Q_ALPHA_05[5])
    cd = float(q_alpha * np.sqrt((k * (k + 1)) / (6 * n)))
    groups = nemenyi_groups(mean_ranks, cd)

    table = pd.DataFrame(
        {
            "method": mean_ranks.index,
            "mean_rank": mean_ranks.values,
            "nemenyi_group": [groups[m] for m in mean_ranks.index],
            "mean_score": pivot[mean_ranks.index].mean().values,
            "std_score": pivot[mean_ranks.index].std(ddof=1).values,
            "friedman_stat": friedman_stat,
            "friedman_p": friedman_p,
            "nemenyi_cd": cd,
            "n_splits": n,
            "n_methods": k,
            "alpha": alpha,
        }
    )
    omnibus = {
        "friedman_stat": float(friedman_stat),
        "friedman_p": float(friedman_p),
        "nemenyi_cd": cd,
        "n_splits": float(n),
        "n_methods": float(k),
    }
    return table, omnibus


def build_pivots(
    mlp: dict[str, pd.DataFrame],
    worc: dict[str, np.ndarray],
    exclude_worc: bool,
) -> dict[str, pd.DataFrame]:
    pivots: dict[str, pd.DataFrame] = {}
    for dataset in DATASETS:
        if dataset not in mlp:
            continue
        pivot = mlp[dataset].copy()
        if not exclude_worc:
            if dataset not in worc:
                raise SystemExit(f"No WORC past performance for {dataset}")
            splits = pivot.index.to_numpy(dtype=int)
            if splits.max() >= len(worc[dataset]):
                raise SystemExit(
                    f"{dataset}: WORC has {len(worc[dataset])} iterations, "
                    f"MLP uses split {splits.max()}"
                )
            pivot["WORC + radiomics"] = worc[dataset][splits]
        pivots[dataset] = pivot
    return pivots


def print_summary(dataset: str, table: pd.DataFrame, omnibus: dict[str, float], alpha: float) -> None:
    print(f"\n=== {dataset} (mean Friedman rank over splits, lower is better) ===")
    for _, row in table.iterrows():
        print(
            f"  {row['mean_rank']:.3f}  [{row['nemenyi_group']}]  "
            f"{row['method']:20s} {row['mean_score']:.3f} ± {row['std_score']:.3f}"
        )
    print(
        f"  Friedman chi^2={omnibus['friedman_stat']:.4f}, "
        f"p={omnibus['friedman_p']:.4g}, "
        f"Nemenyi CD={omnibus['nemenyi_cd']:.4f} "
        f"(N={int(omnibus['n_splits'])} splits, K={int(omnibus['n_methods'])})"
    )
    if omnibus["friedman_p"] >= alpha:
        print(f"  Friedman not significant at alpha={alpha}; CD groups are descriptive.")


LATEX_METHOD_ORDER = (
    ("CNN + mlp", "CNN + MLP"),
    ("CNN + mlp_pca", "CNN + MLP$_\\text{PCA}$"),
    ("ViT + mlp", "ViT + MLP"),
    ("ViT + mlp_pca", "ViT + MLP$_\\text{PCA}$"),
    ("WORC + radiomics", "WORC"),
)


def format_p(p: float) -> str:
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def build_latex_table(result: pd.DataFrame, metric: str, alpha: float) -> str:
    methods = [
        (key, label)
        for key, label in LATEX_METHOD_ORDER
        if key in set(result["method"])
    ]
    n_splits = int(result["n_splits"].iloc[0])
    lines = [
        "% Requires \\usepackage{booktabs,amsmath}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{l{'c' * len(methods)}cc}}",
        "\\toprule",
        "Dataset & "
        + " & ".join(label for _, label in methods)
        + " & $\\chi^2_F$ & $p$ \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        sub = result[result["dataset"] == dataset].set_index("method")
        if sub.empty:
            continue
        best_rank = sub["mean_rank"].min()
        cells = []
        for key, _ in methods:
            row = sub.loc[key]
            text = (
                f"{row['mean_score']:.3f} $\\pm$ {row['std_score']:.3f} "
                f"({row['mean_rank']:.2f}\\textsuperscript{{{row['nemenyi_group']}}})"
            )
            if np.isclose(row["mean_rank"], best_rank):
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        first = sub.iloc[0]
        lines.append(
            f"{dataset} & "
            + " & ".join(cells)
            + f" & {first['friedman_stat']:.2f} & {format_p(first['friedman_p'])} \\\\"
        )
    cd = float(result["nemenyi_cd"].iloc[0])
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        (
            f"\\caption{{Per-dataset comparison over {n_splits} paired CV splits. "
            f"Cells show {metric} mean $\\pm$ SD and, in parentheses, the mean "
            f"Friedman rank (lower is better) with Nemenyi group letters; methods "
            f"sharing a letter do not differ significantly "
            f"($\\alpha={alpha}$, CD$={cd:.2f}$). Best mean rank in bold. "
            f"$\\chi^2_F$ and $p$ refer to the Friedman omnibus test per dataset.}}"
        ),
        "\\label{tab:mlp_per_dataset_friedman}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    mlp = load_mlp_per_split(args.mlp_dir.resolve(), args.metric)
    worc = {} if args.exclude_worc else load_worc_per_split(
        args.worc_dir.resolve(), args.metric
    )
    pivots = build_pivots(mlp, worc, args.exclude_worc)
    if not pivots:
        raise SystemExit("No per-split results found.")

    tables: list[pd.DataFrame] = []
    for dataset, pivot in pivots.items():
        table, omnibus = friedman_nemenyi(pivot, args.alpha)
        tables.append(table.assign(dataset=dataset))
        print_summary(dataset, table, omnibus, args.alpha)

    result = pd.concat(tables, ignore_index=True)
    result = result[["dataset"] + [c for c in result.columns if c != "dataset"]]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_mlp_only" if args.exclude_worc else ""
    out_path = output_dir / f"mlp_benchmark_per_dataset_ranks{suffix}.csv"
    result.to_csv(out_path, index=False)
    tex_path = out_path.with_suffix(".tex")
    tex_path.write_text(
        build_latex_table(result, args.metric, args.alpha), encoding="utf-8"
    )
    print(f"\nWrote {out_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
