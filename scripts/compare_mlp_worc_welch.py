"""Welch unpaired t-tests comparing MLP CV AUCs vs WORC radiomics AUCs.

Loads per-split AUCs from ``Results/nnFoundation/MLP_Results`` and
``All_performances["AUC"]`` from ``Results/WORC/pastperformance``, then runs
Welch's unpaired t-test for every method pair within each dataset.

Example:

    python scripts/compare_mlp_worc_welch.py
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MLP_DIR = REPO_ROOT / "Results" / "nnFoundation" / "MLP_Results"
DEFAULT_WORC_DIR = REPO_ROOT / "Results" / "WORC" / "pastperformance"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Results" / "nnFoundation"

DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")
DATASET_ORDER = {name: index for index, name in enumerate(DATASETS)}
METHOD_ORDER = {
    "CNN_mlp": 0,
    "CNN_mlp_pca": 1,
    "ViT_mlp": 2,
    "ViT_mlp_pca": 3,
    "WORC_radiomics": 4,
}

PER_SPLIT_RE = re.compile(
    r"^(?P<dataset>Lipo|Desmoid|Liver|GIST|CRLM|Melanoma)_"
    r"(?P<stem>nncnn|nnvit)_"
    r"(?P<model>mlp_pca|mlp|linear)_cv_per_split\.csv$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Welch unpaired t-tests on AUC across MLP CV methods and "
            "WORC radiomics baselines, per dataset."
        )
    )
    parser.add_argument(
        "--mlp-dir",
        type=Path,
        default=DEFAULT_MLP_DIR,
        help="Directory with *_cv_per_split.csv files.",
    )
    parser.add_argument(
        "--worc-dir",
        type=Path,
        default=DEFAULT_WORC_DIR,
        help="Directory with WORCDatabase_performance_*_WORC371.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for pairwise and winners CSVs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for winners table (default: 0.05).",
    )
    return parser.parse_args()


def method_id(stem: str, model: str) -> str:
    features = "CNN" if stem == "nncnn" else "ViT"
    return f"{features}_{model}"


def extract_auc_scores(frame: pd.DataFrame) -> np.ndarray:
    if "auroc" in frame.columns:
        values = frame["auroc"].to_numpy(dtype=float)
    elif "auc" in frame.columns:
        values = frame["auc"].to_numpy(dtype=float)
    else:
        raise ValueError("Per-split CSV missing 'auroc' and 'auc' columns.")
    return values


def load_mlp_scores(mlp_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    if not mlp_dir.is_dir():
        raise SystemExit(f"MLP directory not found: {mlp_dir}")

    by_dataset: dict[str, dict[str, np.ndarray]] = {name: {} for name in DATASETS}
    matched = 0
    for path in sorted(mlp_dir.glob("*_cv_per_split.csv")):
        match = PER_SPLIT_RE.match(path.name)
        if match is None:
            continue
        if match.group("model") == "linear":
            # Keep MLP / MLP+PCA only for the planned comparison set.
            continue
        dataset = match.group("dataset")
        mid = method_id(match.group("stem"), match.group("model"))
        frame = pd.read_csv(path)
        by_dataset[dataset][mid] = extract_auc_scores(frame)
        matched += 1

    if matched == 0:
        raise SystemExit(f"No matching *_cv_per_split.csv files in {mlp_dir}")
    return by_dataset


def load_worc_scores(worc_dir: Path) -> dict[str, np.ndarray]:
    if not worc_dir.is_dir():
        raise SystemExit(f"WORC directory not found: {worc_dir}")

    scores: dict[str, np.ndarray] = {}
    for path in sorted(worc_dir.glob("WORCDatabase_performance_*_WORC371.json")):
        dataset = path.name.removeprefix("WORCDatabase_performance_").removesuffix(
            "_WORC371.json"
        )
        if dataset not in DATASET_ORDER:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        all_perf = payload.get("All_performances") or {}
        auc = all_perf.get("AUC")
        if not auc:
            raise SystemExit(f"No All_performances['AUC'] in {path.name}")
        scores[dataset] = np.asarray(auc, dtype=float)
    if not scores:
        raise SystemExit(f"No WORC past-performance JSONs in {worc_dir}")
    return scores


def welch_unpaired(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float]:
    result = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    t_stat = float(result.statistic)
    p_value = float(result.pvalue)
    if np.isnan(t_stat) or np.isnan(p_value):
        return float("nan"), float("nan")
    return t_stat, p_value


def ordered_method_pairs(methods: list[str]) -> list[tuple[str, str]]:
    ranked = sorted(methods, key=lambda m: METHOD_ORDER.get(m, 99))
    return list(combinations(ranked, 2))


def build_pairwise_rows(
    by_dataset: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        methods = by_dataset.get(dataset) or {}
        if len(methods) < 2:
            continue
        for method_a, method_b in ordered_method_pairs(list(methods.keys())):
            scores_a = methods[method_a]
            scores_b = methods[method_b]
            mean_a = float(np.mean(scores_a))
            mean_b = float(np.mean(scores_b))
            t_stat, p_value = welch_unpaired(scores_a, scores_b)
            rows.append(
                {
                    "dataset": dataset,
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": "AUC",
                    "n_a": int(len(scores_a)),
                    "n_b": int(len(scores_b)),
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_diff": mean_a - mean_b,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "test": "welch_unpaired",
                }
            )
    return rows


def build_winner_rows(
    by_dataset: dict[str, dict[str, np.ndarray]],
    pairwise: pd.DataFrame,
    alpha: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        methods = by_dataset.get(dataset) or {}
        if not methods:
            continue
        ranking = sorted(
            (
                (method, float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0)
                for method, scores in methods.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        winner, winner_mean, winner_std = ranking[0]
        if len(ranking) == 1:
            rows.append(
                {
                    "dataset": dataset,
                    "winner": winner,
                    "winner_mean_auc": winner_mean,
                    "winner_std_auc": winner_std,
                    "runner_up": None,
                    "runner_up_mean_auc": None,
                    "mean_diff": None,
                    "p_value": None,
                    "significant_0.05": False,
                    "test": "welch_unpaired",
                }
            )
            continue

        runner_up, runner_mean, _ = ranking[1]
        pair = pairwise[
            (pairwise["dataset"] == dataset)
            & (
                ((pairwise["method_a"] == winner) & (pairwise["method_b"] == runner_up))
                | ((pairwise["method_a"] == runner_up) & (pairwise["method_b"] == winner))
            )
        ]
        if pair.empty:
            mean_diff = winner_mean - runner_mean
            p_value = float("nan")
        else:
            entry = pair.iloc[0]
            # Orient diff as winner - runner_up
            if entry["method_a"] == winner:
                mean_diff = float(entry["mean_diff"])
            else:
                mean_diff = -float(entry["mean_diff"])
            p_value = float(entry["p_value"])

        rows.append(
            {
                "dataset": dataset,
                "winner": winner,
                "winner_mean_auc": winner_mean,
                "winner_std_auc": winner_std,
                "runner_up": runner_up,
                "runner_up_mean_auc": runner_mean,
                "mean_diff": mean_diff,
                "p_value": p_value,
                "significant_0.05": bool(
                    p_value == p_value and p_value < alpha
                ),
                "test": "welch_unpaired",
            }
        )
    return rows


def print_rankings(by_dataset: dict[str, dict[str, np.ndarray]], winners: pd.DataFrame) -> None:
    for dataset in DATASETS:
        methods = by_dataset.get(dataset) or {}
        if not methods:
            continue
        ranking = sorted(
            ((m, float(np.mean(s))) for m, s in methods.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        print(f"\n{dataset}:")
        for rank, (method, mean_auc) in enumerate(ranking, start=1):
            print(f"  {rank}. {method:16s} mean AUC={mean_auc:.4f}")
        row = winners[winners["dataset"] == dataset]
        if not row.empty:
            w = row.iloc[0]
            print(
                f"  winner={w['winner']} vs {w['runner_up']}: "
                f"diff={w['mean_diff']:.4f} p={w['p_value']:.4g} "
                f"sig@0.05={w['significant_0.05']}"
            )


def main() -> None:
    args = parse_args()
    mlp_dir = args.mlp_dir.resolve()
    worc_dir = args.worc_dir.resolve()
    output_dir = args.output_dir.resolve()

    by_dataset = load_mlp_scores(mlp_dir)
    worc_scores = load_worc_scores(worc_dir)
    for dataset, scores in worc_scores.items():
        by_dataset.setdefault(dataset, {})["WORC_radiomics"] = scores

    pairwise_rows = build_pairwise_rows(by_dataset)
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["_d"] = pairwise["dataset"].map(DATASET_ORDER)
    pairwise["_a"] = pairwise["method_a"].map(METHOD_ORDER)
    pairwise["_b"] = pairwise["method_b"].map(METHOD_ORDER)
    pairwise = (
        pairwise.sort_values(["_d", "_a", "_b"], kind="mergesort")
        .drop(columns=["_d", "_a", "_b"])
        .reset_index(drop=True)
    )

    winners = pd.DataFrame(build_winner_rows(by_dataset, pairwise, args.alpha))
    winners["_d"] = winners["dataset"].map(DATASET_ORDER)
    winners = (
        winners.sort_values("_d", kind="mergesort")
        .drop(columns=["_d"])
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pairwise_path = output_dir / "mlp_worc_pairwise_welch.csv"
    winners_path = output_dir / "mlp_worc_dataset_winners.csv"
    pairwise.to_csv(pairwise_path, index=False)
    winners.to_csv(winners_path, index=False)

    print(f"Wrote {pairwise_path} ({len(pairwise)} pairs)")
    print(f"Wrote {winners_path} ({len(winners)} datasets)")
    print_rankings(by_dataset, winners)


if __name__ == "__main__":
    main()
