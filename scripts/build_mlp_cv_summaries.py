"""Aggregate nnFoundation MLP CV summaries and WORC radiomics baselines.

Reads top-level ``*_cv_summary.csv`` files from
``Results/nnFoundation/MLP_Results`` and WORC past-performance JSONs from
``Results/WORC/pastperformance``. Writes a single comparison CSV with mean and
std for each metric (plus dataset / features / model metadata).

Example:

    python scripts/build_mlp_cv_summaries.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_DIR = REPO_ROOT / "Results" / "nnFoundation" / "MLP_Results"
DEFAULT_WORC_DIR = REPO_ROOT / "Results" / "WORC" / "pastperformance"
DEFAULT_OUTPUT = REPO_ROOT / "Results" / "all_mlp_cv_summaries.csv"

DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")
DATASET_ORDER = {name: index for index, name in enumerate(DATASETS)}
FEATURES_ORDER = {"CNN": 0, "ViT": 1, "WORC": 2}
MODEL_ORDER = {"mlp": 0, "mlp_pca": 1, "linear": 2, "radiomics": 3}
METRICS_ORDER = (
    "AUC",
    "Accuracy",
    "BCA",
    "F1_score",
    "NPV",
    "Precision",
    "Sensitivity",
    "Specificity",
    "AP",
)
SUMMARY_NAME_RE = re.compile(
    r"^(?P<dataset>Lipo|Desmoid|Liver|GIST|CRLM|Melanoma)_"
    r"(?P<stem>nncnn|nnvit)_"
    r"(?P<model>mlp_pca|mlp|linear)_cv_summary\.csv$"
)
WORC_STAT_RE = re.compile(
    r"^(?P<mean>[-+eE0-9.]+)\s*\((?P<lo>[-+eE0-9.]+),\s*(?P<hi>[-+eE0-9.]+)\)\s*$"
)
WORC_METRIC_MAP = {
    "AUC": "AUC",
    "Accuracy": "Accuracy",
    "BCA": "BCA",
    "F1-score": "F1_score",
    "NPV": "NPV",
    "Precision": "Precision",
    "Sensitivity": "Sensitivity",
    "Specificity": "Specificity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Results/all_mlp_cv_summaries.csv from MLP CV summaries "
            "and WORC past-performance JSONs (mean/std only)."
        )
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=DEFAULT_SUMMARY_DIR,
        help="Directory with *_cv_summary.csv files (default: MLP_Results).",
    )
    parser.add_argument(
        "--worc-dir",
        type=Path,
        default=DEFAULT_WORC_DIR,
        help="Directory with WORCDatabase_performance_*_WORC371.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path (default: Results/all_mlp_cv_summaries.csv).",
    )
    parser.add_argument(
        "--skip-worc",
        action="store_true",
        help="Do not append WORC radiomics baselines.",
    )
    return parser.parse_args()


def normalize_metric_name(metric: str) -> str:
    return str(metric).replace("-", "_").replace(" ", "_")


def row_from_summary(path: Path) -> dict[str, object] | None:
    match = SUMMARY_NAME_RE.match(path.name)
    if match is None:
        return None
    summary = pd.read_csv(path)
    row: dict[str, object] = {
        "dataset": match.group("dataset"),
        "features": "CNN" if match.group("stem") == "nncnn" else "ViT",
        "model": match.group("model"),
        "source_file": path.name,
    }
    if "n_splits" in summary.columns and not summary.empty:
        row["n_splits"] = int(summary["n_splits"].iloc[0])
    for _, entry in summary.iterrows():
        metric = normalize_metric_name(entry["metric"])
        if "mean" in summary.columns:
            row[f"{metric}_mean"] = float(entry["mean"])
        if "std" in summary.columns:
            row[f"{metric}_std"] = float(entry["std"])
    return row


def collect_mlp_rows(summary_dir: Path) -> list[dict[str, object]]:
    if not summary_dir.is_dir():
        raise SystemExit(f"Summary directory not found: {summary_dir}")
    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for path in sorted(summary_dir.glob("*_cv_summary.csv")):
        row = row_from_summary(path)
        if row is None:
            skipped.append(path.name)
            continue
        rows.append(row)
    if skipped:
        print(f"Skipped {len(skipped)} non-matching summary file(s).")
    if not rows:
        raise SystemExit(f"No matching *_cv_summary.csv files in {summary_dir}")
    return rows


def parse_worc_statistics(statistics: dict) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key, value in statistics.items():
        name = key.replace(" 95%:", "").strip()
        metric = WORC_METRIC_MAP.get(name)
        if metric is None:
            continue
        match = WORC_STAT_RE.match(str(value).strip())
        if match is None:
            raise ValueError(f"Unrecognized WORC statistic for {key!r}: {value!r}")
        parsed[f"{metric}_mean"] = float(match.group("mean"))
    return parsed


def parse_worc_all_performances(all_performances: dict) -> dict[str, float | int]:
    parsed: dict[str, float | int] = {}
    for name, metric in WORC_METRIC_MAP.items():
        values = all_performances.get(name)
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        parsed[f"{metric}_std"] = float(np.std(array, ddof=0))
        parsed["n_splits"] = int(len(array))
    return parsed


def collect_worc_rows(worc_dir: Path) -> list[dict[str, object]]:
    if not worc_dir.is_dir():
        raise SystemExit(f"WORC past-performance directory not found: {worc_dir}")
    rows: list[dict[str, object]] = []
    for path in sorted(worc_dir.glob("WORCDatabase_performance_*_WORC371.json")):
        dataset = path.name.removeprefix("WORCDatabase_performance_").removesuffix(
            "_WORC371.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        row: dict[str, object] = {
            "dataset": dataset,
            "features": "WORC",
            "model": "radiomics",
            "source_file": path.name,
        }
        row.update(parse_worc_statistics(payload.get("Statistics") or {}))
        row.update(parse_worc_all_performances(payload.get("All_performances") or {}))
        rows.append(row)
    if not rows:
        raise SystemExit(f"No WORC past-performance JSONs in {worc_dir}")
    return rows


def order_columns(frame: pd.DataFrame) -> list[str]:
    meta = [
        column
        for column in ("dataset", "features", "model", "n_splits", "source_file")
        if column in frame.columns
    ]
    metric_cols: list[str] = []
    for metric in METRICS_ORDER:
        for suffix in ("_mean", "_std"):
            name = f"{metric}{suffix}"
            if name in frame.columns:
                metric_cols.append(name)
    leftovers = [column for column in frame.columns if column not in meta + metric_cols]
    return meta + metric_cols + leftovers


def sort_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.assign(
            _dataset=frame["dataset"].map(DATASET_ORDER),
            _features=frame["features"].map(FEATURES_ORDER),
            _model=frame["model"].map(MODEL_ORDER),
        )
        .sort_values(["_dataset", "_features", "_model"], kind="mergesort")
        .drop(columns=["_dataset", "_features", "_model"])
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    rows = collect_mlp_rows(args.summary_dir.resolve())
    if not args.skip_worc:
        rows.extend(collect_worc_rows(args.worc_dir.resolve()))

    frame = pd.DataFrame(rows)
    frame = sort_rows(frame)
    frame = frame[order_columns(frame)]

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    print(f"Wrote {output} ({len(frame)} rows, {len(frame.columns)} columns)")
    preview_cols = [
        column
        for column in ("dataset", "features", "model", "AUC_mean", "AUC_std", "BCA_mean")
        if column in frame.columns
    ]
    print(frame[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
