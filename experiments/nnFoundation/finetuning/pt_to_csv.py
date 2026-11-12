"""Convert a glovita_infer predictions.pt file to CSV.

Example:

    python experiments/nnFoundation/finetuning/pt_to_csv.py \\
      Results/nnFoundation/finetune/infer_Liver_test.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert glovita_infer *.pt predictions to a CSV table."
    )
    parser.add_argument(
        "pt_file",
        type=Path,
        help="Path to predictions.pt (keys: preds, labels, probs, case_ids).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV output path (default: <pt_file stem>.csv next to the .pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = torch.load(args.pt_file, map_location="cpu", weights_only=False)
    for key in ("preds", "labels", "probs"):
        if key not in data:
            raise SystemExit(f"{args.pt_file} missing key {key!r}; found {list(data)}")

    preds = data["preds"].detach().cpu().numpy().reshape(-1)
    labels = data["labels"].detach().cpu().numpy().reshape(-1)
    probs = data["probs"].detach().cpu().numpy()
    n = len(preds)

    case_ids = data.get("case_ids")
    if case_ids is None:
        case_ids = [str(i) for i in range(n)]
    elif len(case_ids) != n:
        raise SystemExit(
            f"case_ids length {len(case_ids)} != n={n}"
        )

    frame = {
        "case_id": list(case_ids),
        "label": labels.astype(int),
        "pred": preds.astype(int),
    }
    if probs.ndim == 1:
        frame["prob"] = probs
    else:
        for i in range(probs.shape[1]):
            frame[f"prob_{i}"] = probs[:, i]

    df = pd.DataFrame(frame)
    out = args.out or args.pt_file.with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
