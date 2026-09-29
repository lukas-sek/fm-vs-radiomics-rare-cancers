"""Run FMCIB feature extraction from a CSV manifest."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd
from fmcib.run import get_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract FMCIB features from a CSV.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input FMCIB CSV (image_path, coords or precropped patches, label)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output path for the feature CSV",
    )
    parser.add_argument(
        "--precropped",
        type=lambda s: s.lower() in ("1", "true", "yes", "y"),
        required=True,
        metavar="BOOL",
        help="true for pre-extracted 50^3 patches; false for native volumes with coords",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root for relative image_path entries (default: current working directory)",
    )
    return parser.parse_args()


def resolve_image_paths(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Make image_path absolute so MONAI LoadImaged works regardless of cwd."""
    if "image_path" not in df.columns:
        raise SystemExit("Input CSV must contain an 'image_path' column")

    out = df.copy()
    resolved: list[str] = []
    missing: list[str] = []
    for raw in out["image_path"]:
        path = Path(str(raw))
        if not path.is_absolute():
            path = (data_root / path).resolve()
        else:
            path = path.resolve()
        if not path.exists():
            missing.append(str(path))
        resolved.append(str(path))

    if missing:
        preview = "\n  ".join(missing[:5])
        raise SystemExit(
            f"Missing {len(missing)} image file(s). First few:\n  {preview}\n"
            f"Hint: run from the repo root, or pass --data-root pointing at it "
            f"(paths in the CSV are relative to that root)."
        )

    out["image_path"] = resolved
    return out


def main() -> None:
    args = parse_args()
    data_root = (args.data_root or Path.cwd()).resolve()

    df = pd.read_csv(args.input_csv)
    df = resolve_image_paths(df, data_root)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        df.to_csv(tmp_path, index=False)

    try:
        feature_df = get_features(str(tmp_path), precropped=args.precropped)
    finally:
        tmp_path.unlink(missing_ok=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(feature_df)} rows -> {args.output_csv}")
    print(feature_df.head())


if __name__ == "__main__":
    main()
