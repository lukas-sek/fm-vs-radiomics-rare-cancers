"""Cohort-level WORC+radiomics vs ViT+mlp head-to-head (summary-based Welch).

Loads ``Results/all_mlp_cv_summaries.csv``, computes Delta-AUC, Welch SE/df,
95% CI, p-value, and Hedges' g per cohort, then writes a markdown table and
forest plot.

Example:

    python scripts/cohort_worc_vs_vit_head_to_head.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "Results" / "all_mlp_cv_summaries.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "Results" / "nnFoundation"
DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WORC radiomics vs ViT+mlp cohort head-to-head comparison."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Summary CSV path (default: Results/all_mlp_cv_summaries.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for markdown, CSV, and forest plot.",
    )
    return parser.parse_args()


def hedges_magnitude(g: float) -> str:
    ag = abs(g)
    if ag >= 0.80:
        return "Large"
    if ag >= 0.50:
        return "Moderate"
    if ag >= 0.20:
        return "Small"
    return "Negligible"


def local_verdict(m1: float, m2: float, diff: float, p_val: float, g: float) -> str:
    if m1 <= 0.52 and m2 <= 0.52:
        return "Both uninformative (chance level ≈ 0.50)"
    if p_val < 0.05:
        winner = "WORC" if diff > 0 else "ViT"
        return f"{winner} statistically superior (p = {p_val:.3f})"
    if p_val < 0.10:
        winner = "WORC" if diff > 0 else "ViT"
        return f"Borderline trend favoring {winner} (p = {p_val:.3f})"
    if abs(g) >= 0.80:
        return f"Substantial advantage, but overlaps 0 (p = {p_val:.3f})"
    return f"Equivalent performance (p = {p_val:.3f})"


def cohort_head_to_head(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    records: list[dict[str, object]] = []

    for cohort in DATASETS:
        worc_rows = df[(df["dataset"] == cohort) & (df["features"] == "WORC")]
        vit_rows = df[
            (df["dataset"] == cohort)
            & (df["features"] == "ViT")
            & (df["model"] == "mlp")
        ]
        if worc_rows.empty or vit_rows.empty:
            raise SystemExit(f"Missing WORC or ViT+mlp row for cohort {cohort!r}")

        worc = worc_rows.iloc[0]
        vit = vit_rows.iloc[0]

        m1 = float(worc["AUC_mean"])
        s1 = float(worc["AUC_std"])
        n1 = int(worc["n_splits"])
        m2 = float(vit["AUC_mean"])
        s2 = float(vit["AUC_std"])
        n2 = int(vit["n_splits"])
        diff = m1 - m2

        se = float(np.sqrt(s1**2 / n1 + s2**2 / n2))
        if se == 0.0:
            nu = float("nan")
            t_crit = float("nan")
            ci_low = ci_high = diff
            t_stat = float("nan")
            p_val = float("nan")
        else:
            nu = float(
                (s1**2 / n1 + s2**2 / n2) ** 2
                / (
                    ((s1**2 / n1) ** 2 / (n1 - 1))
                    + ((s2**2 / n2) ** 2 / (n2 - 1))
                )
            )
            t_crit = float(stats.t.ppf(0.975, df=nu))
            ci_low = diff - t_crit * se
            ci_high = diff + t_crit * se
            t_stat = diff / se
            p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=nu)))

        s_pool = float(
            np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        )
        j_fact = 1 - (3 / (4 * (n1 + n2) - 9))
        g = float((diff / s_pool) * j_fact) if s_pool > 0 else float("nan")
        mag = hedges_magnitude(g) if g == g else "n/a"
        verdict = local_verdict(m1, m2, diff, p_val, g)

        records.append(
            {
                "Cohort": cohort,
                "WORC AUC (Mean ± SD)": f"{m1:.3f} ± {s1:.3f}",
                "ViT AUC (Mean ± SD)": f"{m2:.3f} ± {s2:.3f}",
                "Delta AUC (WORC - ViT)": f"{diff:+.3f}",
                "95% CI of Difference": f"[{ci_low:+.3f}, {ci_high:+.3f}]",
                "Hedges' g": f"{g:.2f}",
                "p-value": f"{p_val:.3f}",
                # numeric columns for forest plot / CSV
                "local_verdict": verdict,
                "worc_mean": m1,
                "worc_std": s1,
                "worc_n": n1,
                "vit_mean": m2,
                "vit_std": s2,
                "vit_n": n2,
                "diff": diff,
                "se": se,
                "df": nu,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "t_stat": t_stat,
                "p_value": p_val,
                "hedges_g": g,
                "hedges_mag": mag,
            }
        )

    return pd.DataFrame(records)


def write_markdown_table(res_df: pd.DataFrame, path: Path) -> None:
    cols = [
        "Cohort",
        "WORC AUC (Mean ± SD)",
        "ViT AUC (Mean ± SD)",
        "Delta AUC (WORC - ViT)",
        "95% CI of Difference",
        "Hedges' g",
        "p-value",
    ]
    display = res_df[cols]
    lines = [
        "# WORC + radiomics vs ViT + mlp (cohort head-to-head)",
        "",
        "Summary-based Welch t-test on AUC (WORC n=100, ViT+mlp n=10). "
        "Delta AUC = WORC - ViT. Positive favors WORC.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_forest_plot(res_df: pd.DataFrame, path: Path) -> None:
    # Top-to-bottom = DATASETS order (same as table).
    y = np.arange(len(res_df))[::-1]
    diffs = res_df["diff"].to_numpy(dtype=float)
    lows = res_df["ci_low"].to_numpy(dtype=float)
    highs = res_df["ci_high"].to_numpy(dtype=float)
    labels = res_df["Cohort"].tolist()
    xerr = np.vstack([diffs - lows, highs - diffs])

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    ax.axvline(0.0, color="0.4", linewidth=1.0, linestyle="--", zorder=0)
    ax.errorbar(
        diffs,
        y,
        xerr=xerr,
        fmt="o",
        color="0.15",
        ecolor="0.35",
        elinewidth=1.4,
        capsize=3.5,
        markersize=6,
        zorder=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Delta AUC (WORC - ViT) with 95% CI")
    ax.set_title("Cohort head-to-head: WORC radiomics vs ViT + mlp")
    ax.set_xlim(min(-0.05, lows.min() - 0.02), max(0.25, highs.max() + 0.02))
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    res_df = cohort_head_to_head(csv_path)

    md_path = out_dir / "worc_vs_vit_cohort_head_to_head.md"
    csv_out = out_dir / "worc_vs_vit_cohort_head_to_head.csv"
    plot_path = out_dir / "forest_plot.png"

    write_markdown_table(res_df, md_path)
    res_df.to_csv(csv_out, index=False)
    write_forest_plot(res_df, plot_path)

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_out}")
    print(f"Wrote {plot_path}")
    print()
    print(md_path.read_text(encoding="utf-8").encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
