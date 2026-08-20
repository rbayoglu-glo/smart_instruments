from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_file_order(file_name: str) -> int:
    match = re.match(r"\s*(\d+)\s*-", str(file_name))
    if match:
        return int(match.group(1))
    return 10**9


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    out_dir = repo_root / "output" / "linear_region_stiffness_cycle_grids"
    summary_csv = out_dir / "all_interventions_linear_region_stiffness_summary.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError("Summary CSV is empty.")

    df["_order"] = df["file_name"].map(parse_file_order)
    df = df.sort_values(["_order", "file_name"], kind="stable").reset_index(drop=True)

    x = range(len(df))
    y = df["k_mean_n_per_mm"].to_numpy()
    yerr = df["k_std_n_per_mm"].to_numpy()
    labels = df["intervention"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    bars = ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=6,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    ax.set_title("Linear-Region Stiffness by Intervention (Mean ± SD)")
    ax.set_xlabel("Intervention (experimental order)")
    ax.set_ylabel("Stiffness (N/mm)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Annotate each bar with n_valid/n_cycles for context.
    for idx, rect in enumerate(bars):
        n_valid = int(df.loc[idx, "n_valid"])
        n_cycles = int(df.loc[idx, "n_cycles"])
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height + yerr[idx] + 0.2,
            f"{n_valid}/{n_cycles}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    out_png = out_dir / "all_interventions_linear_region_stiffness_mean_std_bar_by_order.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
