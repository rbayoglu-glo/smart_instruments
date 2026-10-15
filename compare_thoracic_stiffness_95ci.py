from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from thoracic_level_processing_config import GROUPS, LEVEL_CONFIGS, OUTPUT_DIR

R2_THRESHOLD = 0.95
STIFFNESS_COLUMN = "slope_force_range_n_per_mm"
SIDES = ("Left", "Right")

SIDE_SUMMARY_CSV = OUTPUT_DIR / "intraop_thoracic_stiffness_95ci_summary.csv"
GROUP_SUMMARY_CSV = OUTPUT_DIR / "intraop_thoracic_group_stiffness_95ci_summary.csv"
SIDE_PLOT = OUTPUT_DIR / "intraop_thoracic_group_left_right_stiffness_95ci.png"
GROUP_PLOT = OUTPUT_DIR / "intraop_thoracic_group_stiffness_95ci.png"


def ci95_half_width(values: pd.Series) -> float:
    values = values.dropna().to_numpy(float)
    if values.size < 2:
        return np.nan
    return float(
        student_t.ppf(0.975, values.size - 1)
        * np.std(values, ddof=1)
        / np.sqrt(values.size)
    )


def load_true_linear_cycles() -> pd.DataFrame:
    rows = []
    for level, trials in LEVEL_CONFIGS.items():
        for trial in trials:
            stem = f"intraop_{trial.replace(' ', '_')}"
            path = OUTPUT_DIR / f"{stem}_translational_stiffness_by_cycle.csv"
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            frame["level"] = level
            frame["side"] = trial.rsplit(" ", 1)[-1]
            frame["true_linear"] = (
                frame["ok"].astype(str).str.strip().str.lower().eq("true")
                & (pd.to_numeric(frame["r2"], errors="coerce") > R2_THRESHOLD)
            )
            rows.append(frame)
    data = pd.concat(rows, ignore_index=True)
    return data[data["true_linear"]].dropna(subset=[STIFFNESS_COLUMN]).copy()


def summarize_side(data: pd.DataFrame) -> pd.DataFrame:
    levels = list(LEVEL_CONFIGS)
    summary = (
        data.groupby(["level", "side"], sort=False)[STIFFNESS_COLUMN]
        .agg(
            mean_n_per_mm="mean",
            sd_n_per_mm="std",
            ci95_half_width_n_per_mm=ci95_half_width,
            n_cycles="count",
        )
        .reindex(pd.MultiIndex.from_product([levels, SIDES], names=["level", "side"]))
        .reset_index()
    )
    summary["sd_n_per_mm"] = summary["sd_n_per_mm"].fillna(0.0)
    summary["ci95_low_n_per_mm"] = summary["mean_n_per_mm"] - summary["ci95_half_width_n_per_mm"]
    summary["ci95_high_n_per_mm"] = summary["mean_n_per_mm"] + summary["ci95_half_width_n_per_mm"]
    summary["group"] = summary["level"].map({level: group for group, levels in GROUPS.items() for level in levels})
    return summary


def summarize_groups(data: pd.DataFrame) -> pd.DataFrame:
    group_by_level = {level: group for group, levels in GROUPS.items() for level in levels}
    grouped = data.assign(group=data["level"].map(group_by_level))
    summary = (
        grouped.groupby("group", sort=False)[STIFFNESS_COLUMN]
        .agg(
            mean_n_per_mm="mean",
            sd_n_per_mm="std",
            ci95_half_width_n_per_mm=ci95_half_width,
            n_cycles="count",
        )
        .reindex(list(GROUPS))
        .reset_index()
    )
    summary["sd_n_per_mm"] = summary["sd_n_per_mm"].fillna(0.0)
    summary["ci95_low_n_per_mm"] = summary["mean_n_per_mm"] - summary["ci95_half_width_n_per_mm"]
    summary["ci95_high_n_per_mm"] = summary["mean_n_per_mm"] + summary["ci95_half_width_n_per_mm"]
    return summary


def plot_side_summary(summary: pd.DataFrame) -> None:
    ncols = 2
    nrows = int(np.ceil(len(GROUPS) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(13.0, 4.5 * nrows), constrained_layout=True, squeeze=False
    )
    colors = {"Left": "tab:blue", "Right": "tab:orange"}
    for ax, (group, levels) in zip(axes.ravel(), GROUPS.items()):
        x = np.arange(len(levels))
        width = 0.36
        for index, side in enumerate(SIDES):
            subset = summary[(summary["group"] == group) & (summary["side"] == side)].set_index("level").reindex(levels)
            means = subset["mean_n_per_mm"].to_numpy(float)
            errors = subset["ci95_half_width_n_per_mm"].fillna(0.0).to_numpy(float)
            valid = np.isfinite(means)
            offset = (index - 0.5) * width
            bars = ax.bar(
                x[valid] + offset,
                means[valid],
                width,
                yerr=errors[valid],
                capsize=4,
                color=colors[side],
                label=side,
                error_kw={"elinewidth": 1.1, "capthick": 1.1},
            )
            for bar, mean, count in zip(
                bars,
                means[valid],
                subset["n_cycles"].to_numpy(float)[valid],
            ):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{mean:.1f}\nn={int(count)}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(x, levels)
        ax.set_ylabel("stiffness [N/mm]")
        ax.set_title(group, loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes.ravel()[len(GROUPS):]:
        ax.set_visible(False)
    fig.suptitle(
        "Thoracic Left versus Right true-linear stiffness by requested group\n"
        "Mean +/- 95% CI across cycles; strict R2 > 0.95",
        fontsize=13,
    )
    fig.savefig(SIDE_PLOT, dpi=180)
    plt.close(fig)


def plot_group_summary(summary: pd.DataFrame) -> None:
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    valid = summary["mean_n_per_mm"].notna().to_numpy(bool)
    bars = ax.bar(
        x[valid],
        summary.loc[valid, "mean_n_per_mm"],
        yerr=summary.loc[valid, "ci95_half_width_n_per_mm"].fillna(0.0),
        capsize=5,
        color="tab:blue",
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    for bar, mean, count in zip(
        bars,
        summary.loc[valid, "mean_n_per_mm"],
        summary.loc[valid, "n_cycles"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.1f}\nn={int(count)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x, summary["group"])
    ax.set_ylabel("stiffness [N/mm]")
    ax.set_title(
        "Thoracic stiffness by requested group\n"
        "Left and Right true-linear cycles pooled; mean +/- 95% CI",
        loc="left",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(GROUP_PLOT, dpi=180)
    plt.close(fig)


def main() -> None:
    data = load_true_linear_cycles()
    side_summary = summarize_side(data)
    group_summary = summarize_groups(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    side_summary.to_csv(SIDE_SUMMARY_CSV, index=False)
    group_summary.to_csv(GROUP_SUMMARY_CSV, index=False)
    plot_side_summary(side_summary)
    plot_group_summary(group_summary)
    print(f"saved {SIDE_SUMMARY_CSV}")
    print(f"saved {SIDE_PLOT}")
    print(f"saved {GROUP_SUMMARY_CSV}")
    print(f"saved {GROUP_PLOT}")
    print("\nThoracic side summary:")
    print(side_summary.to_string(index=False))
    print("\nThoracic group summary:")
    print(group_summary.to_string(index=False))


if __name__ == "__main__":
    main()
