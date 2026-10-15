from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from compare_thoracic_stiffness_95ci import load_true_linear_cycles as load_thoracic_true_linear_cycles
from intact_level_processing_config import OUTPUT_DIR

LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5")
ALL_LEVELS = (
    "T1-T2", "T2-T3", "T3-T4", "T4-T5", "T5-T6", "T6-T7", "T7-T8", "T8-T9",
    "T9-T10", "T10-T11", "T11-T12", "T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5",
)
SIDES = ("Left", "Right")
STIFFNESS_COLUMN = "slope_force_range_n_per_mm"
R2_THRESHOLD = 0.95

SIDE_SUMMARY_CSV = OUTPUT_DIR / "intraop_left_right_stiffness_95ci_summary.csv"
SIDE_PLOT = OUTPUT_DIR / "intraop_left_right_stiffness_95ci.png"
LEVEL_SUMMARY_CSV = OUTPUT_DIR / "intraop_level_stiffness_95ci_summary.csv"
LEVEL_PLOT = OUTPUT_DIR / "intraop_level_stiffness_95ci.png"
THORACIC_SIDE_SUMMARY_CSV = OUTPUT_DIR / "intraop_thoracic_stiffness_95ci_summary.csv"
ALL_LEVEL_SIDE_SUMMARY_CSV = OUTPUT_DIR / "intraop_all_levels_left_right_stiffness_95ci_summary.csv"
ALL_LEVEL_SIDE_PLOT = OUTPUT_DIR / "intraop_all_levels_left_right_stiffness_95ci_horizontal.png"
ALL_LEVEL_POOLED_SUMMARY_CSV = OUTPUT_DIR / "intraop_all_levels_pooled_stiffness_95ci_summary.csv"
ALL_LEVEL_POOLED_PLOT = OUTPUT_DIR / "intraop_all_levels_pooled_stiffness_95ci_horizontal.png"


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
    frames = []
    for level in LEVELS:
        for side in SIDES:
            path = OUTPUT_DIR / f"intraop_{level}_{side}_translational_stiffness_by_cycle.csv"
            frame = pd.read_csv(path)
            frame["level"] = level
            frame["side"] = side
            frame["true_linear"] = (
                frame["ok"].astype(str).str.strip().str.lower().eq("true")
                & (pd.to_numeric(frame["r2"], errors="coerce") > R2_THRESHOLD)
            )
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    return data[data["true_linear"]].dropna(subset=[STIFFNESS_COLUMN]).copy()


def summarize_side(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(["level", "side"], sort=False)[STIFFNESS_COLUMN]
        .agg(
            mean_n_per_mm="mean",
            sd_n_per_mm="std",
            ci95_half_width_n_per_mm=ci95_half_width,
            n_cycles="count",
        )
        .reindex(pd.MultiIndex.from_product([LEVELS, SIDES], names=["level", "side"]))
        .reset_index()
    )
    summary["sd_n_per_mm"] = summary["sd_n_per_mm"].fillna(0.0)
    summary["ci95_low_n_per_mm"] = summary["mean_n_per_mm"] - summary["ci95_half_width_n_per_mm"]
    summary["ci95_high_n_per_mm"] = summary["mean_n_per_mm"] + summary["ci95_half_width_n_per_mm"]
    return summary


def summarize_level(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby("level", sort=False)[STIFFNESS_COLUMN]
        .agg(
            mean_n_per_mm="mean",
            sd_n_per_mm="std",
            ci95_half_width_n_per_mm=ci95_half_width,
            n_cycles="count",
        )
        .reindex(LEVELS)
        .reset_index()
    )
    summary["sd_n_per_mm"] = summary["sd_n_per_mm"].fillna(0.0)
    summary["ci95_low_n_per_mm"] = summary["mean_n_per_mm"] - summary["ci95_half_width_n_per_mm"]
    summary["ci95_high_n_per_mm"] = summary["mean_n_per_mm"] + summary["ci95_half_width_n_per_mm"]
    return summary


def plot_side_summary(summary: pd.DataFrame) -> None:
    x = np.arange(len(LEVELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    colors = {"Left": "tab:blue", "Right": "tab:orange"}
    for index, side in enumerate(SIDES):
        subset = summary[summary["side"] == side]
        offset = (index - 0.5) * width
        bars = ax.bar(
            x + offset,
            subset["mean_n_per_mm"],
            width,
            yerr=subset["ci95_half_width_n_per_mm"],
            capsize=5,
            color=colors[side],
            label=side,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        for bar, mean, count in zip(bars, subset["mean_n_per_mm"], subset["n_cycles"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{mean:.1f}\nn={int(count)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x, LEVELS)
    ax.set_ylabel("translational stiffness [N/mm]")
    ax.set_title(
        "Intraoperative Left versus Right stiffness by level\n"
        "Mean +/- 95% CI across true-linear cycles (R2 > 0.95)",
        loc="left",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Side", loc="upper left")
    fig.savefig(SIDE_PLOT, dpi=180)
    plt.close(fig)


def plot_all_level_side_summary(lumbar_summary: pd.DataFrame) -> None:
    columns = [
        "level",
        "side",
        "mean_n_per_mm",
        "sd_n_per_mm",
        "ci95_half_width_n_per_mm",
        "n_cycles",
        "ci95_low_n_per_mm",
        "ci95_high_n_per_mm",
    ]
    thoracic_summary = pd.read_csv(THORACIC_SIDE_SUMMARY_CSV)
    summary = pd.concat(
        [thoracic_summary[columns], lumbar_summary[columns]], ignore_index=True
    )
    summary["level"] = summary["level"].astype(str).str.strip()
    summary["_has_summary"] = True
    full_index = pd.MultiIndex.from_product(
        [ALL_LEVELS, SIDES], names=["level", "side"]
    )
    summary = summary.set_index(["level", "side"]).reindex(full_index).reset_index()
    summary["n_cycles"] = pd.to_numeric(summary["n_cycles"], errors="coerce").fillna(0).astype(int)
    has_summary = summary["_has_summary"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    summary["status"] = np.where(
        ~has_summary,
        "not analyzed",
        np.where(summary["mean_n_per_mm"].isna(), "no valid fit", "available"),
    )
    summary[columns + ["status"]].to_csv(ALL_LEVEL_SIDE_SUMMARY_CSV, index=False)

    positions = np.arange(len(ALL_LEVELS), dtype=float)
    bar_height = 0.30
    offsets = {"Left": -bar_height / 2, "Right": bar_height / 2}
    colors = {"Left": "tab:blue", "Right": "tab:orange"}
    fig, ax = plt.subplots(
        figsize=(13.0, max(8.0, 0.58 * len(ALL_LEVELS) + 1.5)),
        constrained_layout=True,
    )
    upper_bounds = []
    for side in SIDES:
        subset = summary[summary["side"] == side].set_index("level").reindex(ALL_LEVELS)
        means = pd.to_numeric(subset["mean_n_per_mm"], errors="coerce").to_numpy(float)
        errors = pd.to_numeric(
            subset["ci95_half_width_n_per_mm"], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        counts = subset["n_cycles"].to_numpy(int)
        valid = np.isfinite(means)
        y = positions + offsets[side]
        ax.barh(
            y[valid],
            means[valid],
            height=bar_height * 0.88,
            xerr=errors[valid],
            capsize=3,
            color=colors[side],
            label=side,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        upper_bounds.extend((means[valid] + errors[valid]).tolist())
        for index in np.flatnonzero(valid):
            ax.text(
                means[index] + errors[index] + 0.20,
                y[index],
                f"{means[index]:.1f} (n={counts[index]})",
                va="center",
                ha="left",
                fontsize=7,
            )
        for index in np.flatnonzero(~valid):
            status = str(subset["status"].iloc[index])
            label = "not analyzed" if status == "not analyzed" else "no valid fit"
            ax.text(0.25, y[index], label, va="center", ha="left", fontsize=7, color="0.4")

    xmax = max(upper_bounds, default=1.0)
    ax.set_xlim(0.0, xmax + max(3.0, 0.14 * xmax))
    ax.set_yticks(positions, ALL_LEVELS)
    ax.invert_yaxis()
    ax.set_xlabel("translational stiffness [N/mm]")
    ax.set_ylabel("level")
    ax.set_title(
        "Left versus Right stiffness by thoracic and lumbar level\n"
        "Mean +/- 95% CI across strict true-linear cycles (R2 > 0.95)",
        loc="left",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="Side", loc="lower right")
    fig.savefig(ALL_LEVEL_SIDE_PLOT, dpi=180)
    plt.close(fig)


def summarize_all_level_pooled(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby("level", sort=False)[STIFFNESS_COLUMN]
        .agg(
            mean_n_per_mm="mean",
            sd_n_per_mm="std",
            ci95_half_width_n_per_mm=ci95_half_width,
            n_cycles="count",
        )
        .reindex(ALL_LEVELS)
        .reset_index()
    )
    summary["n_cycles"] = pd.to_numeric(summary["n_cycles"], errors="coerce").fillna(0).astype(int)
    summary["sd_n_per_mm"] = summary["sd_n_per_mm"].fillna(0.0)
    summary["ci95_low_n_per_mm"] = summary["mean_n_per_mm"] - summary["ci95_half_width_n_per_mm"]
    summary["ci95_high_n_per_mm"] = summary["mean_n_per_mm"] + summary["ci95_half_width_n_per_mm"]
    summary["status"] = np.where(summary["mean_n_per_mm"].isna(), "no valid cycles", "available")
    return summary


def plot_all_level_pooled_summary(summary: pd.DataFrame) -> None:
    means = pd.to_numeric(summary["mean_n_per_mm"], errors="coerce").to_numpy(float)
    errors = pd.to_numeric(summary["ci95_half_width_n_per_mm"], errors="coerce").fillna(0.0).to_numpy(float)
    counts = summary["n_cycles"].to_numpy(int)
    positions = np.arange(len(ALL_LEVELS), dtype=float)
    valid = np.isfinite(means)

    fig, ax = plt.subplots(
        figsize=(13.0, max(8.0, 0.58 * len(ALL_LEVELS) + 1.5)),
        constrained_layout=True,
    )
    ax.barh(
        positions[valid],
        means[valid],
        height=0.54,
        xerr=errors[valid],
        capsize=3,
        color="tab:blue",
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    for index in np.flatnonzero(valid):
        ax.text(
            means[index] + errors[index] + 0.20,
            positions[index],
            f"{means[index]:.1f} (n={counts[index]})",
            va="center",
            ha="left",
            fontsize=7,
        )
    for index in np.flatnonzero(~valid):
        ax.text(0.25, positions[index], "no valid cycles", va="center", ha="left", fontsize=7, color="0.4")

    upper = means[valid] + errors[valid]
    xmax = float(np.max(upper)) if upper.size else 1.0
    ax.set_xlim(0.0, xmax + max(3.0, 0.14 * xmax))
    ax.set_yticks(positions, ALL_LEVELS)
    ax.invert_yaxis()
    ax.set_xlabel("translational stiffness [N/mm]")
    ax.set_ylabel("level")
    ax.set_title(
        "Pooled Left and Right stiffness by thoracic and lumbar level\n"
        "Mean +/- 95% CI across strict true-linear cycles (R2 > 0.95)",
        loc="left",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(ALL_LEVEL_POOLED_PLOT, dpi=180)
    plt.close(fig)


def plot_level_summary(summary: pd.DataFrame) -> None:
    x = np.arange(len(LEVELS))
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    bars = ax.bar(
        x,
        summary["mean_n_per_mm"],
        yerr=summary["ci95_half_width_n_per_mm"],
        capsize=5,
        color="tab:blue",
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    for bar, mean, count in zip(bars, summary["mean_n_per_mm"], summary["n_cycles"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.1f}\nn={int(count)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x, LEVELS)
    ax.set_ylabel("translational stiffness [N/mm]")
    ax.set_title(
        "Intraoperative stiffness by lumbar level\n"
        "Left and Right true-linear cycles pooled; mean +/- 95% CI",
        loc="left",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(LEVEL_PLOT, dpi=180)
    plt.close(fig)


def main() -> None:
    data = load_true_linear_cycles()
    side_summary = summarize_side(data)
    level_summary = summarize_level(data)
    thoracic_data = load_thoracic_true_linear_cycles()
    all_level_cycles = pd.concat([data, thoracic_data], ignore_index=True)
    all_level_pooled_summary = summarize_all_level_pooled(all_level_cycles)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    side_summary.to_csv(SIDE_SUMMARY_CSV, index=False)
    level_summary.to_csv(LEVEL_SUMMARY_CSV, index=False)
    all_level_pooled_summary.to_csv(ALL_LEVEL_POOLED_SUMMARY_CSV, index=False)
    plot_side_summary(side_summary)
    plot_level_summary(level_summary)
    plot_all_level_side_summary(side_summary)
    plot_all_level_pooled_summary(all_level_pooled_summary)
    print(f"saved {SIDE_SUMMARY_CSV}")
    print(f"saved {SIDE_PLOT}")
    print(f"saved {LEVEL_SUMMARY_CSV}")
    print(f"saved {LEVEL_PLOT}")
    print(f"saved {ALL_LEVEL_SIDE_SUMMARY_CSV}")
    print(f"saved {ALL_LEVEL_SIDE_PLOT}")
    print(f"saved {ALL_LEVEL_POOLED_SUMMARY_CSV}")
    print(f"saved {ALL_LEVEL_POOLED_PLOT}")
    print("\nSide summary:")
    print(side_summary.to_string(index=False))
    print("\nLevel summary:")
    print(level_summary.to_string(index=False))


if __name__ == "__main__":
    main()
