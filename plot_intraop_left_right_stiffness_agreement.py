from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from compare_intraop_stiffness_95ci import LEVELS, SIDES, ci95_half_width, load_true_linear_cycles, summarize_side
from intact_level_processing_config import OUTPUT_DIR

OUT_FIG = OUTPUT_DIR / "intraop_left_right_stiffness_agreement_95ci.png"
OUT_PAIRED_CSV = OUTPUT_DIR / "intraop_left_right_stiffness_paired_95ci.csv"
OUT_STATS_CSV = OUTPUT_DIR / "intraop_left_right_stiffness_agreement_statistics.csv"


def pair_levels(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot(index="level", columns="side")
    paired = pd.DataFrame(
        {
            "level": LEVELS,
            "left_mean_n_per_mm": [wide.loc[level, ("mean_n_per_mm", "Left")] for level in LEVELS],
            "right_mean_n_per_mm": [wide.loc[level, ("mean_n_per_mm", "Right")] for level in LEVELS],
            "left_ci95_half_width_n_per_mm": [wide.loc[level, ("ci95_half_width_n_per_mm", "Left")] for level in LEVELS],
            "right_ci95_half_width_n_per_mm": [wide.loc[level, ("ci95_half_width_n_per_mm", "Right")] for level in LEVELS],
            "left_n_cycles": [wide.loc[level, ("n_cycles", "Left")] for level in LEVELS],
            "right_n_cycles": [wide.loc[level, ("n_cycles", "Right")] for level in LEVELS],
        }
    )
    paired["level_mean_n_per_mm"] = (
        paired["left_mean_n_per_mm"] + paired["right_mean_n_per_mm"]
    ) / 2.0
    paired["right_minus_left_n_per_mm"] = (
        paired["right_mean_n_per_mm"] - paired["left_mean_n_per_mm"]
    )
    return paired


def agreement_statistics(paired: pd.DataFrame) -> pd.DataFrame:
    differences = paired["right_minus_left_n_per_mm"].to_numpy(float)
    bias = float(np.mean(differences))
    sd_difference = float(np.std(differences, ddof=1))
    ci_half_width = float(
        student_t.ppf(0.975, len(differences) - 1)
        * sd_difference
        / np.sqrt(len(differences))
    )
    loa_half_width = 1.96 * sd_difference
    stats = [
        ("n_paired_levels", len(differences)),
        ("bias_right_minus_left_n_per_mm", bias),
        ("bias_ci95_lower_n_per_mm", bias - ci_half_width),
        ("bias_ci95_upper_n_per_mm", bias + ci_half_width),
        ("sd_paired_difference_n_per_mm", sd_difference),
        ("loa_lower_n_per_mm", bias - loa_half_width),
        ("loa_upper_n_per_mm", bias + loa_half_width),
        (
            "pearson_correlation_descriptive",
            float(np.corrcoef(paired["left_mean_n_per_mm"], paired["right_mean_n_per_mm"])[0, 1]),
        ),
    ]
    return pd.DataFrame(stats, columns=["statistic", "value"])


def plot_cycle_distributions(ax: plt.Axes, data: pd.DataFrame) -> None:
    positions = []
    values = []
    sides = []
    for index, level in enumerate(LEVELS):
        for offset, side in [(-0.18, "Left"), (0.18, "Right")]:
            values.append(
                data[(data["level"] == level) & (data["side"] == side)][
                    "slope_force_range_n_per_mm"
                ].to_numpy(float)
            )
            positions.append(index + offset)
            sides.append(side)
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.26,
        patch_artist=True,
        showfliers=False,
    )
    colors = {"Left": "tab:blue", "Right": "tab:orange"}
    for patch, side in zip(box["boxes"], sides):
        patch.set_facecolor(colors[side])
        patch.set_alpha(0.35)
    rng = np.random.default_rng(7)
    for position, value, side in zip(positions, values, sides):
        jitter = rng.uniform(-0.06, 0.06, len(value))
        ax.scatter(
            position + jitter,
            value,
            s=18,
            color=colors[side],
            alpha=0.8,
            zorder=3,
            label=side if position == positions[sides.index(side)] else None,
        )
    ax.set_xticks(range(len(LEVELS)), LEVELS)
    ax.set_ylabel("cycle stiffness [N/mm]")
    ax.set_title("Cycle-level distributions", loc="left", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="None", color=colors["Left"], label="Left"),
        plt.Line2D([], [], marker="o", linestyle="None", color=colors["Right"], label="Right"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)


def main() -> None:
    data = load_true_linear_cycles()
    side_summary = summarize_side(data)
    paired = pair_levels(side_summary)
    stats = agreement_statistics(paired)
    paired.to_csv(OUT_PAIRED_CSV, index=False)
    stats.to_csv(OUT_STATS_CSV, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)
    colors = {"Left": "tab:blue", "Right": "tab:orange"}
    x = np.arange(len(LEVELS))

    for side, mean_column, ci_column in [
        ("Left", "left_mean_n_per_mm", "left_ci95_half_width_n_per_mm"),
        ("Right", "right_mean_n_per_mm", "right_ci95_half_width_n_per_mm"),
    ]:
        axes[0, 0].errorbar(
            x,
            paired[mean_column],
            yerr=paired[ci_column],
            marker="o",
            capsize=4,
            lw=1.3,
            color=colors[side],
            label=side,
        )
    for index, row in paired.iterrows():
        axes[0, 0].plot(
            [index, index],
            [row["left_mean_n_per_mm"], row["right_mean_n_per_mm"]],
            color="0.65",
            lw=1.0,
            zorder=0,
        )
    axes[0, 0].set_xticks(x, LEVELS)
    axes[0, 0].set_ylabel("mean stiffness [N/mm]")
    axes[0, 0].set_title("Paired Left-Right level means", loc="left", fontsize=10)
    axes[0, 0].legend(loc="upper left")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].scatter(
        paired["left_mean_n_per_mm"],
        paired["right_mean_n_per_mm"],
        s=55,
        color="tab:purple",
    )
    limits = np.asarray(
        [
            paired[["left_mean_n_per_mm", "right_mean_n_per_mm"]].min().min(),
            paired[["left_mean_n_per_mm", "right_mean_n_per_mm"]].max().max(),
        ]
    )
    pad = max(0.5, 0.08 * np.ptp(limits))
    limits = np.array([limits[0] - pad, limits[1] + pad])
    axes[0, 1].plot(limits, limits, "--", color="0.35", label="identity")
    for _, row in paired.iterrows():
        axes[0, 1].annotate(
            row["level"],
            (row["left_mean_n_per_mm"], row["right_mean_n_per_mm"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    pearson = float(stats.loc[stats["statistic"] == "pearson_correlation_descriptive", "value"].iloc[0])
    axes[0, 1].text(
        0.03,
        0.97,
        f"descriptive r={pearson:.2f}\nn={len(paired)} levels",
        transform=axes[0, 1].transAxes,
        va="top",
        fontsize=9,
    )
    axes[0, 1].set(xlim=limits, ylim=limits, xlabel="Left mean stiffness [N/mm]", ylabel="Right mean stiffness [N/mm]")
    axes[0, 1].set_title("Right versus Left", loc="left", fontsize=10)
    axes[0, 1].legend(loc="upper left", fontsize=8)
    axes[0, 1].grid(alpha=0.25)

    bias = float(stats.loc[stats["statistic"] == "bias_right_minus_left_n_per_mm", "value"].iloc[0])
    loa_low = float(stats.loc[stats["statistic"] == "loa_lower_n_per_mm", "value"].iloc[0])
    loa_high = float(stats.loc[stats["statistic"] == "loa_upper_n_per_mm", "value"].iloc[0])
    bias_ci_low = float(stats.loc[stats["statistic"] == "bias_ci95_lower_n_per_mm", "value"].iloc[0])
    bias_ci_high = float(stats.loc[stats["statistic"] == "bias_ci95_upper_n_per_mm", "value"].iloc[0])
    axes[1, 0].axhspan(bias_ci_low, bias_ci_high, color="tab:red", alpha=0.10, label="95% CI for bias")
    axes[1, 0].axhline(bias, color="tab:red", ls="--", label=f"bias={bias:.2f}")
    axes[1, 0].axhline(loa_low, color="tab:red", ls=":", label=f"LoA={loa_low:.2f}, {loa_high:.2f}")
    axes[1, 0].axhline(loa_high, color="tab:red", ls=":")
    axes[1, 0].scatter(
        paired["level_mean_n_per_mm"],
        paired["right_minus_left_n_per_mm"],
        s=55,
        color="tab:green",
    )
    for _, row in paired.iterrows():
        axes[1, 0].annotate(
            row["level"],
            (row["level_mean_n_per_mm"], row["right_minus_left_n_per_mm"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1, 0].axhline(0.0, color="0.35", lw=0.8)
    axes[1, 0].set(xlabel="Left-Right mean stiffness [N/mm]", ylabel="Right - Left [N/mm]")
    axes[1, 0].set_title("Bland-Altman side agreement", loc="left", fontsize=10)
    axes[1, 0].legend(loc="upper left", fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    plot_cycle_distributions(axes[1, 1], data)
    fig.suptitle(
        "Left-Right agreement of intraoperative translational stiffness\n"
        "True-linear cycles only (R2 > 0.95); fit range = 22 N to min(67 N, cycle peak); 95% CIs",
        fontsize=13,
    )
    fig.savefig(OUT_FIG, dpi=180)
    plt.close(fig)
    print(f"saved {OUT_FIG}")
    print(f"saved {OUT_PAIRED_CSV}")
    print(f"saved {OUT_STATS_CSV}")


if __name__ == "__main__":
    main()
