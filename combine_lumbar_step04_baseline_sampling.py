from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_l4l5_intact_prior_method as analysis
from intact_level_processing_config import OUTPUT_DIR, get_level_config
from lamina_spreader_lr import break_tracking_gaps
from plot_sync_check_all_levels_sides_stacked_true_raw_filtered_baseline_rampstart import find_force_peaks

LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5")
OUT = OUTPUT_DIR / "Step_04_intraop_Lumbar_all_levels_rampstart.png"


def collect_results() -> list[tuple[str, str, dict]]:
    results = []
    for level in LEVELS:
        config = get_level_config(level)
        analysis.FORCE_DIR = config["force_dir"]
        analysis.TRACKING_DIR = config["tracking_dir"]
        analysis.TRIALS = config["trials"]
        for trial_name in analysis.TRIALS:
            info, _ = analysis.process_trial(trial_name)
            results.append((level, trial_name, info))
    return results


def main() -> None:
    results = collect_results()
    fig, axes = plt.subplots(len(results), 1, figsize=(14.0, 18.0), squeeze=False)
    axes = axes.ravel()

    for ax, (level, trial_name, info) in zip(axes, results):
        force_t = info["force_t_sync"]
        force_dynamic = info["force_dynamic"]
        theta_t = info["theta_t"]
        theta_dynamic = info["theta_dynamic"]
        theta_on_force = info["theta_on_force"]
        peaks = find_force_peaks(
            force_t, force_dynamic, float(np.nanmax(force_dynamic))
        )
        accepted_ids = {int(candidate["cycle_id"]) for candidate in info["candidates"]}
        invalid_peaks = [
            (cycle_id, int(peak))
            for cycle_id, peak in enumerate(peaks, start=1)
            if cycle_id not in accepted_ids
        ]

        h_force, = ax.plot(force_t, force_dynamic, color="tab:blue", lw=1.0, label="force")
        ax.set_ylabel("force [N]", color="tab:blue")
        ax.set_ylim(-25.0, 150.0)
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        theta_t_brk, theta_brk = break_tracking_gaps(theta_t, theta_dynamic)
        h_theta, = ax2.plot(theta_t_brk, theta_brk, color="tab:orange", lw=0.95, label="theta")
        ax2.axhline(0.0, color="tab:orange", lw=0.9, ls="--", alpha=0.8)
        ax2.set_ylabel("theta [deg]", color="tab:orange")
        ax2.set_ylim(-4.0, 7.0)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        starts = np.asarray([candidate["i0"] for candidate in info["candidates"]], dtype=int)
        h_start_force = ax.scatter(
            force_t[starts], force_dynamic[starts], s=28, marker="o",
            facecolor="white", edgecolor="tab:red", zorder=6,
            label="ramp start (force)",
        )
        h_start_theta = ax2.scatter(
            force_t[starts], theta_on_force[starts], s=28, marker="D",
            facecolor="white", edgecolor="tab:red", zorder=6,
            label="ramp start (theta)",
        )
        invalid_indices = np.asarray([peak for _, peak in invalid_peaks], dtype=int)
        h_invalid_force = ax.scatter(
            force_t[invalid_indices], force_dynamic[invalid_indices], s=34,
            marker="x", color="tab:purple", zorder=6,
            label="invalid cycle (force peak)",
        )
        h_invalid_theta = ax2.scatter(
            force_t[invalid_indices], theta_on_force[invalid_indices], s=34,
            marker="x", color="tab:purple", zorder=6,
            label="invalid cycle (theta)",
        )
        for candidate in info["candidates"]:
            idx = int(candidate["i0"])
            ax2.annotate(
                f"C{candidate['cycle_id']}",
                (force_t[idx], theta_on_force[idx]),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
                color="tab:red", zorder=7,
            )
        for cycle_id, peak in invalid_peaks:
            ax.annotate(
                f"C{cycle_id}",
                (force_t[peak], force_dynamic[peak]),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
                color="tab:purple", zorder=7,
            )

        ax.set_title(
            f"{level} | {trial_name.rsplit(' ', 1)[-1]} | "
            f"{len(info['candidates'])} accepted, {len(invalid_peaks)} invalid cycles",
            loc="left", fontsize=9,
        )
        if ax is axes[0]:
            handles = [
                h_force, h_theta, h_start_force, h_start_theta,
                h_invalid_force, h_invalid_theta,
            ]
            ax.legend(handles, [handle.get_label() for handle in handles], loc="upper left", fontsize=7)

    axes[-1].set_xlabel("time [s]")
    fig.suptitle(
        "Step 04: lumbar levels ramp-start sync checks\n"
        "Shared force/theta processing and cycle rules for all levels and sides",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
