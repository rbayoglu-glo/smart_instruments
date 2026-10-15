from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_l4l5_intact_prior_method as analysis
from intact_level_processing_config import OUTPUT_DIR
from lamina_spreader_lr import break_tracking_gaps
from plot_sync_check_all_levels_sides_stacked_true_raw_filtered_baseline_rampstart import (
    MAX_LOOKBACK_S,
    find_force_peaks,
    find_thoracic_sync_lag,
    ramp_start_from_peak,
)
from thoracic_level_processing_config import RAMPSTART_GROUPS, get_level_config


OUTPUT_DIR = Path(OUTPUT_DIR)


def configure(level: str) -> tuple[dict, dict[str, float]]:
    config = get_level_config(level)
    analysis.LEVEL_NAME = level
    analysis.OUTPUT_PREFIX = "intraop_"
    analysis.FORCE_DIR = config["force_dir"]
    analysis.TRACKING_DIR = config["tracking_dir"]
    analysis.TRIALS = config["trials"]
    analysis.OUT_DIR = config["output_dir"]
    analysis.PLOT_OUT_DIR = config["output_dir"]
    analysis.RELATIVE_TIME_MODE = True
    all_trials = analysis.TRIALS
    analysis.TRIALS, rejected = analysis.split_synchronizable_trials()
    for name, correlation in rejected.items():
        print(f"EXCLUDED {name}: force/theta sync correlation {correlation:.2f} below gate")
    return all_trials, rejected


def make_unsynced_diagnostic(
    trial_name: str,
    trial_config: dict,
    correlation: float,
) -> dict:
    tracking = analysis.compute_theta_from_sh_pose(
        analysis.TRACKING_DIR / trial_config["tracking_folder"], valid_only=False
    )
    force_t, _, force_filtered = analysis.load_force(
        analysis.FORCE_DIR / trial_config["force_file"]
    )
    theta_t = (tracking["t_true_est"] - tracking["t_true_est"].min()).dt.total_seconds().to_numpy()
    theta_raw = np.where(
        tracking["valid"].to_numpy(bool), tracking["theta_deg"].to_numpy(float), np.nan
    )
    theta_filtered = analysis.filter_theta(
        theta_raw, 1.0 / np.nanmedian(np.diff(theta_t))
    )
    lag, _ = find_thoracic_sync_lag(force_t, force_filtered, theta_t, theta_filtered)
    force_t_sync = force_t - lag
    force_baseline = analysis.compute_force_baseline(force_filtered)
    force_dynamic = force_filtered - force_baseline
    peaks = find_force_peaks(force_t_sync, force_dynamic, float(np.nanmax(force_dynamic)))

    force_feet = []
    previous_peak = -1
    for cycle_id, peak in enumerate(peaks, start=1):
        lo = max(
            int(np.searchsorted(force_t_sync, force_t_sync[peak] - MAX_LOOKBACK_S)),
            previous_peak + 1,
            0,
        )
        previous_peak = int(peak)
        foot = ramp_start_from_peak(force_dynamic, force_t_sync, int(peak), lo)
        if foot is not None:
            force_feet.append((cycle_id, int(foot[0])))

    previous_trials = analysis.TRIALS
    previous_gate = analysis.THORACIC_MIN_SYNC_CORRELATION
    analysis.TRIALS = {trial_name: trial_config}
    analysis.THORACIC_MIN_SYNC_CORRELATION = 0.0
    try:
        analysis.process_trial(trial_name)
    except RuntimeError as exc:
        processing_failure = str(exc)
    else:
        processing_failure = "processing completed, but synchronization is below the global acceptance gate"
    finally:
        analysis.TRIALS = previous_trials
        analysis.THORACIC_MIN_SYNC_CORRELATION = previous_gate

    return {
        "force_t_sync": force_t_sync,
        "force_dynamic": force_dynamic,
        "theta_t": theta_t,
        "theta_dynamic": theta_filtered,
        "theta_on_force": np.full(force_t_sync.size, np.nan),
        "force_baseline_N": force_baseline,
        "theta0_deg": np.nan,
        "candidates": [],
        "diagnostic_only": True,
        "diagnostic_correlation": correlation,
        "diagnostic_failure": processing_failure,
        "diagnostic_force_feet": force_feet,
    }


def plot_group(group_name: str, levels: tuple[str, ...]) -> Path:
    records = []
    for level in levels:
        all_trials, rejected = configure(level)
        for trial_name, trial_config in all_trials.items():
            if trial_name in rejected:
                info = make_unsynced_diagnostic(
                    trial_name, trial_config, rejected[trial_name]
                )
            else:
                info = analysis.process_trial(trial_name)[0]
            records.append((level, trial_name, info))

    output = OUTPUT_DIR / f"Step_04_intraop_Thoracic_{group_name}_rampstart.png"
    fig, axes = plt.subplots(
        len(records),
        1,
        figsize=(14.0, max(4.0, 2.5 * len(records))),
        squeeze=False,
    )
    axes = axes.ravel()
    for row, (level, trial_name, info) in enumerate(records):
        ax = axes[row]
        force_t = info["force_t_sync"]
        force_dynamic = info["force_dynamic"]
        theta_t = info["theta_t"]
        theta_dynamic = info["theta_dynamic"]
        theta_on_force = info["theta_on_force"]
        diagnostic_only = info.get("diagnostic_only", False)
        peaks = find_force_peaks(force_t, force_dynamic, float(np.nanmax(force_dynamic)))
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
        theta_label = "theta (unreferenced; sync unresolved)" if diagnostic_only else "theta"
        h_theta, = ax2.plot(theta_t_brk, theta_brk, color="tab:orange", lw=0.95, label=theta_label)
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
            label="force peak (diagnostic only)" if diagnostic_only else "invalid cycle (force peak)",
        )
        if diagnostic_only:
            h_force_feet = ax.scatter(
                [force_t[index] for _, index in info["diagnostic_force_feet"]],
                [force_dynamic[index] for _, index in info["diagnostic_force_feet"]],
                s=38, marker="^", color="tab:cyan", zorder=7,
                label="force-only ramp foot (not accepted)",
            )
            for cycle_id, index in info["diagnostic_force_feet"]:
                ax.annotate(
                    f"C{cycle_id} foot",
                    (force_t[index], force_dynamic[index]),
                    xytext=(0, -10), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color="tab:cyan", zorder=7,
                )
        else:
            h_invalid_theta = ax2.scatter(
                force_t[invalid_indices], theta_on_force[invalid_indices], s=34,
                marker="x", color="tab:purple", zorder=6,
                label="invalid cycle (theta)",
            )
        for candidate in info["candidates"]:
            index = int(candidate["i0"])
            ax2.annotate(
                f"C{candidate['cycle_id']}",
                (force_t[index], theta_on_force[index]),
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

        if diagnostic_only:
            row_title = (
                f"{level} | {trial_name.rsplit(' ', 1)[-1]} | UNSYNCED diagnostic only "
                f"(r={info['diagnostic_correlation']:.2f}); "
                f"{info['diagnostic_failure']}"
            )
        else:
            row_title = (
                f"{level} | {trial_name.rsplit(' ', 1)[-1]} | "
                f"{len(info['candidates'])} accepted, {len(invalid_peaks)} invalid cycles | "
                f"force0 {info['force_baseline_N']:+.2f} N, theta0 {info['theta0_deg']:+.2f} deg"
            )
        ax.set_title(row_title, loc="left", fontsize=9)
        if row == 0:
            if diagnostic_only:
                handles = [h_force, h_theta, h_invalid_force, h_force_feet]
            else:
                handles = [
                    h_force, h_theta, h_start_force, h_start_theta,
                    h_invalid_force, h_invalid_theta,
                ]
            ax.legend(handles, [handle.get_label() for handle in handles], loc="upper left", fontsize=7)
        if row == len(records) - 1:
            ax.set_xlabel("time [s]")

    subtitle = "Shared baseline, force/theta processing, and global cycle rules"
    if any(info.get("diagnostic_only") for _, _, info in records):
        subtitle += "; unsynced rows show force-only diagnostics and are excluded from stiffness"
    fig.suptitle(
        f"Step 04 thoracic group {group_name}: ramp-start cycle review\n{subtitle}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"saved {output}")
    return output


def main() -> None:
    for group_name, levels in RAMPSTART_GROUPS.items():
        plot_group(group_name, levels)


if __name__ == "__main__":
    main()
