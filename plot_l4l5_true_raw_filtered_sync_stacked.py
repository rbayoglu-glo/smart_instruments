#%%
"""Stacked filtered force/theta synchronization check for L4-L5 sessions."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OLD_ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OLD_ANALYSIS_DIR))

from lamina_spreader_lr import break_tracking_gaps  # noqa: E402
from plot_sync_check_all_levels_sides_stacked_true_raw_filtered import (  # noqa: E402
    filter_force,
    filter_theta,
)
from plot_sync_check_all_levels_sides_stacked_true_raw_shpose_only import (  # noqa: E402
    compute_theta_from_sh_pose,
)
from plot_l4l5_true_raw_sync_stacked import (  # noqa: E402
    FORCE_COL,
    FORCE_SIGN,
    LAG_GRID_STEP_S,
    LAG_SEARCH_LIMIT_S,
    LAG_STEP_S,
    THETA_YLIM_DEG,
    TRACKING_DIR,
    TRIALS,
    gap_seconds,
    lag_for_plot,
)
from plot_sync_check_all_levels_sides_stacked_true_raw_filtered_baseline_rampstart import (  # noqa: E402
    THORACIC_MIN_SYNC_CORRELATION,
    find_thoracic_sync_lag,
)


BASE = Path(__file__).resolve().parent.parent
FORCE_DIR = BASE / "force_data"
OUT_DIR = Path(__file__).resolve().parent / "output"
FORCE_YLIM_N = (-25.0, 150.0)
TRACKING_START_OFFSETS_S = {}


def load_force(force_path: Path) -> tuple[np.ndarray, np.ndarray]:
    force_df = pd.read_csv(force_path)
    force_t = force_df["Time"].to_numpy(float) / 1e6
    force_dir = FORCE_SIGN * pd.to_numeric(force_df[FORCE_COL], errors="coerce").to_numpy(float)
    return force_t, force_dir


def main() -> None:
    fig, axes = plt.subplots(len(TRIALS), 1, figsize=(14.0, 3.0 * len(TRIALS)), squeeze=False)
    axes = axes.ravel()

    for index, (trial_name, config) in enumerate(TRIALS.items()):
        ax = axes[index]
        tracking = compute_theta_from_sh_pose(TRACKING_DIR / config["tracking_folder"], valid_only=False)
        force_t, force_dir = load_force(FORCE_DIR / config["force_file"])

        theta_t = (
            (tracking["t_true_est"] - tracking["t_true_est"].min()).dt.total_seconds().to_numpy()
            + float(TRACKING_START_OFFSETS_S.get(trial_name, 0.0))
        )
        valid = tracking["valid"].to_numpy(bool)
        theta_raw = np.where(valid, tracking["theta_deg"].to_numpy(float), np.nan)
        theta_fs = 1.0 / np.nanmedian(np.diff(theta_t))
        force_fs = 1.0 / np.nanmedian(np.diff(force_t))
        theta_filtered = filter_theta(theta_raw, theta_fs)
        force_filtered = filter_force(force_dir, force_fs)

        if RELATIVE_TIME_MODE:
            lag, score = find_thoracic_sync_lag(force_t, force_filtered, theta_t, theta_filtered)
            if score < THORACIC_MIN_SYNC_CORRELATION:
                raise RuntimeError(f"{trial_name}: force/theta synchronization unresolved (correlation {score:.2f})")
        else:
            lag, score = lag_for_plot(force_t, force_filtered, theta_t, theta_filtered)
        force_t_sync = force_t - lag

        h_force, = ax.plot(
            force_t_sync,
            force_filtered,
            lw=1.0,
            color="tab:blue",
            label="force (-1 x raw, MAD + low-pass), lag-synced",
        )
        ax.set_ylabel("force [N]", color="tab:blue")
        ax.set_ylim(*FORCE_YLIM_N)
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        theta_t_brk, theta_brk = break_tracking_gaps(theta_t, theta_filtered)
        h_theta, = ax2.plot(
            theta_t_brk,
            theta_brk,
            lw=0.95,
            color="tab:orange",
            label="theta (median + MAD + low-pass)",
        )
        ax2.axhline(0.0, color="tab:orange", lw=0.9, ls=":", alpha=0.8)
        ax2.set_ylabel("theta [deg]", color="tab:orange")
        ax2.set_ylim(*THETA_YLIM_DEG)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        n_rejected = int(np.count_nonzero(~valid))
        ax.set_title(
            f"{trial_name}   filtered SH-pose-only sync check "
            f"(lag {lag:+.2f} s, correlation {score:.3f}, {n_rejected} rejected frames, "
            f"{gap_seconds(theta_t):.0f} s with no SH frames)",
            fontsize=10,
        )
        if index == 0:
            ax.legend([h_force, h_theta], [h_force.get_label(), h_theta.get_label()], loc="upper left", fontsize=8)
        if index == len(axes) - 1:
            ax.set_xlabel("time [s]")

    fig.suptitle(
        "L4-L5 individual motion segments: filtered sync check (stacked, SH pose only)\n"
        "Force: direction-corrected + MAD despike + Butterworth low-pass. "
        "Theta: median filter + MAD despike + Butterworth low-pass; no theta0 subtraction.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = OUT_DIR / "L4-L5_true_raw_filtered_sync_stacked_shpose_only.png"
    try:
        fig.savefig(out_path, dpi=160)
    except OSError:
        out_path = out_path.with_stem(f"{out_path.stem}_alt")
        fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()