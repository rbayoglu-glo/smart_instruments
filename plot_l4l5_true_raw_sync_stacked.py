#%%
"""Stacked true-raw force/theta synchronization check for L4-L5 sessions."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OLD_ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OLD_ANALYSIS_DIR))

from lamina_spreader_lr import FORCE_COL, FORCE_SIGN  # noqa: E402
from plot_sync_check_all_levels_sides_stacked_true_raw_shpose_only import (  # noqa: E402
    compute_theta_from_sh_pose,
)
from plot_sync_check_all_levels_sides_stacked_true_raw_filtered_baseline_rampstart import (  # noqa: E402
    THORACIC_MIN_SYNC_CORRELATION,
    find_thoracic_sync_lag,
)
from plot_sync_check_all_levels_sides_stacked_true_raw_filtered import (  # noqa: E402
    filter_force,
    filter_theta,
)


BASE = Path(__file__).resolve().parent.parent
TRACKING_DIR = BASE / "tracking_data"
FORCE_DIR = BASE / "force_data"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

TRIALS = {
    "L4-L5 Left": {
        "force_file": "Session #80.csv",
        "tracking_folder": "2026-12-17_[01-56-11]",
    },
    "L4-L5 Right": {
        "force_file": "Session #81.csv",
        "tracking_folder": "2026-12-17_[02-05-40]",
    },
}

FORCE_YLIM_N = (-25.0, 150.0)
THETA_YLIM_DEG = (-4.0, 7.0)
LAG_SEARCH_LIMIT_S = 200.0
LAG_STEP_S = 0.5
LAG_GRID_STEP_S = 0.1
LAG_FORCE_MIN_N = -20.0
LAG_FORCE_MAX_N = 100.0
TRACKING_START_OFFSETS_S = {}


def lag_for_plot(force_t: np.ndarray, force: np.ndarray, theta_t: np.ndarray, theta: np.ndarray) -> tuple[float, float]:
    """Find a relative lag; robust clipping is used only for lag scoring."""
    grid = np.arange(0.0, min(force_t[-1], theta_t[-1]), LAG_GRID_STEP_S)
    force_score = force.copy()
    force_score[(force_score < LAG_FORCE_MIN_N) | (force_score > LAG_FORCE_MAX_N)] = np.nan
    force_score = pd.Series(force_score).interpolate(limit_direction="both").to_numpy(float)
    force_grid = np.interp(grid, force_t, force_score)
    theta_grid = np.interp(grid, theta_t, np.nan_to_num(theta, nan=np.nanmedian(theta)))
    force_grid = (force_grid - np.mean(force_grid)) / (np.std(force_grid) + 1e-9)
    theta_grid = (theta_grid - np.mean(theta_grid)) / (np.std(theta_grid) + 1e-9)

    best_score = -np.inf
    best_lag = 0.0
    for lag in np.arange(-LAG_SEARCH_LIMIT_S, LAG_SEARCH_LIMIT_S + LAG_STEP_S, LAG_STEP_S):
        shifted_force = np.interp(grid + lag, grid, force_grid, left=np.nan, right=np.nan)
        valid = np.isfinite(shifted_force)
        if valid.sum() < 20:
            continue
        score = float(np.corrcoef(theta_grid[valid], shifted_force[valid])[0, 1])
        if score > best_score:
            best_score = score
            best_lag = float(lag)
    return best_lag, best_score


def load_force(force_path: Path) -> tuple[np.ndarray, np.ndarray]:
    force_df = pd.read_csv(force_path)
    force_t = force_df["Time"].to_numpy(float) / 1e6
    force_dir = FORCE_SIGN * pd.to_numeric(force_df[FORCE_COL], errors="coerce").to_numpy(float)
    return force_t, force_dir


def gap_seconds(t: np.ndarray) -> float:
    if t.size < 2:
        return 0.0
    dt = float(np.nanmedian(np.diff(t)))
    return float(np.sum(np.maximum(np.diff(t) - 3.0 * dt, 0.0)))


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
        theta = tracking["theta_deg"].to_numpy(float)
        valid = tracking["valid"].to_numpy(bool)
        if RELATIVE_TIME_MODE:
            # same filtered signals as the analysis path so every step shares one lag
            theta_filtered = filter_theta(
                np.where(valid, theta, np.nan), 1.0 / np.nanmedian(np.diff(theta_t))
            )
            force_filtered = filter_force(force_dir, 1.0 / np.nanmedian(np.diff(force_t)))
            lag, score = find_thoracic_sync_lag(force_t, force_filtered, theta_t, theta_filtered)
            if score < THORACIC_MIN_SYNC_CORRELATION:
                raise RuntimeError(f"{trial_name}: force/theta synchronization unresolved (correlation {score:.2f})")
        else:
            lag, score = lag_for_plot(force_t, force_dir, theta_t[valid], theta[valid])
        force_t_sync = force_t - lag

        theta_valid = np.where(valid, theta, np.nan)
        theta_rejected = np.where(valid, np.nan, theta)
        h_force, = ax.plot(force_t_sync, force_dir, lw=0.9, color="tab:blue", label="force (-1 x raw), lag-synced")
        ax.set_ylabel("force [N]", color="tab:blue")
        ax.set_ylim(*FORCE_YLIM_N)
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        h_theta, = ax2.plot(theta_t, theta_valid, lw=0.9, color="tab:orange", label="theta valid (SH pose)")
        h_rejected, = ax2.plot(theta_t, theta_rejected, lw=0.9, color="tab:red", label="theta rejected (SH pose)")
        ax2.axhline(0.0, color="tab:orange", lw=0.9, ls=":", alpha=0.8)
        ax2.set_ylabel("theta [deg]", color="tab:orange")
        ax2.set_ylim(*THETA_YLIM_DEG)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        n_rejected = int(np.count_nonzero(~valid))
        ax.set_title(
            f"{trial_name}   SH-pose-only raw theta + force-direction corrected sync check "
            f"(lag {lag:+.2f} s, correlation {score:.3f}, {n_rejected} rejected frames, "
            f"{gap_seconds(theta_t):.0f} s with no SH frames)",
            fontsize=10,
        )
        if index == 0:
            ax.legend([h_force, h_theta, h_rejected], [h_force.get_label(), h_theta.get_label(), h_rejected.get_label()],
                      loc="upper left", fontsize=8)
        if index == len(axes) - 1:
            ax.set_xlabel("time [s]")

    fig.suptitle(
        "L4-L5 individual motion segments: true-raw theta sync check (stacked, SH pose only)\n"
        "Force direction corrected (-1 x raw) + relative lag refinement; theta is raw theta_deg with no theta0 subtraction.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = OUT_DIR / "L4-L5_true_raw_sync_stacked_shpose_only.png"
    try:
        fig.savefig(out_path, dpi=160)
    except OSError:
        out_path = out_path.with_stem(f"{out_path.stem}_alt")
        fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()