#%%
"""Plot L4-L5 intact cycle-3 LB and AR moment versus Optotrack angle."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

from plot_intact_l4l5_flexion_extension_6dof import (
    CORRELATION_GRID_STEP_S,
    CYCLE_NUMBER,
    LAB_DIR,
    LAG_LIMIT_S,
    LAG_STEP_S,
    OUT_DIR,
    OPTOTRAK_TIME_COL,
    TEST_DIR,
    fe_cycle_boundaries,
    estimate_lag,
)


MOTOR_TIME_COL = "Time (s)"
AXES = {
    "LB": {
        "motor_file": "Book2.xlsx",
        "moment_col": "My (Nm)",
        "opto_file": "Lamina spreader L4-L5_2026_09_17_095853_002_cal.xls",
        "angle_col": "Relative2(RB1, RB8) deg [Rz]",
        "moment_label": "My moment [N·m] (left positive, right negative)",
        "angle_label": "LB angle change, Optotrack ΔRz [deg]",
    },
    "AR": {
        "motor_file": "Book3.xlsx",
        "moment_col": "Mz (Nm)",
        "opto_file": "Lamina spreader L4-L5_2026_09_17_095853_003_cal.xls",
        "angle_col": "Relative2(RB1, RB8) deg [Rx]",
        "moment_label": "Mz moment [N·m] (left positive, right negative)",
        "angle_label": "AR angle change, Optotrack ΔRx [deg]",
    },
}


def load_motor(config: dict) -> tuple[np.ndarray, np.ndarray]:
    motor = pd.read_excel(TEST_DIR / "Motor" / config["motor_file"])
    time_s = pd.to_numeric(motor[MOTOR_TIME_COL], errors="coerce").to_numpy(float)
    moment_nm = pd.to_numeric(motor[config["moment_col"]], errors="coerce").to_numpy(float)
    valid = np.isfinite(time_s) & np.isfinite(moment_nm)
    return time_s[valid], moment_nm[valid]


def load_optotrak(config: dict) -> tuple[np.ndarray, np.ndarray, float]:
    path = TEST_DIR / "Optotrak" / config["opto_file"]
    # Optotrack exports are tab-delimited text with a trailing empty field.
    opto = pd.read_csv(path, sep="\t", usecols=range(7))
    time_s = pd.to_numeric(opto[OPTOTRAK_TIME_COL], errors="coerce").to_numpy(float)
    angle_raw = pd.to_numeric(opto[config["angle_col"]], errors="coerce").to_numpy(float)
    valid = np.isfinite(time_s) & np.isfinite(angle_raw)
    time_s, angle_raw = time_s[valid], angle_raw[valid]
    reference_deg = float(angle_raw[0])
    return time_s, angle_raw - reference_deg, reference_deg


def plot_cycle3(axis_name: str, config: dict) -> None:
    motor_time_s, moment_nm = load_motor(config)
    opto_time_s, angle_delta_deg, angle_reference_deg = load_optotrak(config)
    lag_s, lag_corr = estimate_lag(motor_time_s, moment_nm, opto_time_s, angle_delta_deg)

    boundaries = fe_cycle_boundaries(motor_time_s, moment_nm)
    if len(boundaries) <= CYCLE_NUMBER:
        raise RuntimeError(
            f"{axis_name}: found only {len(boundaries) - 1} complete moment cycles; "
            f"cannot isolate cycle {CYCLE_NUMBER}."
        )
    cycle_start_s = boundaries[CYCLE_NUMBER - 1]
    cycle_end_s = boundaries[CYCLE_NUMBER]

    # The lag convention pairs motor(t) with Optotrack angle(t - lag).
    opto_start_motor_s = opto_time_s[0] + lag_s
    opto_end_motor_s = opto_time_s[-1] + lag_s
    paired_start_s = max(cycle_start_s, opto_start_motor_s)
    paired_end_s = min(cycle_end_s, opto_end_motor_s)
    if paired_end_s <= paired_start_s:
        raise RuntimeError(f"{axis_name}: cycle {CYCLE_NUMBER} has no synchronized Optotrack overlap.")

    inner_times = motor_time_s[(motor_time_s > paired_start_s) & (motor_time_s < paired_end_s)]
    paired_time_s = np.concatenate(([paired_start_s], inner_times, [paired_end_s]))
    paired_moment_nm = np.interp(paired_time_s, motor_time_s, moment_nm)
    paired_opto_time_s = paired_time_s - lag_s
    paired_angle_deg = np.interp(paired_opto_time_s, opto_time_s, angle_delta_deg)
    paired = pd.DataFrame({
        "cycle": CYCLE_NUMBER,
        "motor_time_s": paired_time_s,
        "optotrack_time_s": paired_opto_time_s,
        "moment_nm": paired_moment_nm,
        "angle_change_deg": paired_angle_deg,
    })

    cycle_duration_s = cycle_end_s - cycle_start_s
    paired_duration_s = paired_end_s - paired_start_s
    coverage = paired_duration_s / cycle_duration_s
    stem = f"L4-L5_intact_{axis_name}_cycle{CYCLE_NUMBER}_moment_vs_optotrack_angle"
    out_csv = OUT_DIR / f"{stem}.csv"
    paired.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    points = np.column_stack([paired_angle_deg, paired_moment_nm])
    if points.shape[0] > 1:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        collection = LineCollection(segments, cmap="viridis", linewidth=1.5, alpha=0.75)
        collection.set_array(paired_time_s[:-1])
        ax.add_collection(collection)
        colorbar = fig.colorbar(collection, ax=ax)
        colorbar.set_label("elapsed motor time [s]")
    ax.scatter(paired_angle_deg, paired_moment_nm, c="0.2", s=8, alpha=0.35, zorder=2)
    ax.scatter(paired_angle_deg[0], paired_moment_nm[0], color="tab:green", s=38, zorder=3, label="start")
    ax.scatter(paired_angle_deg[-1], paired_moment_nm[-1], color="tab:red", s=38, zorder=3, label="end")
    ax.axvline(0.0, color="0.65", lw=0.8)
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.set_xlabel(config["angle_label"])
    ax.set_ylabel(config["moment_label"])
    ax.set_title(
        f"L4-L5 intact: {axis_name} cycle {CYCLE_NUMBER}, moment vs Optotrack angle\n"
        f"First-row angle reference={angle_reference_deg:+.3f}°; motor Angle unused; "
        f"lag={lag_s:+.2f} s, r={lag_corr:.3f}; Optotrack coverage={coverage:.0%}",
        loc="left",
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    out_png = OUT_DIR / f"{stem}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(
        f"{axis_name} cycle {CYCLE_NUMBER}: motor interval {cycle_start_s:.3f}-{cycle_end_s:.3f} s; "
        f"lag={lag_s:+.2f} s, r={lag_corr:.3f}, coverage={coverage:.1%}, paired rows={len(paired)}"
    )
    print(f"saved {out_png}")
    print(f"saved {out_csv}")


def main() -> None:
    for axis_name, config in AXES.items():
        plot_cycle3(axis_name, config)


if __name__ == "__main__":
    main()