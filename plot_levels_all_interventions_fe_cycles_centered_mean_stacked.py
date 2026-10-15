#%%
"""Plot complete-cycle FE curves for the requested spinal levels."""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from plot_intact_l4l5_fe_loading_unloading_mean_by_cycle import centered_mean_curve
from plot_l4l5_all_interventions_lb_ar_cycle3_centered_mean import align_angle_to_motor


LAB_DIR = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\UTN24_26016")
LEVEL_ROOT = LAB_DIR / "segment" / "6dof"
OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)
LEVELS = ("L2-L3", "T12-L1", "T10-T11")
COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:cyan",
]
MEDIAN_WINDOW = 5
PEAK_PROMINENCE_DEG = 0.75
PEAK_MIN_SEPARATION_S = 5.0


def numeric_book_number(path: Path) -> int:
    return int(re.search(r"(\d+)", path.stem).group(1))


def numeric_opto_number(path: Path) -> int:
    return int(re.search(r"_(\d+)_cal", path.name).group(1))


def intervention_folders(level_dir: Path) -> list[Path]:
    # Skips interventions not performed for a specimen (no motor workbooks).
    return sorted(
        [
            path
            for path in level_dir.iterdir()
            if path.is_dir() and path.name[:1].isdigit() and any((path / "Motor").glob("Book*.xlsx"))
        ],
        key=lambda path: int(path.name.split(".")[0]),
    )


def load_motor(folder: Path) -> tuple[np.ndarray, np.ndarray, str]:
    candidates = []
    for path in sorted((folder / "Motor").glob("Book*.xlsx"), key=numeric_book_number):
        data = pd.read_excel(path)
        time_s = pd.to_numeric(data["Time (s)"], errors="coerce").to_numpy(float)
        motor_angle = pd.to_numeric(data["Angle (deg)"], errors="coerce").to_numpy(float)
        moment = pd.to_numeric(data["Mx (Nm)"], errors="coerce").to_numpy(float)
        valid = np.isfinite(time_s) & np.isfinite(motor_angle) & np.isfinite(moment)
        time_s = time_s[valid]
        motor_angle = motor_angle[valid]
        moment = moment[valid]
        candidates.append((float(np.ptp(moment)), path, time_s, motor_angle, moment))
    if not candidates:
        raise RuntimeError(f"No Motor workbooks found in {folder}.")
    _, path, time_s, motor_angle, moment = max(candidates, key=lambda candidate: candidate[0])
    return time_s, motor_angle, moment, path.name


def load_optotrak(folder: Path) -> tuple[np.ndarray, np.ndarray, str]:
    candidates = []
    for path in sorted((folder / "Optotrak").glob("*.xls"), key=numeric_opto_number):
        data = pd.read_csv(path, sep="\t", index_col=False, usecols=range(7))
        time_s = pd.to_numeric(data["time"], errors="coerce").to_numpy(float)
        raw_angle = pd.to_numeric(data["Relative2(RB1, RB8) deg [Ry]"], errors="coerce").to_numpy(float)
        valid = np.isfinite(time_s) & np.isfinite(raw_angle)
        time_s = time_s[valid]
        angle = -(raw_angle[valid] - raw_angle[valid][0])
        candidates.append((float(np.ptp(angle)), path, time_s, angle))
    if not candidates:
        raise RuntimeError(f"No Optotrak files found in {folder}.")
    _, path, time_s, angle = max(candidates, key=lambda candidate: candidate[0])
    return time_s, angle, path.name


def detect_fe_cycle_boundaries(time_s: np.ndarray, angle: np.ndarray) -> list[float]:
    valid = np.isfinite(angle)
    time_s = time_s[valid]
    angle = angle[valid]
    if time_s.size < 3:
        raise RuntimeError("FE trace has too few valid samples.")
    smooth = pd.Series(angle).rolling(
        MEDIAN_WINDOW,
        center=True,
        min_periods=1,
    ).median().to_numpy(float)
    dt = float(np.nanmedian(np.diff(time_s)))
    troughs, _ = find_peaks(
        -smooth,
        prominence=PEAK_PROMINENCE_DEG,
        distance=max(1, int(round(PEAK_MIN_SEPARATION_S / dt))),
    )
    if troughs.size == 0:
        raise RuntimeError("No complete FE troughs found.")

    boundaries = [float(time_s[0])]
    for trough in troughs[:3]:
        indices = np.arange(time_s.size - 1)
        crossings = indices[
            (indices > trough)
            & (smooth[:-1] < 0.0)
            & (smooth[1:] >= 0.0)
        ]
        if crossings.size:
            index = int(crossings[0])
            fraction = -smooth[index] / (smooth[index + 1] - smooth[index])
            boundaries.append(float(time_s[index] + fraction * (time_s[index + 1] - time_s[index])))
        elif trough == troughs[min(2, troughs.size - 1)] and abs(float(smooth[-1])) <= 0.5:
            boundaries.append(float(time_s[-1]))
        else:
            break
    if len(boundaries) < 2:
        raise RuntimeError("Could not define a complete FE cycle.")
    return boundaries


def load_cases(level: str) -> list[dict]:
    cases = []
    level_dir = LEVEL_ROOT / level
    for color, folder in zip(COLORS, intervention_folders(level_dir)):
        name = re.sub(r"^\d+\.\s*", "", folder.name)
        motor_time, motor_angle, moment, motor_file = load_motor(folder)
        opto_time, opto_angle, opto_file = load_optotrak(folder)
        boundaries_for_lag = max(1.0, (motor_time[-1] - motor_time[0]) / 3.0)
        aligned_angle, aligned_valid, lag, correlation = align_angle_to_motor(
            motor_time,
            motor_angle,
            opto_time,
            opto_angle,
            boundaries_for_lag,
        )
        aligned_valid &= np.isfinite(aligned_angle)
        boundaries = detect_fe_cycle_boundaries(motor_time[aligned_valid], aligned_angle[aligned_valid])
        available_cycles = []
        for cycle_number in range(1, len(boundaries)):
            cycle_start, cycle_end = boundaries[cycle_number - 1:cycle_number + 1]
            keep = aligned_valid & (motor_time >= cycle_start) & (motor_time < cycle_end)
            complete_coverage = (
                cycle_start >= opto_time[0] + lag
                and cycle_end <= opto_time[-1] + lag
                and int(keep.sum()) >= 20
            )
            if complete_coverage:
                available_cycles.append(cycle_number)
        cases.append({
            "name": name,
            "color": color,
            "motor_time": motor_time,
            "moment": moment,
            "opto_time": opto_time,
            "aligned_angle": aligned_angle,
            "aligned_valid": aligned_valid,
            "boundaries": boundaries,
            "available_cycles": available_cycles,
            "lag": lag,
            "correlation": correlation,
            "motor_file": motor_file,
            "opto_file": opto_file,
        })
    return cases


def common_complete_cycles(cases: list[dict]) -> tuple[int, ...]:
    common = set(cases[0]["available_cycles"])
    for case in cases[1:]:
        common.intersection_update(case["available_cycles"])
    return tuple(sorted(common))


def plot_level(level: str) -> None:
    cases = load_cases(level)
    cycle_numbers = common_complete_cycles(cases)
    if not cycle_numbers:
        raise RuntimeError(f"{level}: no complete FE cycle is common to all interventions.")

    cycle_tag = "_".join(str(cycle) for cycle in cycle_numbers)
    out_stem = f"{level}_all_interventions_FE_cycles_{cycle_tag}_centered_mean_stacked"
    out_png = OUT_DIR / f"{out_stem}.png"
    out_csv = OUT_DIR / f"{out_stem}.csv"
    out_range_csv = OUT_DIR / f"{out_stem}_angle_ranges.csv"
    fig, axes = plt.subplots(
        len(cycle_numbers),
        1,
        figsize=(8.5, 4.5 * len(cycle_numbers)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    mean_rows = []
    range_rows = []

    for cycle_index, (cycle_number, ax) in enumerate(zip(cycle_numbers, axes)):
        for case in cases:
            cycle_start, cycle_end = case["boundaries"][cycle_number - 1:cycle_number + 1]
            keep = (
                case["aligned_valid"]
                & (case["motor_time"] >= cycle_start)
                & (case["motor_time"] < cycle_end)
            )
            cycle_angle = case["aligned_angle"][keep]
            cycle_moment = case["moment"][keep]
            if len(cycle_angle) < 20:
                raise RuntimeError(
                    f"{level} {case['name']} cycle {cycle_number}: too few samples."
                )

            mean_angle, mean_moment, _, _ = centered_mean_curve(cycle_angle, cycle_moment)
            ax.plot(
                mean_angle,
                mean_moment,
                color=case["color"],
                lw=2.0,
                label=case["name"],
            )
            for phase, (angle_value, moment_value) in enumerate(zip(mean_angle, mean_moment)):
                mean_rows.append({
                    "level": level,
                    "axis": "FE",
                    "cycle": cycle_number,
                    "intervention": case["name"],
                    "phase_index": phase,
                    "angle_mean_deg": angle_value,
                    "moment_mx_mean_nm": moment_value,
                    "motor_angle_lag_s": case["lag"],
                    "motor_angle_lag_correlation": case["correlation"],
                    "motor_file": case["motor_file"],
                    "optotrak_file": case["opto_file"],
                    "n_cycle_rows": len(cycle_angle),
                })
            range_rows.append({
                "level": level,
                "axis": "FE",
                "cycle": cycle_number,
                "intervention": case["name"],
                "cycle_extracted_angle_min_deg": float(np.min(cycle_angle)),
                "cycle_extracted_angle_max_deg": float(np.max(cycle_angle)),
                "centered_mean_curve_angle_min_deg": float(np.min(mean_angle)),
                "centered_mean_curve_angle_max_deg": float(np.max(mean_angle)),
            })

        ax.axvline(0.0, color="0.65", lw=0.8)
        ax.axhline(0.0, color="0.65", lw=0.8)
        ax.set_title(f"Cycle {cycle_number}", loc="left", fontsize=10)
        ax.set_ylabel("Mx [Nm]")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.grid(alpha=0.25)
        ax.text(
            8.0,
            1.0,
            "Extension",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
        )
        ax.text(
            -1.5,
            -9.0,
            "Flexion",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
        )
        if cycle_index == 0:
            ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("FE angle change, ΔRy [deg]")
    fig.suptitle(f"{level} Flexion/Extension (complete synchronized cycles)", fontsize=12)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    pd.DataFrame(mean_rows).to_csv(out_csv, index=False)
    pd.DataFrame(range_rows).to_csv(out_range_csv, index=False)
    print(f"saved {out_png}")
    print(f"saved {out_csv}")
    print(f"saved {out_range_csv}")
    print(f"{level} FE: cycles={list(cycle_numbers)}")
    for case in cases:
        print(
            f"  {case['name']}: motor={case['motor_file']}, optotrak={case['opto_file'][-7:]}, "
            f"lag={case['lag']:+.2f}s, r={case['correlation']:.3f}, "
            f"available_cycles={case['available_cycles']}"
        )


def main() -> None:
    for level in LEVELS:
        plot_level(level)


if __name__ == "__main__":
    main()
