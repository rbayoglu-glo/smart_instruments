#%%
"""Export direct Book1 Mx and raw Optotrack angle for three intact FE cycles."""

from pathlib import Path

import numpy as np
import pandas as pd

from plot_intact_l4l5_flexion_extension_6dof import TEST_DIR, fe_cycle_boundaries


MOTOR_FILE = TEST_DIR / "Motor" / "Book1.xlsx"
OPTOTRAK_FILE = TEST_DIR / "Optotrak" / "Lamina spreader L4-L5_2026_09_17_095853_001_cal.xls"
OUT_FILE = Path(__file__).resolve().parent / "plot_outputs" / "L4-L5_intact_FE_moment_vs_optotrack_angle.csv"
MAX_PAIRING_ERROR_S = 0.005


def main() -> None:
    motor = pd.read_excel(MOTOR_FILE)
    motor_time = pd.to_numeric(motor["Time (s)"], errors="coerce").to_numpy(float)
    moment_mx = pd.to_numeric(motor["Mx (Nm)"], errors="coerce").to_numpy(float)
    motor_sample = np.arange(1, len(motor) + 1, dtype=int)
    valid_motor = np.isfinite(motor_time) & np.isfinite(moment_mx)
    motor_time = motor_time[valid_motor]
    moment_mx = moment_mx[valid_motor]
    motor_sample = motor_sample[valid_motor]

    optotrak = pd.read_csv(OPTOTRAK_FILE, sep="\t", usecols=range(8))
    opto_time = pd.to_numeric(optotrak["time"], errors="coerce").to_numpy(float)
    raw_angle = pd.to_numeric(optotrak["angle"], errors="coerce").to_numpy(float)
    opto_row = np.arange(2, len(optotrak) + 2, dtype=int)
    valid_opto = np.isfinite(opto_time) & np.isfinite(raw_angle)
    opto_time = opto_time[valid_opto]
    raw_angle = raw_angle[valid_opto]
    opto_row = opto_row[valid_opto]

    boundaries = fe_cycle_boundaries(motor_time, moment_mx)
    if len(boundaries) < 4:
        raise RuntimeError(f"Expected three complete FE cycles; found {len(boundaries) - 1}.")

    # Retain every recorded Book1 Mx row. The tail after the nominal third
    # positive-going crossing is part of the third-cycle settling phase.
    cycle = np.searchsorted(np.asarray(boundaries[1:3]), motor_time, side="right") + 1
    right = np.searchsorted(opto_time, motor_time, side="left").clip(1, len(opto_time) - 1)
    left = right - 1
    choose_left = np.abs(motor_time - opto_time[left]) <= np.abs(opto_time[right] - motor_time)
    nearest = np.where(choose_left, left, right)
    pairing_error = opto_time[nearest] - motor_time
    measured = np.abs(pairing_error) <= MAX_PAIRING_ERROR_S

    matched_angle = np.full(motor_time.shape, np.nan)
    matched_time = np.full(motor_time.shape, np.nan)
    matched_row = np.full(motor_time.shape, np.nan)
    matched_angle[measured] = raw_angle[nearest[measured]]
    matched_time[measured] = opto_time[nearest[measured]]
    matched_row[measured] = opto_row[nearest[measured]]

    out = pd.DataFrame({
        "cycle": cycle,
        "motor_sample_index": motor_sample,
        "motor_time_s": motor_time,
        "optotrack_source_row": pd.array(matched_row, dtype="Int64"),
        "optotrack_time_s": matched_time,
        "optotrack_pairing_error_s": np.where(measured, pairing_error, np.nan),
        "moment_mx_nm": moment_mx,
        "angle": matched_angle,
        "optotrack_angle_measured": measured,
    })
    OUT_FILE.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    print(f"saved {OUT_FILE}")
    print(f"Book1 rows={len(out)}, cycles={out.groupby('cycle').size().to_dict()}, "
          f"raw angle rows={int(measured.sum())}")


if __name__ == "__main__":
    main()