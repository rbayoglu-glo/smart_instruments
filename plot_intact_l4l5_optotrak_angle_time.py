#%%
"""Plot the direct first-row-subtracted Optotrack FE angle versus time."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
INPUT_XLSX = OUT_DIR / "L4-L5_intact_FE_raw_and_referenced.xlsx"
OUT_PNG = OUT_DIR / "L4-L5_intact_Optotrak_FE_angle_vs_time.png"


def main() -> None:
    data = pd.read_excel(INPUT_XLSX, sheet_name="angle_first_row_subtracted")
    required = {"optotrak_time_s", "fe_angle_first_row_subtracted_sign_corrected_deg"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Angle sheet is missing columns: {sorted(missing)}")

    valid = data["optotrak_time_s"].notna() & data["fe_angle_first_row_subtracted_sign_corrected_deg"].notna()
    data = data.loc[valid]

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(
        data["optotrak_time_s"],
        data["fe_angle_first_row_subtracted_sign_corrected_deg"],
        color="tab:blue",
        lw=0.8,
        label="Optotrack FE angle",
    )
    ax.axhline(0.0, color="0.55", lw=0.8, ls="--")
    ax.set_xlabel("Optotrack time [s]")
    ax.set_ylabel("sign-corrected FE angle change from first row [deg]")
    ax.set_title("L4-L5 intact: Optotrack flexion/extension angle versus time", loc="left")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    print(f"saved {OUT_PNG}")
    print(f"plotted rows={len(data)}, time range={data['optotrak_time_s'].min():.3f}-{data['optotrak_time_s'].max():.3f} s")
    print(
        f"angle range={data['fe_angle_first_row_subtracted_sign_corrected_deg'].min():+.3f}"
        f" to {data['fe_angle_first_row_subtracted_sign_corrected_deg'].max():+.3f} deg"
    )


if __name__ == "__main__":
    main()
