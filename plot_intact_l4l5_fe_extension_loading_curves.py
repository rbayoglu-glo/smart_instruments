#%%
"""Plot intact L4-L5 extension-loading moment curves for three cycles."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
INPUT_XLSX = OUT_DIR / "L4-L5_intact_FE_raw_and_referenced.xlsx"
OUT_PNG = OUT_DIR / "L4-L5_intact_FE_extension_loading_cycle1.png"
COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}


def main() -> None:
    data = pd.read_excel(INPUT_XLSX, sheet_name="angle_resampled_to_motor")
    required = {
        "opto_cycle",
        "motor_angle_deg",
        "moment_mx_nm",
        "fe_angle_first_row_subtracted_sign_corrected_resampled_deg",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Resampled sheet is missing columns: {sorted(missing)}")

    data = data[data["opto_cycle"].isin([1, 2, 3])].copy()
    if data.empty:
        raise RuntimeError("No Optotrack-defined cycles found.")

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.7), sharey=True, constrained_layout=True)
    panels = [
        (axes[0], "motor_angle_deg", "Motor Angle [deg]", "Mx vs Motor Angle"),
        (
            axes[1],
            "fe_angle_first_row_subtracted_sign_corrected_resampled_deg",
            "Optotrak sign-corrected FE angle [deg]",
            "Mx vs Optotrak FE angle",
        ),
    ]
    counts = {}

    for ax, x_column, xlabel, title in panels:
        for cycle in [1]:
            subset = data[data["opto_cycle"] == cycle].copy()
            subset = subset[np.isfinite(subset[x_column]) & np.isfinite(subset["moment_mx_nm"])]
            if subset.empty:
                continue

            # Extension loading ends at the maximum sign-corrected Optotrak FE angle.
            peak_position = int(np.argmax(subset["fe_angle_first_row_subtracted_sign_corrected_resampled_deg"].to_numpy(float)))
            subset = subset.iloc[: peak_position + 1]
            x = subset[x_column].to_numpy(float)
            y = subset["moment_mx_nm"].to_numpy(float)
            color = COLORS[cycle]
            ax.plot(x, y, color=color, lw=1.35, label=f"Cycle {cycle} (n={len(x)})")
            ax.scatter(x[0], y[0], color=color, marker="o", s=28, zorder=3)
            ax.scatter(x[-1], y[-1], color=color, marker="x", s=36, linewidth=1.1, zorder=3)
            counts.setdefault(cycle, []).append(len(x))

        ax.axvline(0.0, color="0.65", lw=0.8)
        ax.axhline(0.0, color="0.65", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    axes[0].set_ylabel("Book1 Mx [N m]\n(flexion negative, extension positive)")
    fig.suptitle(
        "L4-L5 intact: cycle-1 extension-loading moment curve\n"
        "Optotrack-defined cycle; exact Book1 Mx; no regression or average curve",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)
    print(f"saved {OUT_PNG}")
    print(f"extension rows per cycle [Motor, Optotrak]: {counts}")


if __name__ == "__main__":
    main()
