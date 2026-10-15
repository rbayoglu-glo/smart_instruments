#%%
"""Plot intact L4-L5 FE moment versus direct Optotrack angle for three cycles."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
INPUT_CSV = OUT_DIR / "L4-L5_intact_FE_moment_vs_optotrack_angle.csv"
OUT_PNG = OUT_DIR / "L4-L5_intact_FE_moment_vs_angle_three_cycles_overlay.png"
CYCLE_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}


def main() -> None:
    data = pd.read_csv(INPUT_CSV)
    required = {"cycle", "moment_mx_nm", "angle", "optotrack_angle_measured"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input data sheet is missing columns: {sorted(missing)}")

    measured = data["optotrack_angle_measured"].astype(str).str.lower().eq("true")
    data = data[measured].copy()
    cycles = sorted(data["cycle"].unique())
    if cycles != [1, 2, 3]:
        raise RuntimeError(f"Expected direct angle data for cycles 1-3; found {cycles}.")

    fig, ax = plt.subplots(figsize=(8.6, 6.2), constrained_layout=True)
    for cycle in cycles:
        subset = data[data["cycle"] == cycle]
        angle = subset["angle"].to_numpy(float)
        moment = subset["moment_mx_nm"].to_numpy(float)
        color = CYCLE_COLORS[cycle]
        ax.plot(angle, moment, color=color, lw=1.4, label=f"Cycle {cycle} (n={len(subset)})")
        ax.scatter(angle[0], moment[0], color=color, marker="o", s=30, zorder=3)
        ax.scatter(angle[-1], moment[-1], color=color, marker="x", s=36, linewidth=1.2, zorder=3)

    ax.axvline(0.0, color="0.65", lw=0.8)
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.set_xlabel("Optotrack angle [deg]")
    ax.set_ylabel("Book1 Mx [N m]\n(flexion negative, extension positive)")
    ax.set_title(
        "L4-L5 intact: FE moment versus direct Optotrack angle\n"
        "Cycles 1-3 overlaid; raw `angle` values and exact Book1 Mx samples only",
        loc="left",
        fontsize=12,
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    try:
        fig.savefig(OUT_PNG, dpi=180)
    except OSError:
        fig.savefig(OUT_PNG.with_stem(f"{OUT_PNG.stem}_alt"), dpi=180)
    plt.close(fig)
    print(f"saved {OUT_PNG}")
    print("plotted rows per cycle: " + ", ".join(
        f"{cycle}={int((data['cycle'] == cycle).sum())}" for cycle in cycles
    ))


if __name__ == "__main__":
    main()