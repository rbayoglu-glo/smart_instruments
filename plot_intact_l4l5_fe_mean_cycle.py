#%%
"""Plot raw complete FE cycles with a phase-normalized mean trajectory."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
INPUT_XLSX = OUT_DIR / "L4-L5_intact_FE_raw_and_referenced.xlsx"
OUT_PNG = OUT_DIR / "L4-L5_intact_FE_mean_cycle_motor_vs_optotrak.png"
OUT_CSV = OUT_DIR / "L4-L5_intact_FE_mean_cycle.csv"
CYCLES = [1, 2, 3]
MEAN_POINTS = 300
COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}


def phase_resample(subset: pd.DataFrame, x_column: str) -> tuple[np.ndarray, np.ndarray]:
    phase = np.linspace(0.0, 1.0, len(subset))
    target = np.linspace(0.0, 1.0, MEAN_POINTS)
    x = subset[x_column].to_numpy(float)
    y = subset["moment_mx_nm"].to_numpy(float)
    return np.interp(target, phase, x), np.interp(target, phase, y)


def main() -> None:
    data = pd.read_excel(INPUT_XLSX, sheet_name="angle_aligned_to_motor")
    data = data[data["full_opto_cycle"].isin(CYCLES) & data["optotrack_angle_measured"]].copy()
    if data.empty:
        raise RuntimeError("No aligned complete-cycle data found.")

    panels = [
        ("motor_angle_deg", "Motor Angle [deg]", "Mx vs Motor Angle"),
        ("fe_angle_sign_corrected_deg", "Optotrak sign-corrected FE angle [deg]", "Mx vs Optotrak FE angle"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.7), sharey=True, constrained_layout=True)

    for col, (x_column, xlabel, title) in enumerate(panels):
        ax = axes[col]
        for cycle in CYCLES:
            subset = data[data["full_opto_cycle"] == cycle].sort_values("motor_time_s")
            x = subset[x_column].to_numpy(float)
            y = subset["moment_mx_nm"].to_numpy(float)
            ax.plot(x, y, color=COLORS[cycle], lw=1.0, alpha=0.55, label=f"cycle {cycle}")
            ax.scatter(x[0], y[0], color=COLORS[cycle], marker="o", s=24, zorder=3)
            ax.scatter(x[-1], y[-1], color=COLORS[cycle], marker="x", s=32, linewidth=1.0, zorder=3)
        ax.axvline(0.0, color="0.65", lw=0.8)
        ax.axhline(0.0, color="0.65", lw=0.8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)

    means = {}
    for x_column, _, _ in panels:
        x_curves = []
        y_curves = []
        for cycle in CYCLES:
            subset = data[data["full_opto_cycle"] == cycle].sort_values("motor_time_s")
            x_curve, y_curve = phase_resample(subset, x_column)
            x_curves.append(x_curve)
            y_curves.append(y_curve)
        mean_x = np.mean(np.vstack(x_curves), axis=0)
        mean_y = np.mean(np.vstack(y_curves), axis=0)
        means[x_column] = (mean_x, mean_y)

    for col, (x_column, xlabel, title) in enumerate(panels):
        mean_x, mean_y = means[x_column]
        axes[col].plot(mean_x, mean_y, color="black", lw=2.4, label="mean trajectory")
        axes[col].legend(loc="best", fontsize=8)
        axes[col].set_title(f"All cycles: {title}", loc="left", fontsize=10)

    axes[0].set_ylabel("Book1 Mx [N m]\n(flexion negative, extension positive)")
    fig.suptitle(
        "L4-L5 intact: complete FE cycles with phase-normalized mean trajectory\n"
        "Raw cycles shown in color; black curve is the pointwise mean after phase normalization; no regression fit",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    # Store both mean x axes in one long-form table for unambiguous reuse.
    mean_csv = pd.DataFrame({
        "phase": np.linspace(0.0, 1.0, MEAN_POINTS),
        "motor_angle_deg_mean": means["motor_angle_deg"][0],
        "opto_angle_deg_mean": means["fe_angle_sign_corrected_deg"][0],
        "moment_mx_nm_mean": means["motor_angle_deg"][1],
    })
    mean_csv.to_csv(OUT_CSV, index=False)
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_CSV}")
    print(f"cycles used={CYCLES}, points per mean curve={MEAN_POINTS}")


if __name__ == "__main__":
    main()
