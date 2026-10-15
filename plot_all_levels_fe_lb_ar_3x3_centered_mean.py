#%%
"""Combine level-specific FE, LB, and AR centered-mean plots in a 3x3 grid."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_all_levels_fe_cycle3_centered_mean_stacked import plot_fe_cycle3
from plot_all_levels_lb_ar_cycle_centered_mean_stacked import plot_axis


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_STEM = "all_levels_FE_LB_AR_centered_mean_3x3"
LEVELS = ("T12-L1", "L2-L3", "L4-L5")
LEVEL_GROUPS = (("T2-T3", "T4-T5"), ("T6-T7", "T8-T9"))
ALL_LEVELS = ("T2-T3", "T4-T5", "T6-T7", "T8-T9", "T12-L1", "L2-L3", "L4-L5")
ALL_LEVELS_OUT_STEM = "all_levels_FE_LB_AR_centered_mean_3x3_all_levels"
DEFAULT_STEMS = {
    "FE": "all_levels_FE_cycle3_centered_mean_stacked",
    "LB": "all_levels_LB_cycle3_centered_mean_stacked",
    "AR": "all_levels_AR_cycle3_centered_mean_stacked",
}
AXES = {
    "FE": {
        "cycle": None,
        "angle_column": "angle_mean_deg",
        "moment_column": "moment_mx_mean_nm",
        "angle_label": "FE angle change, ΔRy [deg]",
        "moment_label": "Mx [Nm]",
        "title": "FE, cycle 3",
    },
    "LB": {
        "cycle": 3,
        "angle_column": "angle_deg",
        "moment_column": "moment_nm",
        "angle_label": "LB angle change, ΔRz [deg]",
        "moment_label": "My [Nm]",
        "title": "LB, cycle 3",
    },
    "AR": {
        "cycle": 3,
        "angle_column": "angle_deg",
        "moment_column": "moment_nm",
        "angle_label": "AR angle change, ΔRx [deg]",
        "moment_label": "Mz [Nm]",
        "title": "AR, cycle 3",
    },
}
INTERVENTION_COLORS = {
    "Intact": "tab:blue",
    "Partial facetectomy 1": "tab:orange",
    "Partial facetectomy 2": "tab:green",
    "Full uni facetectomy": "tab:red",
    "Full bilat facetectomy": "tab:purple",
    "PLL Release": "tab:brown",
    "Laminectomy": "tab:pink",
    "Ligamentum Flavum Release": "tab:cyan",
    "Resected Ligamentum Flavum": "tab:cyan",
}


def load_axis_data(axis: str, config: dict, stem: str) -> pd.DataFrame:
    data = pd.read_csv(OUT_DIR / f"{stem}.csv")
    if config["cycle"] is not None:
        data = data[data["cycle"] == config["cycle"]].copy()
    if axis == "FE":
        data["curve_type"] = "centered_loading_ramp"
        data["moment_nm"] = data[config["moment_column"]]
    else:
        data["moment_nm"] = data[config["moment_column"]]
    data["axis"] = axis
    return data


def plot_3x3(levels: tuple[str, ...], stems: dict[str, str], out_stem: str, title: str) -> None:
    out_png = OUT_DIR / f"{out_stem}.png"
    out_csv = OUT_DIR / f"{out_stem}.csv"
    out_range_csv = OUT_DIR / f"{out_stem}_angle_ranges.csv"
    axis_data = {axis: load_axis_data(axis, config, stems[axis]) for axis, config in AXES.items()}
    fig, axes = plt.subplots(
        len(levels),
        len(AXES),
        figsize=(18.0, 16.0 / 3 * len(levels)),
        sharex="col",
        sharey="col",
        constrained_layout=True,
    )
    curve_rows = []
    range_rows = []

    for row_index, level in enumerate(levels):
        for column_index, (axis, config) in enumerate(AXES.items()):
            ax = axes[row_index, column_index]
            panel = axis_data[axis]
            panel = panel[panel["level"] == level]
            if panel.empty:
                raise RuntimeError(f"No {axis} data found for {level}.")

            groups = panel.groupby(
                ["intervention_order", "intervention", "curve_type", "cycle"],
                sort=False,
            )
            for (order, intervention, curve_type, cycle), group in groups:
                group = group.sort_values("phase_index")
                angle = group[config["angle_column"]].to_numpy(float)
                moment = group[config["moment_column"]].to_numpy(float)
                partial = curve_type == "partial_observed_trace"
                label = f"{int(order)}. {intervention}"
                if cycle != 3:
                    label += f" (cycle {int(cycle)})"
                if partial:
                    label += " (partial)"
                ax.plot(
                    angle,
                    moment,
                    color=INTERVENTION_COLORS.get(intervention, "0.25"),
                    lw=1.8 if partial else 2.0,
                    ls="--" if partial else "-",
                    label=label,
                )
                curve_rows.extend(group.to_dict(orient="records"))
                range_rows.append({
                    "level": level,
                    "axis": axis,
                    "cycle": int(cycle),
                    "intervention_order": int(order),
                    "intervention": intervention,
                    "curve_type": curve_type,
                    "angle_min_deg": float(angle.min()),
                    "angle_max_deg": float(angle.max()),
                    "moment_min_nm": float(moment.min()),
                    "moment_max_nm": float(moment.max()),
                    "n_points": len(group),
                })

            ax.axvline(0.0, color="0.65", lw=0.7)
            ax.axhline(0.0, color="0.65", lw=0.7)
            ax.set_title(f"{level} | {config['title']}", loc="left", fontsize=9)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.grid(alpha=0.22)
            ax.set_xlabel(config["angle_label"], fontsize=8)
            ax.set_ylabel(config["moment_label"], fontsize=8)
            positive_side_label = "Left" if axis in ("LB", "AR") else "Right"
            negative_side_label = "Right" if axis in ("LB", "AR") else "Left"
            ax.text(
                8.2,
                8.6,
                positive_side_label,
                ha="center",
                va="center",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
            )
            ax.text(
                -8.2,
                -8.6,
                negative_side_label,
                ha="center",
                va="center",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
            )
            ax.legend(
                loc="upper left",
                fontsize=6,
                ncol=2,
                title="Intervention order",
                title_fontsize=7,
                handlelength=1.3,
                columnspacing=0.7,
                labelspacing=0.25,
            )

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    pd.DataFrame(curve_rows).to_csv(out_csv, index=False)
    pd.DataFrame(range_rows).to_csv(out_range_csv, index=False)
    print(f"saved {out_png}")
    print(f"saved {out_csv}")
    print(f"saved {out_range_csv}")
    print(pd.DataFrame(range_rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def generate_level_group_inputs(levels: tuple[str, ...]) -> dict[str, str]:
    tag = "_".join(levels)
    names = ", ".join(levels)
    stems = {
        "FE": f"{tag}_FE_cycle3_centered_mean_stacked",
        "LB": f"{tag}_LB_cycle3_centered_mean_stacked",
        "AR": f"{tag}_AR_cycle3_centered_mean_stacked",
    }
    plot_fe_cycle3(levels, stems["FE"], f"Flexion/Extension Cycle 3: {names}")
    plot_axis("LB", 3, levels, stems["LB"], f"Lateral Bending Cycle 3: {names}")
    plot_axis("AR", 3, levels, stems["AR"], f"Axial Rotation Cycle 3: {names}")
    return stems


def main() -> None:
    plot_3x3(LEVELS, DEFAULT_STEMS, OUT_STEM, "Lumbar FE, LB, and AR Cycle Loading Ramps")
    for levels in LEVEL_GROUPS:
        stems = generate_level_group_inputs(levels)
        out_stem = f"all_levels_FE_LB_AR_centered_mean_3x3_{'_'.join(levels)}"
        plot_3x3(levels, stems, out_stem, f"Lumbar FE, LB, and AR Cycle Loading Ramps: {', '.join(levels)}")
    all_stems = generate_level_group_inputs(ALL_LEVELS)
    plot_3x3(ALL_LEVELS, all_stems, ALL_LEVELS_OUT_STEM, "Lumbar FE, LB, and AR Cycle Loading Ramps: All Levels")


if __name__ == "__main__":
    main()
