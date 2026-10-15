#%%
"""Compare raw flexion-loading stiffness across sequential surgical stages."""

from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from plot_all_levels_fe_cycle3_centered_mean_stacked import INTERVENTION_COLORS
from plot_levels_all_interventions_fe_cycles_centered_mean_stacked import load_cases


LEVELS = ("T12-L1", "L2-L3", "L4-L5")
PRIMARY_CYCLE = 3
CHECK_CYCLE = 2
MIN_FIT_POINTS = 6
MIN_LINEAR_R2 = 0.95
WINDOW_LOW_FRACTION = 0.40
WINDOW_HIGH_FRACTION = 0.95
MIN_WINDOW_SPAN_FRACTION = 0.20
WINDOW_GRID_POINTS = 23
OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_STEM = "6dof_flexion_stiffness"


def extract_flexion_loading(case: dict, cycle: int) -> dict:
    if cycle not in case["available_cycles"]:
        raise ValueError("cycle_not_fully_synchronized")
    start, end = case["boundaries"][cycle - 1:cycle + 1]
    mask = (case["motor_time"] >= start) & (case["motor_time"] < end)
    valid = mask & case["aligned_valid"] & np.isfinite(case["aligned_angle"])
    if int(valid.sum()) != int(mask.sum()) or int(valid.sum()) < MIN_FIT_POINTS:
        raise ValueError("incomplete_motor_sample_coverage")
    if start < case["opto_time"][0] + case["lag"] or end > case["opto_time"][-1] + case["lag"]:
        raise ValueError("incomplete_optotrak_time_coverage")

    moment = np.asarray(case["moment"][valid], dtype=float)
    angle = np.asarray(case["aligned_angle"][valid], dtype=float)
    time_s = np.asarray(case["motor_time"][valid], dtype=float)
    peak_index = int(np.argmin(moment))
    if peak_index < 2 or moment[peak_index] >= 0.0:
        raise ValueError("no_flexion_loading_peak")

    positive = np.flatnonzero(moment[:peak_index] > 0.0)
    if positive.size:
        start_index = int(positive[-1])
        fraction = moment[start_index] / (moment[start_index] - moment[start_index + 1])
        angle_at_zero = float(angle[start_index] + fraction * (angle[start_index + 1] - angle[start_index]))
    elif abs(float(moment[0])) <= 0.05 * abs(float(moment[peak_index])):
        start_index = 0
        angle_at_zero = float(angle[0])
    else:
        raise ValueError("loading_zero_crossing_not_observed")

    indices = np.arange(start_index, peak_index + 1)
    indices = indices[moment[indices] <= 0.0]
    if indices.size < MIN_FIT_POINTS:
        raise ValueError("too_few_loading_samples")
    return {
        "time_s": time_s[indices],
        "angle_deg": angle_at_zero - angle[indices],
        "moment_nm": -moment[indices],
        "angle_at_zero_deg": angle_at_zero,
        "cycle_start_s": float(start),
        "cycle_end_s": float(end),
        "n_cycle_samples": int(valid.sum()),
        "n_loading_samples": int(indices.size),
        "peak_flexion_moment_nm": float(-moment[peak_index]),
    }


def fit_stiffness(branch: dict, low_nm: float, high_nm: float) -> dict:
    angle = branch["angle_deg"]
    moment = branch["moment_nm"]
    selected = (
        np.isfinite(angle)
        & np.isfinite(moment)
        & (moment >= low_nm)
        & (moment <= high_nm)
    )
    result = {
        "k_nm_per_deg": np.nan,
        "intercept_nm": np.nan,
        "r2": np.nan,
        "rmse_nm": np.nan,
        "n_fit_samples": int(selected.sum()),
        "fit_angle_span_deg": np.nan,
        "fit_moment_min_nm": np.nan,
        "fit_moment_max_nm": np.nan,
        "fit_valid": False,
        "fit_reason": "too_few_fit_samples",
    }
    if int(selected.sum()) < MIN_FIT_POINTS:
        return result
    finite_moment = moment[np.isfinite(angle) & np.isfinite(moment)]
    if low_nm < float(finite_moment.min()) or high_nm > float(finite_moment.max()):
        result["fit_reason"] = "fit_window_outside_loading_support"
        return result
    angle_fit = angle[selected]
    moment_fit = moment[selected]
    result["fit_angle_span_deg"] = float(np.ptp(angle_fit))
    result["fit_moment_min_nm"] = float(moment_fit.min())
    result["fit_moment_max_nm"] = float(moment_fit.max())
    if np.ptp(angle_fit) <= 1e-9:
        result["fit_reason"] = "no_angle_variation"
        return result
    regression = linregress(angle_fit, moment_fit)
    prediction = regression.intercept + regression.slope * angle_fit
    result.update({
        "k_nm_per_deg": float(regression.slope),
        "intercept_nm": float(regression.intercept),
        "r2": float(regression.rvalue ** 2),
        "rmse_nm": float(np.sqrt(np.mean((moment_fit - prediction) ** 2))),
    })
    if regression.slope <= 0.0:
        result["fit_reason"] = "nonpositive_loading_stiffness"
    elif result["r2"] < MIN_LINEAR_R2:
        result["fit_reason"] = "below_linear_r2_threshold"
    else:
        result["fit_valid"] = True
        result["fit_reason"] = "ok"
    return result


def normalize_stiffness(data: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in data.groupby(["level", "cycle"], sort=False):
        group = group.sort_values("intervention_order").copy()
        valid = group["fit_valid"] & np.isfinite(group["k_nm_per_deg"]) & (group["k_nm_per_deg"] > 0.0)
        stiffness = group["k_nm_per_deg"].where(valid)
        intact = stiffness[group["intervention"] == "Intact"]
        intact_k = float(intact.iloc[0]) if len(intact) == 1 else np.nan
        group["intact_k_nm_per_deg"] = intact_k
        group["previous_intervention"] = group["intervention"].shift(1)
        group["previous_k_nm_per_deg"] = stiffness.shift(1)
        group["stiffness_retained_pct"] = 100.0 * stiffness / intact_k
        group["cumulative_change_pct"] = group["stiffness_retained_pct"] - 100.0
        group["stage_change_pct"] = 100.0 * (stiffness / stiffness.shift(1) - 1.0)
        group["stage_change_percentage_points"] = group["stiffness_retained_pct"].diff()
        frames.append(group)
    return pd.concat(frames, ignore_index=True) if frames else data.copy()


def select_common_window(branches: list[dict]) -> dict:
    result = {
        "window_low_nm": np.nan,
        "window_high_nm": np.nan,
        "common_peak_moment_nm": np.nan,
        "worst_primary_r2": np.nan,
        "minimum_primary_fit_samples": 0,
        "n_candidate_windows_tested": 0,
        "window_valid": False,
        "window_reason": "no_primary_loading_branches",
        "best_rejected_low_nm": np.nan,
        "best_rejected_high_nm": np.nan,
        "best_rejected_worst_r2": np.nan,
    }
    if not branches:
        return result
    common_peak = float(min(np.max(branch["moment_nm"]) for branch in branches))
    low = max(WINDOW_LOW_FRACTION * common_peak, max(float(np.min(branch["moment_nm"])) for branch in branches))
    high = WINDOW_HIGH_FRACTION * common_peak
    result["common_peak_moment_nm"] = common_peak
    if high - low < MIN_WINDOW_SPAN_FRACTION * common_peak:
        result["window_reason"] = "insufficient_shared_moment_support"
        return result

    edges = np.linspace(low, high, WINDOW_GRID_POINTS)
    candidates = sorted(
        [
            (float(upper - lower), float(lower), float(upper))
            for lower_index, lower in enumerate(edges)
            for upper in edges[lower_index + 1:]
            if upper - lower >= MIN_WINDOW_SPAN_FRACTION * common_peak - 1e-9
        ],
        reverse=True,
    )
    for _, lower, upper in candidates:
        result["n_candidate_windows_tested"] += 1
        fits = [fit_stiffness(branch, lower, upper) for branch in branches]
        if all(fit["fit_valid"] for fit in fits):
            result.update({
                "window_low_nm": lower,
                "window_high_nm": upper,
                "worst_primary_r2": min(fit["r2"] for fit in fits),
                "minimum_primary_fit_samples": min(fit["n_fit_samples"] for fit in fits),
                "window_valid": True,
                "window_reason": "ok",
            })
            return result
        if all(np.isfinite(fit["r2"]) and fit["k_nm_per_deg"] > 0.0 for fit in fits):
            worst_r2 = min(fit["r2"] for fit in fits)
            if not np.isfinite(result["best_rejected_worst_r2"]) or worst_r2 > result["best_rejected_worst_r2"]:
                result["best_rejected_low_nm"] = lower
                result["best_rejected_high_nm"] = upper
                result["best_rejected_worst_r2"] = worst_r2
    result["window_reason"] = "no_shared_linear_window"
    return result


def analyze_level(level: str, cases: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    extracted = {}
    errors = {}
    for order, case in enumerate(cases, 1):
        for cycle in (PRIMARY_CYCLE, CHECK_CYCLE):
            try:
                extracted[(order, cycle)] = extract_flexion_loading(case, cycle)
            except ValueError as error:
                errors[(order, cycle)] = str(error)
    primary = [extracted[(order, PRIMARY_CYCLE)] for order in range(1, len(cases) + 1) if (order, PRIMARY_CYCLE) in extracted]
    window = select_common_window(primary if len(primary) == len(cases) else [])
    if len(primary) != len(cases):
        window["window_reason"] = "incomplete_primary_cycle"
    window.update({
        "level": level,
        "primary_cycle": PRIMARY_CYCLE,
        "check_cycle": CHECK_CYCLE,
        "n_interventions": len(cases),
        "r2_threshold": MIN_LINEAR_R2,
        "minimum_fit_samples": MIN_FIT_POINTS,
        "candidate_low_fraction": WINDOW_LOW_FRACTION,
        "candidate_high_fraction": WINDOW_HIGH_FRACTION,
        "minimum_span_fraction": MIN_WINDOW_SPAN_FRACTION,
    })

    fit_rows = []
    sample_rows = []
    empty_branch = {"angle_deg": np.array([]), "moment_nm": np.array([])}
    for order, case in enumerate(cases, 1):
        for cycle in (PRIMARY_CYCLE, CHECK_CYCLE):
            metadata = {
                "level": level,
                "cycle": cycle,
                "cycle_role": "primary" if cycle == PRIMARY_CYCLE else "repeatability",
                "intervention_order": order,
                "intervention": case["name"],
                "motor_file": case["motor_file"],
                "optotrak_file": case["opto_file"],
                "motor_angle_lag_s": case["lag"],
                "motor_angle_lag_correlation": case["correlation"],
                "window_low_nm": window["window_low_nm"],
                "window_high_nm": window["window_high_nm"],
                "extraction_reason": errors.get((order, cycle), "ok"),
            }
            branch = extracted.get((order, cycle))
            if branch is not None:
                metadata.update({key: value for key, value in branch.items() if key not in ("time_s", "angle_deg", "moment_nm")})
            if window["window_valid"] and branch is not None:
                fit = fit_stiffness(branch, window["window_low_nm"], window["window_high_nm"])
            else:
                fit = fit_stiffness(empty_branch, np.nan, np.nan)
                fit["fit_reason"] = metadata["extraction_reason"] if branch is None else window["window_reason"]
            fit_rows.append({**metadata, **fit})
            if branch is None:
                continue
            for time_s, angle, moment in zip(branch["time_s"], branch["angle_deg"], branch["moment_nm"]):
                sample_rows.append({
                    **metadata,
                    "motor_time_s": float(time_s),
                    "flexion_angle_deg": float(angle),
                    "flexion_moment_nm": float(moment),
                    "in_fit_window": bool(window["window_low_nm"] <= moment <= window["window_high_nm"]),
                })
    fits = normalize_stiffness(pd.DataFrame(fit_rows))
    return fits, pd.DataFrame(sample_rows), window


def primary_summary(fits: pd.DataFrame) -> pd.DataFrame:
    keys = ["level", "intervention_order", "intervention"]
    primary = fits[fits["cycle"] == PRIMARY_CYCLE].copy()
    check_columns = ["k_nm_per_deg", "fit_valid", "fit_reason", "r2", "stiffness_retained_pct", "stage_change_pct"]
    check = fits[fits["cycle"] == CHECK_CYCLE][keys + check_columns].rename(
        columns={column: f"cycle2_{column}" for column in check_columns}
    )
    primary = primary.merge(check, on=keys, how="left", validate="one_to_one")
    valid = primary["fit_valid"] & primary["cycle2_fit_valid"].fillna(False)
    primary["cycle2_to3_stiffness_change_pct"] = (
        100.0 * (primary["k_nm_per_deg"] / primary["cycle2_k_nm_per_deg"] - 1.0)
    ).where(valid)
    primary["cycle2_to3_retained_difference_pp"] = (
        primary["stiffness_retained_pct"] - primary["cycle2_stiffness_retained_pct"]
    ).where(valid)
    return primary


def plot_comparison(summary: pd.DataFrame, windows: pd.DataFrame, out_dir: Path) -> Path:
    levels = windows["level"].tolist()
    fig, axes = plt.subplots(len(levels), 2, figsize=(15, 4.6 * len(levels)), squeeze=False, constrained_layout=True)
    for row, level in enumerate(levels):
        data = summary[summary["level"] == level].sort_values("intervention_order")
        window = windows[windows["level"] == level].iloc[0]
        stages = data["intervention_order"].to_numpy(int)
        labels = [f"{order}.\n{fill(name, 17)}" for order, name in zip(stages, data["intervention"])]
        retention_ax, change_ax = axes[row]
        retention_ax.axhline(100, color="0.25", ls="--", lw=1, label="Intact reference")
        change_ax.axhline(0, color="0.25", lw=1)
        if window["window_valid"]:
            retention_ax.plot(stages, data["stiffness_retained_pct"], "o-", color="tab:blue", label=f"Cycle {PRIMARY_CYCLE}")
            retention_ax.plot(stages, data["cycle2_stiffness_retained_pct"], "s--", color="0.5", ms=4, label=f"Cycle {CHECK_CYCLE} check")
            changes = data["stage_change_pct"].to_numpy(float)
            colors = ["tab:red" if value < 0.0 else "tab:green" for value in changes]
            change_ax.bar(stages, changes, color=colors, alpha=0.8)
            for stage, value in zip(stages, changes):
                if np.isfinite(value):
                    change_ax.annotate(f"{value:+.1f}%", (stage, value), xytext=(0, 5 if value >= 0 else -5), textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
            window_title = f"{window['window_low_nm']:.2f}-{window['window_high_nm']:.2f} Nm (magnitudes)"
        else:
            window_title = "No qualifying shared linear window"
            message = f"Not quantified: {window['window_reason']}"
            if np.isfinite(window["best_rejected_worst_r2"]):
                message += f"\nBest worst-condition R2: {window['best_rejected_worst_r2']:.3f} < {MIN_LINEAR_R2:.2f}"
            for ax in (retention_ax, change_ax):
                ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", fontsize=9)
        retention_ax.set_title(f"{level}: stiffness retained | {window_title}", loc="left", fontsize=10)
        change_ax.set_title(f"{level}: conditional change from preceding stage", loc="left", fontsize=10)
        retention_ax.set_ylabel("Stiffness retained [% of intact]")
        change_ax.set_ylabel("Stage-to-stage stiffness change [%]")
        retention_ax.legend(fontsize=8)
        for ax in (retention_ax, change_ax):
            ax.set_xticks(stages, labels, fontsize=7)
            ax.set_xlim(0.5, stages[-1] + 0.5)
            ax.margins(y=0.2)
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("6DOF flexion-loading stiffness: cumulative and conditional stage effects", fontsize=13)
    path = out_dir / f"{OUT_STEM}_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_loading_fits(summary: pd.DataFrame, samples: pd.DataFrame, windows: pd.DataFrame, out_dir: Path) -> Path:
    levels = windows["level"].tolist()
    fig, axes = plt.subplots(len(levels), 1, figsize=(10, 4.5 * len(levels)), squeeze=False, constrained_layout=True)
    for ax, level in zip(axes[:, 0], levels):
        data = summary[summary["level"] == level].sort_values("intervention_order")
        window = windows[windows["level"] == level].iloc[0]
        for _, condition in data.iterrows():
            branch = samples[(samples["level"] == level) & (samples["cycle"] == PRIMARY_CYCLE) & (samples["intervention_order"] == condition["intervention_order"])]
            if branch.empty:
                continue
            color = INTERVENTION_COLORS.get(condition["intervention"], "0.3")
            ax.plot(branch["flexion_angle_deg"], branch["flexion_moment_nm"], color=color, lw=1.3, alpha=0.65, label=f"{condition['intervention_order']}. {condition['intervention']}")
            selected = branch[branch["in_fit_window"]]
            if condition["fit_valid"] and not selected.empty:
                angle = np.array([selected["flexion_angle_deg"].min(), selected["flexion_angle_deg"].max()])
                prediction = condition["intercept_nm"] + condition["k_nm_per_deg"] * angle
                ax.scatter(selected["flexion_angle_deg"], selected["flexion_moment_nm"], color=color, s=12)
                ax.plot(angle, prediction, color=color, lw=2.5, ls="--")
        if window["window_valid"]:
            ax.axhspan(window["window_low_nm"], window["window_high_nm"], color="0.8", alpha=0.25)
            title = f"{level}: shared band {window['window_low_nm']:.2f}-{window['window_high_nm']:.2f} Nm"
        else:
            title = f"{level}: raw loading only; no qualifying shared linear window"
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Flexion angle from loading zero [deg, magnitude]")
        ax.set_ylabel("Flexion Mx [Nm, magnitude]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2, title="Surgical order")
    fig.suptitle(f"Raw 6DOF cycle {PRIMARY_CYCLE} flexion loading and accepted regressions", fontsize=13)
    path = out_dir / f"{OUT_STEM}_loading_fits.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def markdown_value(value) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)


def markdown_table(data: pd.DataFrame) -> str:
    columns = list(data.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in data.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(markdown_value(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, fits: pd.DataFrame, windows: pd.DataFrame, out_dir: Path) -> Path:
    lines = [
        "# 6DOF flexion-loading stiffness relative to intact",
        "",
        "Primary: cycle 3. Repeatability check: cycle 2, using the same selected band (no refitting the band).",
        "Each level is normalized to its own Intact condition. Surgical order follows the numbered source folders, not a universal intervention-name order.",
        "",
        "## Method",
        "",
        "- Use synchronized Motor Mx and sign-corrected Optotrak segment Ry, not motor angle as the response.",
        "- Extract chronological negative-Mx loading from the last zero crossing before the flexion peak; exclude unloading and extension.",
        "- Mirror flexion signs to positive magnitudes and reference angle to loading zero. Fit M = intercept + K * angle with a free intercept.",
        f"- Select the widest common cycle-3 band between {100 * WINDOW_LOW_FRACTION:.0f}% and {100 * WINDOW_HIGH_FRACTION:.0f}% of the smallest recorded flexion peak at that level, searching {WINDOW_GRID_POINTS} evenly spaced boundaries.",
        f"- Require at least {MIN_FIT_POINTS} raw paired samples and R2 >= {MIN_LINEAR_R2:.2f} in every intervention, with a band spanning at least {100 * MIN_WINDOW_SPAN_FRACTION:.0f}% of the common peak. Ties favor the higher-moment band.",
        "- All conditions at a level use identical bounds. No intervention-specific tuning, fixed 1-4 Nm anchors, extrapolation, or fallback cycles.",
        "- Retained [%] = 100 * K / K_intact; cumulative change [%] = retained - 100.",
        "- Stage change [%] = 100 * (K / K_previous - 1); stage change [percentage points] = retained - retained_previous.",
        "- Invalid fits and gaps in the sequence remain missing; never jump across a rejected stage to compute an incremental change.",
        "",
        "## Selected windows",
        "",
        markdown_table(windows[["level", "window_low_nm", "window_high_nm", "worst_primary_r2", "minimum_primary_fit_samples", "window_reason", "best_rejected_worst_r2"]]),
    ]
    for level in windows["level"]:
        data = summary[summary["level"] == level]
        table = data[["intervention_order", "intervention", "k_nm_per_deg", "r2", "n_fit_samples", "stiffness_retained_pct", "cumulative_change_pct", "stage_change_pct", "stage_change_percentage_points", "cycle2_to3_stiffness_change_pct"]]
        lines.extend(["", f"## {level}", "", markdown_table(table)])
    rejected = fits[~fits["fit_valid"]][["level", "cycle", "intervention", "fit_reason", "r2"]]
    lines.extend([
        "", "## Quality flags", "",
        markdown_table(rejected) if not rejected.empty else "All requested fits passed the shared quality rules.",
        "", "## Interpretation", "",
        "These are high-load flexion-loading regression slopes, not whole-curve stiffness, neutral-zone stiffness, or loading/unloading mean stiffness.",
        "Changes may be positive or negative; do not force monotonic reductions. Compare small stage changes with the cycle-2-to-3 differences before interpreting them.",
        "Sequential stages measure cumulative effects and additional effects conditional on prior surgery. They do not isolate each structure's contribution to intact stiffness.",
        "Cycles are technical repeats, not independent specimens. No population confidence intervals or significance claims are inferred from them.",
        "The analysis retains the existing FE synchronization and cycle definitions; fit quality is not an independent proof of alignment accuracy.",
        "The spreader force-to-pivot-moment conversion is not imposed on the 6DOF loading range.",
        "", "## Outputs", "",
        f"- [{OUT_STEM}_summary.csv]({OUT_STEM}_summary.csv): primary results and cycle-2 comparisons.",
        f"- [{OUT_STEM}_cycles.csv]({OUT_STEM}_cycles.csv): fits, source files, synchronization, extraction, and rejection reasons for both cycles.",
        f"- [{OUT_STEM}_windows.csv]({OUT_STEM}_windows.csv): approved bounds and best rejected-window diagnostics.",
        f"- [{OUT_STEM}_loading_samples.csv]({OUT_STEM}_loading_samples.csv): paired raw loading samples and selected-band membership.",
        f"- [{OUT_STEM}_comparison.png]({OUT_STEM}_comparison.png): retained stiffness and conditional stage changes.",
        f"- [{OUT_STEM}_loading_fits.png]({OUT_STEM}_loading_fits.png): raw loading branches and accepted regressions.",
    ])
    path = out_dir / f"{OUT_STEM}_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    fit_frames = []
    sample_frames = []
    window_rows = []
    for level in LEVELS:
        fits, samples, window = analyze_level(level, load_cases(level))
        fit_frames.append(fits)
        sample_frames.append(samples)
        window_rows.append(window)
        print(f"{level}: {window['window_reason']}; shared band {window['window_low_nm']:.3f}-{window['window_high_nm']:.3f} Nm")
    fits = pd.concat(fit_frames, ignore_index=True)
    samples = pd.concat(sample_frames, ignore_index=True)
    windows = pd.DataFrame(window_rows)
    summary = primary_summary(fits)
    for suffix, data in (("cycles", fits), ("summary", summary), ("windows", windows), ("loading_samples", samples)):
        path = OUT_DIR / f"{OUT_STEM}_{suffix}.csv"
        data.to_csv(path, index=False)
        print(f"saved {path}")
    print(f"saved {plot_comparison(summary, windows, OUT_DIR)}")
    print(f"saved {plot_loading_fits(summary, samples, windows, OUT_DIR)}")
    print(f"saved {write_report(summary, fits, windows, OUT_DIR)}")
    print(summary[["level", "intervention", "k_nm_per_deg", "stiffness_retained_pct", "cumulative_change_pct", "stage_change_pct", "fit_reason"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Stage effects are conditional on prior surgery, not isolated intervention contributions.")


if __name__ == "__main__":
    main()