"""Quantify spreader fit-window instability from cycle-level outputs.

This script decomposes cycle-level variance into:
1) condition (biomechanical intervention) signal,
2) additional variance explained by fit-selection mode switching,
3) residual unexplained variance.

It also reports per-condition mode-attributable variance and strict-only sensitivity.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).parent
OUT_DIR = BASE / "output"
REPORT_DIR = OUT_DIR / "window_mode_instability"

CSV_TRANSITION = OUT_DIR / "sync_check_transition_flags.csv"
CSV_ROT = OUT_DIR / "moment_theta_cycle_grids" / "all_interventions_moment_theta_metrics.csv"
CSV_LIN = OUT_DIR / "linear_region_stiffness_cycle_grids" / "all_interventions_linear_region_stiffness_cycles.csv"

CONDITION_ORDER = ["Intact", "PUF Left", "FUF Left", "FBF", "Posterior Release", "SPO"]
SPREADER_TO_CONDITION = {
    "intact, but holding": "Intact",
    "pubf, left": "PUF Left",
    "puf, left": "PUF Left",
    "fuf, left": "FUF Left",
    "fbf": "FBF",
    "posterior release": "Posterior Release",
    "spo": "SPO",
}


def load_transition_flags() -> set[tuple[str, int]]:
    if not CSV_TRANSITION.exists():
        return set()

    df = pd.read_csv(CSV_TRANSITION)
    flagged = df[df["transition_flag"].astype(str).str.lower().isin({"true", "1", "yes"})]
    return {(str(r.file_name), int(r.cycle)) for r in flagged.itertuples()}


def is_valid_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_metric_table(
    csv_path: Path,
    value_col: str,
    span_col: str,
    flagged: set[tuple[str, int]],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["condition"] = df["intervention"].astype(str).str.strip().str.lower().map(SPREADER_TO_CONDITION)
    df["fit_selection_mode"] = df["fit_selection_mode"].astype(str).str.strip().fillna("")
    df["is_valid"] = is_valid_bool(df["valid"])
    df["is_transition"] = [(str(fn), int(cy)) in flagged for fn, cy in zip(df["file_name"], df["cycle"])]

    keep = (
        df["is_valid"]
        & ~df["is_transition"]
        & df["condition"].notna()
        & df[value_col].notna()
        & (df["fit_selection_mode"].str.len() > 0)
    )
    out = df.loc[
        keep,
        [
            "file_name",
            "cycle",
            "condition",
            "fit_selection_mode",
            value_col,
            "r2",
            span_col,
        ],
    ].copy()
    out = out.rename(columns={value_col: "value", span_col: "fit_span"})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["r2"] = pd.to_numeric(out["r2"], errors="coerce")
    out["fit_span"] = pd.to_numeric(out["fit_span"], errors="coerce")
    out = out[np.isfinite(out["value"])]

    out["condition"] = pd.Categorical(out["condition"], categories=CONDITION_ORDER, ordered=True)
    out = out.sort_values(["condition", "file_name", "cycle"]).reset_index(drop=True)
    return out


def design_one_hot(series: pd.Series) -> np.ndarray:
    cats = pd.Categorical(series)
    k = len(cats.categories)
    if k <= 1:
        return np.zeros((len(series), 0), dtype=float)
    # Drop reference level to avoid exact collinearity with intercept.
    return pd.get_dummies(cats, drop_first=True, dtype=float).to_numpy()


def regression_r2(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return np.nan

    y_mean = float(np.mean(y))
    sst = float(np.sum((y - y_mean) ** 2))
    if sst <= 1e-12:
        return 0.0

    intercept = np.ones((y.size, 1), dtype=float)
    if x.size == 0:
        x_full = intercept
    else:
        x_full = np.column_stack([intercept, x])

    beta = np.linalg.pinv(x_full) @ y
    y_hat = x_full @ beta
    sse = float(np.sum((y - y_hat) ** 2))
    return max(0.0, min(1.0, 1.0 - sse / sst))


def pooled_within_sd(df: pd.DataFrame) -> float:
    num = 0.0
    den = 0
    for _, grp in df.groupby("condition", observed=True):
        vals = grp["value"].to_numpy(dtype=float)
        if vals.size >= 2:
            num += float(np.sum((vals - np.mean(vals)) ** 2))
            den += vals.size - 1
    if den <= 0:
        return np.nan
    return float(np.sqrt(num / den))


def decompose_metric(df: pd.DataFrame, metric_name: str) -> dict:
    y = df["value"].to_numpy(dtype=float)
    cond_x = design_one_hot(df["condition"])
    mode_x = design_one_hot(df["fit_selection_mode"])

    r2_cond = regression_r2(y, cond_x)
    r2_cond_mode = regression_r2(y, np.column_stack([cond_x, mode_x]) if mode_x.size else cond_x)
    r2_mode_inc = max(0.0, float(r2_cond_mode - r2_cond))
    r2_resid = max(0.0, 1.0 - r2_cond - r2_mode_inc)

    strict_only = df[df["fit_selection_mode"].str.startswith("strict", na=False)].copy()
    sd_all = pooled_within_sd(df)
    sd_strict = pooled_within_sd(strict_only)

    return {
        "metric": metric_name,
        "n_cycles": int(len(df)),
        "n_conditions": int(df["condition"].nunique()),
        "n_modes": int(df["fit_selection_mode"].nunique()),
        "strict_fraction": float(np.mean(df["fit_selection_mode"].str.startswith("strict", na=False))),
        "r2_condition_signal": float(r2_cond),
        "r2_mode_incremental": float(r2_mode_inc),
        "r2_unexplained": float(r2_resid),
        "pooled_within_sd_all": sd_all,
        "pooled_within_sd_strict_only": sd_strict,
        "within_sd_reduction_pct_strict_only": (
            100.0 * (sd_all - sd_strict) / sd_all if np.isfinite(sd_all) and np.isfinite(sd_strict) and sd_all > 0 else np.nan
        ),
        "strict_only_n_cycles": int(len(strict_only)),
    }


def one_way_mode_r2(vals: np.ndarray, modes: pd.Series) -> float:
    vals = np.asarray(vals, dtype=float)
    if vals.size < 4:
        return np.nan

    overall = float(np.mean(vals))
    sst = float(np.sum((vals - overall) ** 2))
    if sst <= 1e-12:
        return 0.0

    temp = pd.DataFrame({"v": vals, "m": modes})
    means = temp.groupby("m")["v"].mean()
    counts = temp.groupby("m")["v"].size()
    ss_mode = float(np.sum(counts * (means - overall) ** 2))
    return max(0.0, min(1.0, ss_mode / sst))


def by_condition(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    rows = []
    for cond in CONDITION_ORDER:
        grp = df[df["condition"] == cond]
        if grp.empty:
            continue

        vals = grp["value"].to_numpy(dtype=float)
        mode_counts = grp["fit_selection_mode"].value_counts()
        dominant_mode = mode_counts.index[0]
        dominant_share = float(mode_counts.iloc[0] / len(grp))

        r2_mode = one_way_mode_r2(vals, grp["fit_selection_mode"])
        cv_pct = 100.0 * (np.std(vals, ddof=1) / np.mean(vals)) if len(vals) >= 2 and np.mean(vals) != 0 else np.nan

        rows.append(
            {
                "metric": metric_name,
                "condition": cond,
                "n_cycles": int(len(vals)),
                "n_modes": int(grp["fit_selection_mode"].nunique()),
                "dominant_mode": dominant_mode,
                "dominant_mode_share": dominant_share,
                "mode_r2_within_condition": r2_mode,
                "cv_pct": cv_pct,
                "mean_value": float(np.mean(vals)),
                "std_value": float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan,
                "mean_r2": float(np.nanmean(grp["r2"].to_numpy(dtype=float))),
                "mean_fit_span": float(np.nanmean(grp["fit_span"].to_numpy(dtype=float))),
            }
        )

    return pd.DataFrame(rows)


def mode_counts(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    rows = []
    for cond in CONDITION_ORDER:
        grp = df[df["condition"] == cond]
        if grp.empty:
            continue
        counts = grp["fit_selection_mode"].value_counts()
        total = int(len(grp))
        for mode, n in counts.items():
            rows.append(
                {
                    "metric": metric_name,
                    "condition": cond,
                    "fit_selection_mode": mode,
                    "n_cycles": int(n),
                    "share": float(n / total),
                }
            )
    return pd.DataFrame(rows)


def cv_by_condition(df: pd.DataFrame, metric_name: str, subset_label: str) -> pd.DataFrame:
    rows = []
    for cond in CONDITION_ORDER:
        grp = df[df["condition"] == cond]
        if grp.empty:
            continue
        vals = grp["value"].to_numpy(dtype=float)
        if vals.size < 2:
            cv_pct = np.nan
        else:
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1))
            cv_pct = 100.0 * std_v / mean_v if abs(mean_v) > 1e-12 else np.nan

        rows.append(
            {
                "metric": metric_name,
                "condition": cond,
                "subset": subset_label,
                "n_cycles": int(vals.size),
                "cv_pct": cv_pct,
            }
        )
    return pd.DataFrame(rows)


def plot_report(
    decomp_df: pd.DataFrame,
    by_cond_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=180)

    # Panel 1: variance decomposition.
    ax = axes[0]
    x = np.arange(len(decomp_df))
    w = 0.6
    ax.bar(x, decomp_df["r2_condition_signal"], width=w, label="Condition signal")
    ax.bar(
        x,
        decomp_df["r2_mode_incremental"],
        width=w,
        bottom=decomp_df["r2_condition_signal"],
        label="Fit-mode switching",
    )
    ax.bar(
        x,
        decomp_df["r2_unexplained"],
        width=w,
        bottom=decomp_df["r2_condition_signal"] + decomp_df["r2_mode_incremental"],
        label="Residual",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(decomp_df["metric"].tolist())
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Fraction of cycle variance")
    ax.set_title("Variance decomposition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    # Panel 2: within-condition mode-attributable fraction for k_rot.
    ax = axes[1]
    sub = by_cond_df[by_cond_df["metric"] == "k_rot"].set_index("condition").reindex(CONDITION_ORDER)
    ax.bar(np.arange(len(sub)), sub["mode_r2_within_condition"], color="#ff7f0e")
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(CONDITION_ORDER, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mode R² within condition")
    ax.set_title("k_rot instability by condition")
    ax.grid(axis="y", alpha=0.25)

    # Panel 3: CV all vs strict-only for k_rot.
    ax = axes[2]
    cv_rot = cv_df[cv_df["metric"] == "k_rot"].copy()
    cv_all = (
        cv_rot[cv_rot["subset"] == "all"]
        .set_index("condition")
        .reindex(CONDITION_ORDER)
    )
    cv_strict = (
        cv_rot[cv_rot["subset"] == "strict_only"]
        .set_index("condition")
        .reindex(CONDITION_ORDER)
    )

    x = np.arange(len(CONDITION_ORDER))
    w = 0.38
    ax.bar(x - w / 2, cv_all["cv_pct"], width=w, color="#2ca02c", label="all valid")
    ax.bar(x + w / 2, cv_strict["cv_pct"], width=w, color="#1f77b4", label="strict-only")
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Cycle CV [%]")
    ax.set_title("k_rot variability sensitivity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    txt = []
    for _, r in decomp_df.iterrows():
        red = r["within_sd_reduction_pct_strict_only"]
        red_txt = f"{red:+.1f}%" if np.isfinite(red) else "NA"
        txt.append(
            f"{r['metric']}: mode R2={r['r2_mode_incremental']:.3f}, strict SD change={red_txt}"
        )
    fig.text(0.5, 0.01, " | ".join(txt), ha="center", fontsize=9)

    fig.suptitle("Spreader window-selection instability diagnostic", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    flagged = load_transition_flags()
    print(f"Transition-flagged cycles excluded: {len(flagged)}")

    metric_cfg = {
        "k_rot": (CSV_ROT, "k_linear_nm_per_deg", "fit_span_deg"),
        "k_lin": (CSV_LIN, "k_linear_n_per_mm", "fit_span_mm"),
    }

    all_decomp = []
    all_by_cond = []
    all_modes = []
    all_cv = []

    for metric, (path, value_col, span_col) in metric_cfg.items():
        df = load_metric_table(path, value_col, span_col, flagged)
        if df.empty:
            print(f"{metric}: no rows after filters")
            continue

        decomp = decompose_metric(df, metric)
        byc = by_condition(df, metric)
        modes = mode_counts(df, metric)
        cv_all = cv_by_condition(df, metric, "all")
        cv_strict = cv_by_condition(
            df[df["fit_selection_mode"].str.startswith("strict", na=False)],
            metric,
            "strict_only",
        )

        all_decomp.append(decomp)
        all_by_cond.append(byc)
        all_modes.append(modes)
        all_cv.append(cv_all)
        all_cv.append(cv_strict)

        print(
            f"{metric}: n={decomp['n_cycles']}, modes={decomp['n_modes']}, "
            f"R2_condition={decomp['r2_condition_signal']:.3f}, "
            f"R2_mode_increment={decomp['r2_mode_incremental']:.3f}, "
            f"R2_residual={decomp['r2_unexplained']:.3f}"
        )

    decomp_df = pd.DataFrame(all_decomp)
    by_cond_df = pd.concat(all_by_cond, ignore_index=True) if all_by_cond else pd.DataFrame()
    modes_df = pd.concat(all_modes, ignore_index=True) if all_modes else pd.DataFrame()
    cv_df = pd.concat(all_cv, ignore_index=True) if all_cv else pd.DataFrame()

    out_decomp = REPORT_DIR / "window_mode_variance_decomposition.csv"
    out_by_cond = REPORT_DIR / "window_mode_by_condition.csv"
    out_modes = REPORT_DIR / "window_mode_counts.csv"
    out_cv = REPORT_DIR / "window_mode_cv_by_condition.csv"
    out_png = REPORT_DIR / "window_mode_instability_summary.png"

    decomp_df.to_csv(out_decomp, index=False)
    by_cond_df.to_csv(out_by_cond, index=False)
    modes_df.to_csv(out_modes, index=False)
    cv_df.to_csv(out_cv, index=False)
    print(f"saved {out_decomp}")
    print(f"saved {out_by_cond}")
    print(f"saved {out_modes}")
    print(f"saved {out_cv}")

    if not decomp_df.empty and not by_cond_df.empty and not cv_df.empty:
        plot_report(decomp_df, by_cond_df, cv_df, out_png)
        print(f"saved {out_png}")


if __name__ == "__main__":
    main()
