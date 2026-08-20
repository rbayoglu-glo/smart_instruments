from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_mts_testing as mts
import plot_moment_theta_all_interventions as lam


OUT_DIR = Path(__file__).parent / "output"
LAMINA_SUMMARY_CSV = OUT_DIR / "moment_theta_cycle_grids" / "all_interventions_moment_theta_stiffness_summary.csv"
MTS_FLEXION_CSV = OUT_DIR / "mts_flexion_linear_region_stiffness.csv"
SCALING_CSV = OUT_DIR / "mts_flexion_to_lamina_scaling.csv"
FIT_PLOT_PNG = OUT_DIR / "mts_flexion_rising_branch_linear_fits.png"
SCALE_PLOT_PNG = OUT_DIR / "mts_flexion_to_lamina_scaling_comparison.png"
MOTION_NAME = "Flexion/Extension"
MAX_FLEXION_MOMENT_NM = float(mts.TARGET_MOMENT_NM)


def extract_centered_positive_flexion_branch(
    angle_cycle_deg: np.ndarray,
    moment_cycle_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Extract the centered positive flexion branch used in the MTS chart.

    Returns theta_branch, moment_branch, theta_avg_centered, moment_avg_centered, reason.
    """
    m_avg, a_avg, _, _ = mts.compute_centered_average_curve(angle_cycle_deg, moment_cycle_nm)
    if m_avg.size < 3 or a_avg.size < 3:
        return np.array([]), np.array([]), a_avg, m_avg, "centered_curve_too_short"

    mask = (
        np.isfinite(a_avg)
        & np.isfinite(m_avg)
        & (a_avg >= 0.0)
        & (m_avg >= 0.0)
        & (m_avg <= (MAX_FLEXION_MOMENT_NM + 1e-9))
    )
    idx = np.flatnonzero(mask)
    if idx.size < 3:
        return np.array([]), np.array([]), a_avg, m_avg, "no_positive_flexion_branch"

    # Keep the largest contiguous run in the positive branch window.
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, breaks)
    run = max(runs, key=lambda r: r.size)
    if run.size < 3:
        return np.array([]), np.array([]), a_avg, m_avg, "positive_branch_too_short"

    theta_branch = a_avg[run]
    moment_branch = m_avg[run]
    return theta_branch, moment_branch, a_avg, m_avg, "ok"


def normalize_name(name: str) -> str:
    key = str(name).strip().lower().replace(",", "")
    key = " ".join(key.split())
    if key == "intact":
        return "intact but holding"
    return key


def analyze_branch_linear_region(theta_rel_deg: np.ndarray, moment_rel_nm: np.ndarray) -> dict:
    x_raw = np.asarray(theta_rel_deg, dtype=float)
    y_raw = np.asarray(moment_rel_nm, dtype=float)

    x_env, y_env, env_reason = lam.extract_forward_loading_envelope(x_raw, y_raw)
    if env_reason != "ok":
        return {
            "ok": False,
            "reason": env_reason,
            "x_raw": x_raw,
            "y_raw": y_raw,
            "x_env": x_env,
            "y_env": y_env,
        }

    toe_mask = x_env >= lam.TOE_DEG
    x_lin = x_env[toe_mask]
    y_lin = y_env[toe_mask]
    if x_lin.size < lam.MIN_POINTS:
        return {
            "ok": False,
            "reason": "too_few_points_after_toe_cut",
            "x_raw": x_raw,
            "y_raw": y_raw,
            "x_env": x_env,
            "y_env": y_env,
            "x_lin": x_lin,
            "y_lin": y_lin,
        }

    x_active, y_active, idx_active, flat_thr, flat_reason = lam.remove_flat_subregions(x_lin, y_lin)
    if x_active.size < lam.MIN_POINTS:
        return {
            "ok": False,
            "reason": "too_few_points_after_flat_filter",
            "x_raw": x_raw,
            "y_raw": y_raw,
            "x_env": x_env,
            "y_env": y_env,
            "x_lin": x_lin,
            "y_lin": y_lin,
            "x_active": x_active,
            "y_active": y_active,
            "flat_slope_thr_nm_per_deg": flat_thr,
            "flat_filter_reason": flat_reason,
        }

    fit = lam.find_best_linear_region(x_active, y_active, idx_active)
    if not fit["ok"]:
        return {
            "ok": False,
            "reason": fit["reason"],
            "x_raw": x_raw,
            "y_raw": y_raw,
            "x_env": x_env,
            "y_env": y_env,
            "x_lin": x_lin,
            "y_lin": y_lin,
            "x_active": x_active,
            "y_active": y_active,
            "flat_slope_thr_nm_per_deg": flat_thr,
            "flat_filter_reason": flat_reason,
        }

    i0 = int(fit["i0"])
    i1 = int(fit["i1"])
    return {
        "ok": True,
        "reason": "ok",
        "x_raw": x_raw,
        "y_raw": y_raw,
        "x_env": x_env,
        "y_env": y_env,
        "x_lin": x_lin,
        "y_lin": y_lin,
        "x_active": x_active,
        "y_active": y_active,
        "flat_slope_thr_nm_per_deg": flat_thr,
        "flat_filter_reason": flat_reason,
        "i0": i0,
        "i1": i1,
        **fit,
        "theta0_deg": float(x_active[i0]),
        "theta1_deg": float(x_active[i1]),
        "m0_nm": float(y_active[i0]),
        "m1_nm": float(y_active[i1]),
    }


def plot_fit_grid(results: list[dict], out_png: Path) -> None:
    n = len(results)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.2 * nrows), squeeze=False)

    for i, rec in enumerate(results):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]

        res = rec["fit"]
        if "x_avg_centered" in res and "y_avg_centered" in res:
            ax.plot(
                res["y_avg_centered"],
                res["x_avg_centered"],
                color="0.85",
                lw=1.0,
                ls="--",
                label="avg(load/unload), centered",
            )
        ax.plot(res["y_raw"], res["x_raw"], color="0.80", lw=1.2, label="rising branch")
        if res["x_env"].size:
            ax.plot(res["y_env"], res["x_env"], color="tab:blue", lw=1.5, label="forward envelope")
        if "x_active" in res and np.asarray(res["x_active"]).size:
            ax.plot(res["y_active"], res["x_active"], color="tab:green", lw=1.4, label="envelope (flat removed)")

        ax.axhline(lam.TOE_DEG, color="tab:red", ls="--", lw=1.0, alpha=0.9)

        if res["ok"]:
            i0 = int(res["i0"])
            i1 = int(res["i1"])
            xx = res["x_active"][i0 : i1 + 1]
            yy = res["y_active"][i0 : i1 + 1]

            # Show selected fit points sparsely and with low opacity to avoid visual clutter.
            step = max(1, int(np.ceil(xx.size / 24)))
            ax.scatter(
                yy[::step],
                xx[::step],
                s=12,
                color="tab:red",
                alpha=0.35,
                edgecolors="none",
                label="selected points (sparse)",
                zorder=5,
            )

            yfit = res["slope_nm_per_deg"] * xx + res["intercept_nm"]
            ax.plot(yfit, xx, color="tab:red", lw=1.2, ls=":", label="linear fit")

            title = (
                f"{rec['intervention']} | k={res['slope_nm_per_deg']:.3f} Nm/deg | "
                f"R2={res['r2']:.3f}"
            )
        else:
            title = f"{rec['intervention']} | N/A ({res['reason']})"

        ax.set_title(title, fontsize=8)
        ax.set_xlabel("moment_rel [Nm]")
        ax.set_ylabel("theta_rel [deg]")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        "MTS Flexion Positive Branch (centered 0 to 7.5 Nm): linear-region fit and stiffness\n"
        "Same linear-region logic as lamina spreader (global rules, no intervention-specific tuning)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"saved {out_png}")


def build_mts_flexion_results() -> list[dict]:
    books = mts.get_book_numbers(mts.DATA_DIR)
    interventions = mts.build_intervention_map(books)

    rows: list[dict] = []
    for intervention, book_map in interventions.items():
        book_no = int(book_map[MOTION_NAME])
        file_path = mts.DATA_DIR / f"Book{book_no}.xlsx"

        angle_all, moment_all = mts.read_motion_data(file_path, mts.MOTION_CONFIG[MOTION_NAME]["moment_col"])
        _, _, angle_cycle, moment_cycle = mts.extract_third_cycle(angle_all, moment_all)

        theta_rel, moment_rel, theta_avg, moment_avg, branch_reason = extract_centered_positive_flexion_branch(
            angle_cycle,
            moment_cycle,
        )
        if branch_reason != "ok":
            fit = {
                "ok": False,
                "reason": branch_reason,
                "x_raw": theta_rel,
                "y_raw": moment_rel,
                "x_env": np.array([]),
                "y_env": np.array([]),
                "x_avg_centered": theta_avg,
                "y_avg_centered": moment_avg,
            }
        else:
            fit = analyze_branch_linear_region(theta_rel, moment_rel)
            fit["x_avg_centered"] = theta_avg
            fit["y_avg_centered"] = moment_avg

        rows.append(
            {
                "intervention": intervention,
                "book": book_no,
                "fit": fit,
            }
        )

    return rows


def save_mts_flexion_table(results: list[dict], out_csv: Path) -> pd.DataFrame:
    rows = []
    for rec in results:
        fit = rec["fit"]
        rows.append(
            {
                "Intervention": rec["intervention"],
                "Book": int(rec["book"]),
                "Motion": MOTION_NAME,
                "valid": bool(fit.get("ok", False)),
                "reason": str(fit.get("reason", "")),
                "k_linear_nm_per_deg": float(fit["slope_nm_per_deg"]) if fit.get("ok", False) else np.nan,
                "intercept_nm": float(fit["intercept_nm"]) if fit.get("ok", False) else np.nan,
                "r2": float(fit["r2"]) if fit.get("ok", False) else np.nan,
                "fit_n_points": int(fit["n_fit"]) if fit.get("ok", False) else np.nan,
                "fit_span_deg": float(fit["span_deg"]) if fit.get("ok", False) else np.nan,
                "fit_theta0_deg": float(fit["theta0_deg"]) if fit.get("ok", False) else np.nan,
                "fit_theta1_deg": float(fit["theta1_deg"]) if fit.get("ok", False) else np.nan,
                "fit_m0_nm": float(fit["m0_nm"]) if fit.get("ok", False) else np.nan,
                "fit_m1_nm": float(fit["m1_nm"]) if fit.get("ok", False) else np.nan,
                "fit_selection_mode": str(fit.get("fit_selection_mode", "")),
                "toe_deg": float(lam.TOE_DEG),
                "flat_slope_thr_nm_per_deg": float(fit["flat_slope_thr_nm_per_deg"])
                if "flat_slope_thr_nm_per_deg" in fit and np.isfinite(fit["flat_slope_thr_nm_per_deg"])
                else np.nan,
                "flat_filter_reason": str(fit.get("flat_filter_reason", "")),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")
    return df


def compute_scaling(mts_flexion_df: pd.DataFrame) -> pd.DataFrame:
    if not LAMINA_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing lamina summary: {LAMINA_SUMMARY_CSV}")

    lamina = pd.read_csv(LAMINA_SUMMARY_CSV)
    lamina["key"] = lamina["intervention"].map(normalize_name)

    mts_tab = mts_flexion_df.copy()
    mts_tab["key"] = mts_tab["Intervention"].map(normalize_name)

    merged = mts_tab.merge(
        lamina[["key", "intervention", "k_mean_nm_per_deg", "k_median_nm_per_deg", "n_valid"]],
        on="key",
        how="left",
    )

    merged["scale_laminaMedian_over_mts"] = merged["k_median_nm_per_deg"] / merged["k_linear_nm_per_deg"]
    merged["scale_laminaMean_over_mts"] = merged["k_mean_nm_per_deg"] / merged["k_linear_nm_per_deg"]

    valid_scale = merged["valid"] & np.isfinite(merged["scale_laminaMedian_over_mts"])
    if np.any(valid_scale):
        global_scale = float(np.nanmedian(merged.loc[valid_scale, "scale_laminaMedian_over_mts"]))
    else:
        global_scale = np.nan

    merged["global_scale_median"] = global_scale
    merged["mts_scaled_global_nm_per_deg"] = merged["k_linear_nm_per_deg"] * global_scale
    merged["residual_vs_lamina_median_nm_per_deg"] = merged["mts_scaled_global_nm_per_deg"] - merged["k_median_nm_per_deg"]

    merged.to_csv(SCALING_CSV, index=False)
    print(f"saved {SCALING_CSV}")
    print(f"global scale (median lamina / mts flexion) = {global_scale:.6f}")
    return merged


def plot_scaling_comparison(merged: pd.DataFrame, out_png: Path) -> None:
    df = merged.copy()
    df = df[df["valid"] & np.isfinite(df["k_linear_nm_per_deg"]) & np.isfinite(df["k_median_nm_per_deg"])].copy()
    if df.empty:
        print("skip scaling plot: no valid merged rows")
        return

    x = np.arange(len(df))
    w = 0.27

    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    ax.bar(x - w, df["k_linear_nm_per_deg"], width=w, color="#4C78A8", edgecolor="black", linewidth=0.7, label="MTS flexion (raw)")
    ax.bar(x, df["mts_scaled_global_nm_per_deg"], width=w, color="#72B7B2", edgecolor="black", linewidth=0.7, label="MTS flexion (scaled)")
    ax.bar(x + w, df["k_median_nm_per_deg"], width=w, color="#E45756", edgecolor="black", linewidth=0.7, label="Lamina median")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Intervention"].astype(str), rotation=18, ha="right")
    ax.set_ylabel("Stiffness [Nm/deg]")
    ax.set_title("Flexion-only scaling: MTS to Lamina stiffness")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    if np.isfinite(df["global_scale_median"].iloc[0]):
        s = float(df["global_scale_median"].iloc[0])
        ax.text(
            0.01,
            0.98,
            f"Global scale S (median ratio) = {s:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
        )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("MTS -> Lamina calibration (Flexion only)")
    print("- Uses only Flexion/Extension (LB and AR excluded)")
    print("- Uses rising branch of third cycle")
    print("- Uses same linear-region logic as lamina moment-theta analysis")

    results = build_mts_flexion_results()
    plot_fit_grid(results, FIT_PLOT_PNG)

    mts_df = save_mts_flexion_table(results, MTS_FLEXION_CSV)
    merged = compute_scaling(mts_df)
    plot_scaling_comparison(merged, SCALE_PLOT_PNG)


if __name__ == "__main__":
    main()
