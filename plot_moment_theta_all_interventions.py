from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_spreader_secant_stiffness as ass
import lamina_spreader as ls
import plot_theta_moment_time_all_interventions as base


OUT_DIR = Path(__file__).parent / "output" / "moment_theta_cycle_grids"
THETA_XLIM_DEG = (-4.0, 7.0)
CSV_TRANSITION_FLAGS = Path(__file__).parent / "output" / "sync_check_transition_flags.csv"

# Global linear-region settings shared for all interventions.
MIN_POINTS = 8
MIN_SPAN_DEG = float(ls.tip_mm_to_theta_deg(2.0))
MIN_MOMENT_RISE_NM = float(ls.force_to_moment_nm(5.0))
MIN_SLOPE_NM_PER_DEG = float(0.5 / ls.k_rot_to_k_linear_factor())
MIN_R2_STRICT = 0.97
FLAT_EDGE_SLOPE_MAX_NM_PER_DEG = float(1.0 / ls.k_rot_to_k_linear_factor())
FLAT_EDGE_SLOPE_FRACTION_OF_GLOBAL = 0.30
MIN_FLAT_RUN_SPAN_DEG = float(ls.tip_mm_to_theta_deg(1.2))
INITIAL_REGION_START_FRAC_MAX = 0.35
INITIAL_REGION_END_FRAC_MAX = 0.80
THETA_MIN_DEG = 0.0


def _clean_intervention(name: str) -> str:
    txt = str(name).strip().replace("_", " ")
    txt = re.sub(r"^\d+\s*-\s*", "", txt)
    txt = " ".join(txt.split())
    low = txt.lower()
    if "intact" in low and "holding" in low:
        return "Intact"
    if low == "intact":
        return "Intact"
    if "pubf" in low or low.startswith("puf"):
        return "PUF"
    if low.startswith("fuf"):
        return "FUF"
    return txt


def _display_case_name(file_name: str, intervention: str) -> str:
    m = re.match(r"^(\d+)\s*-\s*", str(file_name).strip())
    if m:
        return f"{m.group(1)} - {_clean_intervention(intervention)}"
    return _clean_intervention(intervention)


def _to_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "yes", "y"])


def load_transition_flag_map() -> dict[tuple[str, int], bool]:
    if not CSV_TRANSITION_FLAGS.exists():
        return {}

    df = pd.read_csv(CSV_TRANSITION_FLAGS)
    req = {"file_name", "cycle", "transition_flag"}
    if df.empty or not req.issubset(df.columns):
        return {}

    tf = _to_bool(df["transition_flag"])
    out: dict[tuple[str, int], bool] = {}
    for r, is_flag in zip(df.itertuples(index=False), tf.to_numpy(bool)):
        out[(str(r.file_name).strip(), int(r.cycle))] = bool(is_flag)
    return out


def apply_transition_exclusions(results: list[dict], file_name: str, flag_map: dict[tuple[str, int], bool]) -> list[dict]:
    out = []
    for res in results:
        key = (str(file_name).strip(), int(res.get("cycle", -1)))
        if flag_map.get(key, False):
            rr = dict(res)
            rr["ok"] = False
            rr["reason"] = "transition_flagged_excluded"
            out.append(rr)
        else:
            out.append(res)
    return out


def _line_fit_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


def extract_forward_loading_envelope(theta_deg: np.ndarray, moment_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Upper loading envelope up to peak moment, monotonic in theta and moment."""
    x = np.asarray(theta_deg, dtype=float)
    y = np.asarray(moment_nm, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return np.array([]), np.array([]), "too_few_points"

    pk = int(np.nanargmax(y))
    if pk < 1:
        return np.array([]), np.array([]), "peak_too_early"

    x = x[: pk + 1]
    y = y[: pk + 1]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_u, idx = np.unique(x, return_index=True)
    y_max = np.maximum.reduceat(y, idx)
    y_env = np.maximum.accumulate(y_max)

    if x_u.size < 2:
        return np.array([]), np.array([]), "envelope_too_short"

    return x_u, y_env, "ok"


def remove_flat_subregions(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Drop long low-slope runs from the envelope before linear fitting."""
    if x.size < 2 or y.size < 2:
        return x, y, np.arange(x.size, dtype=int), np.nan, "too_few_points"

    dx = np.diff(x)
    dy = np.diff(y)
    valid_edge = dx > 1e-9
    if not np.any(valid_edge):
        return x, y, np.arange(x.size, dtype=int), np.nan, "degenerate_dx"

    local_slope = np.zeros_like(dx)
    local_slope[valid_edge] = dy[valid_edge] / dx[valid_edge]
    span = float(x[-1] - x[0])
    global_slope = float((y[-1] - y[0]) / span) if span > 1e-9 else np.nan
    thr = max(
        FLAT_EDGE_SLOPE_MAX_NM_PER_DEG,
        FLAT_EDGE_SLOPE_FRACTION_OF_GLOBAL * global_slope
        if np.isfinite(global_slope)
        else FLAT_EDGE_SLOPE_MAX_NM_PER_DEG,
    )

    flat_edge = valid_edge & (local_slope < thr)
    keep = np.ones(x.size, dtype=bool)

    e = 0
    while e < flat_edge.size:
        if not flat_edge[e]:
            e += 1
            continue

        e0 = e
        while e + 1 < flat_edge.size and flat_edge[e + 1]:
            e += 1
        e1 = e

        run_span = float(x[e1 + 1] - x[e0])
        if run_span >= MIN_FLAT_RUN_SPAN_DEG:
            keep[e0 : e1 + 2] = False

        e += 1

    if int(np.count_nonzero(keep)) < 2:
        return np.array([]), np.array([]), np.array([], dtype=int), thr, "all_flat_after_filter"

    idx = np.flatnonzero(keep)
    return x[keep], y[keep], idx, thr, "ok"


def find_best_linear_region(x: np.ndarray, y: np.ndarray, idx_orig: np.ndarray | None = None) -> dict:
    n = x.size
    if idx_orig is None:
        idx_orig = np.arange(n, dtype=int)

    x_min = float(x[0])
    x_span_total = max(float(x[-1] - x_min), 1e-9)
    x_start_limit = x_min + INITIAL_REGION_START_FRAC_MAX * x_span_total
    x_end_limit = x_min + INITIAL_REGION_END_FRAC_MAX * x_span_total

    best_strict_initial = None
    best_strict_initial_score = None
    best_strict = None
    best_strict_score = None

    best_fallback_initial = None
    best_fallback_initial_score = None
    best_fallback = None
    best_fallback_score = None

    for i0 in range(0, n - MIN_POINTS + 1):
        for i1 in range(i0 + MIN_POINTS - 1, n):
            if int(idx_orig[i1] - idx_orig[i0]) != int(i1 - i0):
                continue

            span = float(x[i1] - x[i0])
            if span < MIN_SPAN_DEG:
                continue

            xx = x[i0 : i1 + 1]
            yy = y[i0 : i1 + 1]
            rise = float(yy[-1] - yy[0])
            if rise < MIN_MOMENT_RISE_NM:
                continue

            slope, intercept, r2 = _line_fit_stats(xx, yy)
            if not np.isfinite(slope) or not np.isfinite(r2) or slope < MIN_SLOPE_NM_PER_DEG:
                continue

            row = {
                "i0": int(i0),
                "i1": int(i1),
                "slope_nm_per_deg": float(slope),
                "intercept_nm": float(intercept),
                "r2": float(r2),
                "span_deg": span,
                "n_fit": int(i1 - i0 + 1),
            }

            is_initial_region = (float(xx[0]) <= (x_start_limit + 1e-12)) and (float(xx[-1]) <= (x_end_limit + 1e-12))
            strict_score = (span, r2, rise, -float(xx[0]), i1 - i0 + 1)
            fallback_score = (r2, rise, span, -float(xx[0]), i1 - i0 + 1)

            if r2 >= MIN_R2_STRICT:
                if is_initial_region and (best_strict_initial is None or strict_score > best_strict_initial_score):
                    best_strict_initial = row
                    best_strict_initial_score = strict_score
                if best_strict is None or strict_score > best_strict_score:
                    best_strict = row
                    best_strict_score = strict_score

            if is_initial_region and (best_fallback_initial is None or fallback_score > best_fallback_initial_score):
                best_fallback_initial = row
                best_fallback_initial_score = fallback_score
            if best_fallback is None or fallback_score > best_fallback_score:
                best_fallback = row
                best_fallback_score = fallback_score

    if best_strict_initial is not None:
        best = best_strict_initial
        best["fit_selection_mode"] = "strict_initial_window"
    elif best_fallback_initial is not None:
        best = best_fallback_initial
        best["fit_selection_mode"] = "fallback_initial_window"
    elif best_strict is not None:
        best = best_strict
        best["fit_selection_mode"] = "strict_long_window"
    elif best_fallback is not None:
        best = best_fallback
        best["fit_selection_mode"] = "fallback_best_r2"
    else:
        return {"ok": False, "reason": "no_valid_linear_window"}

    best["ok"] = True
    best["reason"] = "ok"
    return best


def analyze_cycle(rec: dict) -> dict:
    x_raw = np.asarray(rec["theta_rel"], dtype=float)
    y_raw = np.asarray(rec["moment_rel"], dtype=float)

    x_env, y_env, env_reason = extract_forward_loading_envelope(x_raw, y_raw)
    if env_reason != "ok":
        return {
            "cycle": int(rec["cycle"]),
            "ok": False,
            "reason": env_reason,
            "x_raw": x_raw,
            "y_raw": y_raw,
            "x_env": x_env,
            "y_env": y_env,
            "M_peak_dyn_Nm": float(rec["M_peak_dyn_Nm"]),
            "theta_span_ramp_deg": float(rec["theta_span_ramp_deg"]),
        }

    theta_mask = np.isfinite(x_env) & (x_env >= THETA_MIN_DEG)
    x_lin = x_env[theta_mask]
    y_lin = y_env[theta_mask]
    if x_lin.size < MIN_POINTS:
        return {
            "cycle": int(rec["cycle"]),
            "ok": False,
            "reason": "too_few_points_theta_nonnegative",
            "x_raw": x_raw,
            "y_raw": y_raw,
            "x_env": x_env,
            "y_env": y_env,
            "M_peak_dyn_Nm": float(rec["M_peak_dyn_Nm"]),
            "theta_span_ramp_deg": float(rec["theta_span_ramp_deg"]),
        }

    x_active, y_active, idx_active, flat_thr, flat_reason = remove_flat_subregions(x_lin, y_lin)
    if x_active.size < MIN_POINTS:
        return {
            "cycle": int(rec["cycle"]),
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
            "M_peak_dyn_Nm": float(rec["M_peak_dyn_Nm"]),
            "theta_span_ramp_deg": float(rec["theta_span_ramp_deg"]),
        }

    fit = find_best_linear_region(x_active, y_active, idx_active)
    if not fit["ok"]:
        return {
            "cycle": int(rec["cycle"]),
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
            "M_peak_dyn_Nm": float(rec["M_peak_dyn_Nm"]),
            "theta_span_ramp_deg": float(rec["theta_span_ramp_deg"]),
        }

    i0 = int(fit["i0"])
    i1 = int(fit["i1"])
    return {
        "cycle": int(rec["cycle"]),
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
        "i0": i0,
        "i1": i1,
        **fit,
        "x0_deg": float(x_active[i0]),
        "x1_deg": float(x_active[i1]),
        "m0_nm": float(y_active[i0]),
        "m1_nm": float(y_active[i1]),
        "flat_slope_thr_nm_per_deg": flat_thr,
        "flat_filter_reason": flat_reason,
        "M_peak_dyn_Nm": float(rec["M_peak_dyn_Nm"]),
        "theta_span_ramp_deg": float(rec["theta_span_ramp_deg"]),
    }


def plot_moment_theta_grid(
    results: list[dict],
    file_name: str,
    intervention: str,
    out_png: Path,
    xlim_deg: tuple[float, float] | None = None,
    ylim_nm: tuple[float, float] | None = None,
) -> None:
    n = len(results)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(5.8 * ncols, 4.0 * (nrows + 1)), constrained_layout=True)
    gs = fig.add_gridspec(nrows + 1, ncols, hspace=0.35, wspace=0.25)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)]
    axk = fig.add_subplot(gs[nrows, :])

    for i, res in enumerate(results):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]

        ax.plot(res["x_raw"], res["y_raw"], color="0.80", lw=1.2, label="raw")
        if res["x_env"].size:
            ax.plot(res["x_env"], res["y_env"], color="tab:blue", lw=1.5, label="forward envelope")
        if "x_active" in res and np.asarray(res["x_active"]).size:
            ax.plot(res["x_active"], res["y_active"], color="tab:green", lw=1.4, label="envelope (flat removed)")

        if res["ok"]:
            i0 = int(res["i0"])
            i1 = int(res["i1"])
            xx = res["x_active"][i0 : i1 + 1]
            yy = res["y_active"][i0 : i1 + 1]
            ax.plot(xx, yy, color="tab:red", lw=2.0, label="linear region")

            yfit = res["slope_nm_per_deg"] * xx + res["intercept_nm"]
            ax.plot(xx, yfit, color="tab:red", lw=1.2, ls=":", label="linear fit")

            title = f"cyc {res['cycle']} | k={res['slope_nm_per_deg']:.2f} Nm/deg | R2={res['r2']:.3f}"
        else:
            title = f"cyc {res['cycle']} | N/A ({res['reason']})"

        ax.set_title(title, fontsize=8)
        ax.set_xlabel("theta_rel [deg]")
        ax.set_ylabel("moment_rel [Nm]")
        if xlim_deg is not None:
            ax.set_xlim(*xlim_deg)
        if ylim_nm is not None:
            ax.set_ylim(*ylim_nm)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    cycles = np.asarray([int(res["cycle"]) for res in results], dtype=int)
    k_vals = np.asarray(
        [res.get("slope_nm_per_deg", np.nan) if res.get("ok", False) else np.nan for res in results],
        dtype=float,
    )
    axk.plot(cycles, k_vals, color="tab:blue", lw=1.4, marker="o", ms=4)

    valid = np.isfinite(k_vals)
    if np.any(valid):
        k_mean = float(np.nanmean(k_vals))
        k_std = float(np.nanstd(k_vals, ddof=0))
        axk.axhline(k_mean, color="tab:red", ls="--", lw=1.2, label=f"mean={k_mean:.2f} Nm/deg")
        axk.axhline(k_mean + k_std, color="tab:red", ls=":", lw=1.0)
        axk.axhline(k_mean - k_std, color="tab:red", ls=":", lw=1.0)
        axk.fill_between(cycles, k_mean - k_std, k_mean + k_std, color="tab:red", alpha=0.10, label=f"std={k_std:.2f}")
        axk.text(
            0.01,
            0.97,
            f"mean={k_mean:.2f}, std={k_std:.2f} Nm/deg (valid {int(valid.sum())}/{len(k_vals)})",
            transform=axk.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "0.75"},
        )
        axk.legend(loc="upper left", fontsize=8)

    axk.set_title("Cycle-wise linear-region stiffness from first to last cycle", fontsize=10)
    axk.set_xlabel("cycle number")
    axk.set_ylabel("k_linear [Nm/deg]")
    axk.set_xticks(cycles)
    axk.grid(alpha=0.25)

    case_disp = _display_case_name(file_name, intervention)
    fig.suptitle(
        f"{case_disp}: moment vs theta for all loading cycles\n"
        "Forward envelope + linear-region regression using theta >= 0 deg fit points; no fixed 1-4 Nm anchors",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"saved {out_png}")


def save_metrics(results: list[dict], out_csv: Path, file_name: str, intervention: str) -> None:
    rows = []
    for res in results:
        rows.append(
            {
                "file_name": file_name,
                "intervention": intervention,
                "cycle": int(res["cycle"]),
                "valid": bool(res["ok"]),
                "reason": str(res["reason"]),
                "M_peak_dyn_Nm": float(res.get("M_peak_dyn_Nm", np.nan)),
                "theta_span_ramp_deg": float(res.get("theta_span_ramp_deg", np.nan)),
                "k_linear_nm_per_deg": float(res["slope_nm_per_deg"]) if res.get("ok", False) else np.nan,
                "intercept_nm": float(res["intercept_nm"]) if res.get("ok", False) else np.nan,
                "r2": float(res["r2"]) if res.get("ok", False) else np.nan,
                "fit_n_points": int(res["n_fit"]) if res.get("ok", False) else np.nan,
                "fit_span_deg": float(res["span_deg"]) if res.get("ok", False) else np.nan,
                "fit_theta0_deg": float(res["x0_deg"]) if res.get("ok", False) else np.nan,
                "fit_theta1_deg": float(res["x1_deg"]) if res.get("ok", False) else np.nan,
                "fit_m0_nm": float(res["m0_nm"]) if res.get("ok", False) else np.nan,
                "fit_m1_nm": float(res["m1_nm"]) if res.get("ok", False) else np.nan,
                "fit_selection_mode": str(res.get("fit_selection_mode", "")),
                "flat_slope_thr_nm_per_deg": float(res["flat_slope_thr_nm_per_deg"])
                if "flat_slope_thr_nm_per_deg" in res and np.isfinite(res["flat_slope_thr_nm_per_deg"])
                else np.nan,
                "flat_filter_reason": str(res.get("flat_filter_reason", "")),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transition_flags = load_transition_flag_map()

    tracker = base.ai.tracker_cases().copy()
    tracker["File Name"] = tracker["File Name"].astype(str).str.strip()
    tracker["Intervention"] = tracker["Intervention"].astype(str).str.strip()

    all_rows: list[dict] = []
    summary_rows: list[dict] = []

    for file_name in ass.CASE_ORDER:
        sub = tracker[tracker["File Name"] == file_name]
        if sub.empty:
            print(f"skip (not found in tracker): {file_name}")
            continue

        intervention = str(sub.iloc[0]["Intervention"])
        print(f"\n=== {file_name} | {intervention} ===")

        cycles = base.build_cycle_rows(file_name, intervention)
        if not cycles:
            print("no cycles produced")
            continue

        results = [analyze_cycle(rec) for rec in cycles]
        results = apply_transition_exclusions(results, file_name, transition_flags)

        case_ymin = np.inf
        case_ymax = -np.inf
        for res in results:
            y_raw = np.asarray(res.get("y_raw", np.array([])), dtype=float)
            if y_raw.size > 0 and np.any(np.isfinite(y_raw)):
                case_ymin = min(case_ymin, float(np.nanmin(y_raw)))
                case_ymax = max(case_ymax, float(np.nanmax(y_raw)))

        case_xlim_deg = THETA_XLIM_DEG

        if np.isfinite(case_ymin) and np.isfinite(case_ymax) and case_ymax > case_ymin:
            y_span = case_ymax - case_ymin
            y_pad = max(0.5, 0.06 * y_span)
            case_ylim_nm = (case_ymin - y_pad, case_ymax + y_pad)
        else:
            case_ylim_nm = None

        stem = base.sanitize(file_name)
        out_png = OUT_DIR / f"{stem}_all_cycles_moment_theta.png"
        out_csv = OUT_DIR / f"{stem}_all_cycles_moment_theta_metrics.csv"

        plot_moment_theta_grid(
            results,
            _display_case_name(file_name, intervention),
            _clean_intervention(intervention),
            out_png,
            xlim_deg=case_xlim_deg,
            ylim_nm=case_ylim_nm,
        )
        save_metrics(results, out_csv, file_name, intervention)

        print(
            f"{file_name} theta x-limits [deg]: "
            f"{case_xlim_deg[0]:.2f} to {case_xlim_deg[1]:.2f}"
        )
        if case_ylim_nm is not None:
            print(
                f"{file_name} moment y-limits [Nm]: "
                f"{case_ylim_nm[0]:.2f} to {case_ylim_nm[1]:.2f}"
            )

        for res in results:
            all_rows.append(
                {
                    "file_name": file_name,
                    "intervention": intervention,
                    "cycle": int(res["cycle"]),
                    "valid": bool(res["ok"]),
                    "reason": str(res["reason"]),
                    "k_linear_nm_per_deg": float(res["slope_nm_per_deg"]) if res.get("ok", False) else np.nan,
                    "r2": float(res["r2"]) if res.get("ok", False) else np.nan,
                    "fit_span_deg": float(res["span_deg"]) if res.get("ok", False) else np.nan,
                    "fit_selection_mode": str(res.get("fit_selection_mode", "")),
                }
            )

        k_vals = np.asarray(
            [res.get("slope_nm_per_deg", np.nan) if res.get("ok", False) else np.nan for res in results],
            dtype=float,
        )
        r2_vals = np.asarray(
            [res.get("r2", np.nan) if res.get("ok", False) else np.nan for res in results],
            dtype=float,
        )
        valid = np.isfinite(k_vals)

        summary_rows.append(
            {
                "file_name": file_name,
                "intervention": intervention,
                "n_cycles": int(len(results)),
                "n_valid": int(np.count_nonzero(valid)),
                "k_mean_nm_per_deg": float(np.nanmean(k_vals)) if np.any(valid) else np.nan,
                "k_median_nm_per_deg": float(np.nanmedian(k_vals)) if np.any(valid) else np.nan,
                "k_std_nm_per_deg": float(np.nanstd(k_vals, ddof=0)) if np.any(valid) else np.nan,
                "r2_mean": float(np.nanmean(r2_vals)) if np.any(np.isfinite(r2_vals)) else np.nan,
            }
        )

    if all_rows:
        out_all = OUT_DIR / "all_interventions_moment_theta_metrics.csv"
        pd.DataFrame(all_rows).to_csv(out_all, index=False)
        print(f"saved {out_all}")

    if summary_rows:
        out_summary = OUT_DIR / "all_interventions_moment_theta_stiffness_summary.csv"
        pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
        print(f"saved {out_summary}")


if __name__ == "__main__":
    main()
