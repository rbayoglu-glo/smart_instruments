from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_intact_holding_all_cycles_force_tip_displacement as src
import plot_theta_moment_time_all_interventions as base


OUT_DIR = Path(__file__).parent / "output" / "intact_holding_cycle_traces"
FILE_NAME = "2 - Intact Holding Longer"
DISPLACEMENT_MIN_MM = 0.0
MIN_POINTS = 8
MIN_SPAN_MM = 2.0
MIN_FORCE_RISE_N = 5.0
MIN_SLOPE_N_PER_MM = 0.5
MIN_R2_STRICT = 0.97
FLAT_EDGE_SLOPE_MAX_N_PER_MM = 1.0
FLAT_EDGE_SLOPE_FRACTION_OF_GLOBAL = 0.30
MIN_FLAT_RUN_SPAN_MM = 1.2
INITIAL_REGION_START_FRAC_MAX = 0.35
INITIAL_REGION_END_FRAC_MAX = 0.80


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


def _line_fit_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


def extract_forward_loading_envelope(tip_mm: np.ndarray, force_n: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Upper loading envelope up to peak force, monotonic in displacement and force."""
    x = np.asarray(tip_mm, dtype=float)
    y = np.asarray(force_n, dtype=float)
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

    # Collapse duplicate displacements using max force, then enforce monotonic force envelope.
    x_u, idx = np.unique(x, return_index=True)
    y_max = np.maximum.reduceat(y, idx)
    y_env = np.maximum.accumulate(y_max)

    if x_u.size < 2:
        return np.array([]), np.array([]), "envelope_too_short"

    return x_u, y_env, "ok"


def remove_flat_subregions(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Drop long flat runs from the envelope before linear fitting.

    Returns filtered x/y, kept original indices, used slope threshold, and reason.
    """
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
        FLAT_EDGE_SLOPE_MAX_N_PER_MM,
        FLAT_EDGE_SLOPE_FRACTION_OF_GLOBAL * global_slope if np.isfinite(global_slope) else FLAT_EDGE_SLOPE_MAX_N_PER_MM,
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
        if run_span >= MIN_FLAT_RUN_SPAN_MM:
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
            # Do not bridge across removed flat segments.
            if int(idx_orig[i1] - idx_orig[i0]) != int(i1 - i0):
                continue

            span = float(x[i1] - x[i0])
            if span < MIN_SPAN_MM:
                continue

            xx = x[i0 : i1 + 1]
            yy = y[i0 : i1 + 1]
            rise = float(yy[-1] - yy[0])
            if rise < MIN_FORCE_RISE_N:
                continue

            slope, intercept, r2 = _line_fit_stats(xx, yy)
            if not np.isfinite(slope) or not np.isfinite(r2) or slope < MIN_SLOPE_N_PER_MM:
                continue

            row = {
                    "i0": int(i0),
                    "i1": int(i1),
                    "slope_n_per_mm": float(slope),
                    "intercept_n": float(intercept),
                    "r2": float(r2),
                    "span_mm": span,
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
    x_raw = np.asarray(rec["tip_rel_mm"], dtype=float)
    y_raw = np.asarray(rec["force_dyn_N"], dtype=float)
    theta_start_deg = float(rec.get("theta_start_deg", np.nan))
    non_neutral_start = bool(rec.get("non_neutral_start", False))

    if non_neutral_start:
        return {
            "cycle": int(rec["cycle"]),
            "ok": False,
            "reason": "non_neutral_start_excluded",
            "x_raw": x_raw,
            "y_raw": y_raw,
            "x_env": np.array([]),
            "y_env": np.array([]),
            "theta_start_deg": theta_start_deg,
            "non_neutral_start": non_neutral_start,
        }

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
            "theta_start_deg": theta_start_deg,
            "non_neutral_start": non_neutral_start,
        }

    displacement_mask = np.isfinite(x_env) & (x_env >= DISPLACEMENT_MIN_MM)
    x_lin = x_env[displacement_mask]
    y_lin = y_env[displacement_mask]
    if x_lin.size < MIN_POINTS:
        return {
            "cycle": int(rec["cycle"]),
            "ok": False,
            "reason": "too_few_points_displacement_nonnegative",
            "x_raw": x_raw,
            "y_raw": y_raw,
            "x_env": x_env,
            "y_env": y_env,
            "theta_start_deg": theta_start_deg,
            "non_neutral_start": non_neutral_start,
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
            "flat_slope_thr": flat_thr,
            "flat_filter_reason": flat_reason,
            "theta_start_deg": theta_start_deg,
            "non_neutral_start": non_neutral_start,
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
            "flat_slope_thr": flat_thr,
            "flat_filter_reason": flat_reason,
            "theta_start_deg": theta_start_deg,
            "non_neutral_start": non_neutral_start,
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
        "x0_mm": float(x_active[i0]),
        "x1_mm": float(x_active[i1]),
        "f0_n": float(y_active[i0]),
        "f1_n": float(y_active[i1]),
        "flat_slope_thr": flat_thr,
        "flat_filter_reason": flat_reason,
        "theta_start_deg": theta_start_deg,
        "non_neutral_start": non_neutral_start,
    }


def plot_cycle_grid(
    results: list[dict],
    file_name: str,
    intervention: str,
    out_png: Path,
    xlim_mm: tuple[float, float] | None = None,
    ylim_n: tuple[float, float] | None = None,
) -> None:
    n = len(results)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(5.9 * ncols, 4.0 * (nrows + 1)), constrained_layout=True)
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

            yfit = res["slope_n_per_mm"] * xx + res["intercept_n"]
            ax.plot(xx, yfit, color="tab:red", lw=1.2, ls=":", label="linear fit")

            title = (
                f"cyc {res['cycle']} | k={res['slope_n_per_mm']:.2f} N/mm | "
                f"R2={res['r2']:.3f}"
            )
        else:
            title = f"cyc {res['cycle']} | N/A ({res['reason']})"

        ax.set_title(title, fontsize=8)
        ax.set_xlabel("tip displacement [mm]")
        ax.set_ylabel("force without preload [N]")
        if ylim_n is not None:
            ax.set_ylim(*ylim_n)
        if xlim_mm is not None:
            ax.set_xlim(*xlim_mm)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    cycles = np.asarray([int(r["cycle"]) for r in results], dtype=int)
    kvals = np.asarray([r.get("slope_n_per_mm", np.nan) if r.get("ok", False) else np.nan for r in results], dtype=float)

    axk.plot(cycles, kvals, color="tab:blue", lw=1.4, marker="o", ms=4)
    valid = np.isfinite(kvals)
    if np.any(valid):
        k_mean = float(np.nanmean(kvals))
        k_std = float(np.nanstd(kvals, ddof=0))
        axk.axhline(k_mean, color="tab:red", ls="--", lw=1.2, label=f"mean={k_mean:.2f} N/mm")
        axk.axhline(k_mean + k_std, color="tab:red", ls=":", lw=1.0)
        axk.axhline(k_mean - k_std, color="tab:red", ls=":", lw=1.0)
        axk.fill_between(cycles, k_mean - k_std, k_mean + k_std, color="tab:red", alpha=0.10, label=f"std={k_std:.2f}")
        axk.text(
            0.01,
            0.97,
            f"mean={k_mean:.2f}, std={k_std:.2f} N/mm (valid {int(valid.sum())}/{len(kvals)})",
            transform=axk.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "0.75"},
        )

    axk.set_title("Cycle-wise linear-region stiffness from first to last cycle", fontsize=10)
    axk.set_xlabel("cycle number")
    axk.set_ylabel("k [N/mm]")
    axk.set_xticks(cycles)
    axk.grid(alpha=0.25)
    handles, labels = axk.get_legend_handles_labels()
    if handles:
        axk.legend(loc="upper left", fontsize=8)

    case_disp = _display_case_name(file_name, intervention)
    fig.suptitle(
        f"{case_disp}\n"
        "Forward-loading envelope + linear regression stiffness (displacement >= 0 mm for fit points)\n"
        f"strict fit prefers larger windows (span >= {MIN_SPAN_MM:.1f} mm, R2 >= {MIN_R2_STRICT:.2f}); long flat runs removed",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"saved {out_png}")


def save_metrics(results: list[dict], out_csv: Path) -> None:
    rows = []
    for res in results:
        rows.append(
            {
                "cycle": int(res["cycle"]),
                "valid": bool(res["ok"]),
                "reason": str(res["reason"]),
                "k_linear_n_per_mm": float(res["slope_n_per_mm"]) if res.get("ok", False) else np.nan,
                "intercept_n": float(res["intercept_n"]) if res.get("ok", False) else np.nan,
                "r2": float(res["r2"]) if res.get("ok", False) else np.nan,
                "fit_n_points": int(res["n_fit"]) if res.get("ok", False) else np.nan,
                "fit_span_mm": float(res["span_mm"]) if res.get("ok", False) else np.nan,
                "fit_x0_mm": float(res["x0_mm"]) if res.get("ok", False) else np.nan,
                "fit_x1_mm": float(res["x1_mm"]) if res.get("ok", False) else np.nan,
                "fit_f0_n": float(res["f0_n"]) if res.get("ok", False) else np.nan,
                "fit_f1_n": float(res["f1_n"]) if res.get("ok", False) else np.nan,
                "fit_selection_mode": str(res["fit_selection_mode"]) if res.get("ok", False) else "",
                "flat_slope_thr_n_per_mm": float(res["flat_slope_thr"]) if "flat_slope_thr" in res and np.isfinite(res["flat_slope_thr"]) else np.nan,
                "flat_filter_reason": str(res["flat_filter_reason"]) if "flat_filter_reason" in res else "",
                "theta_start_deg": float(res["theta_start_deg"]) if "theta_start_deg" in res and np.isfinite(res["theta_start_deg"]) else np.nan,
                "non_neutral_start": bool(res.get("non_neutral_start", False)),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tracker = base.ai.tracker_cases().copy()
    tracker["File Name"] = tracker["File Name"].astype(str).str.strip()
    tracker["Intervention"] = tracker["Intervention"].astype(str).str.strip()

    sub = tracker[tracker["File Name"] == FILE_NAME]
    if sub.empty:
        raise RuntimeError(f"Case not found in tracker: {FILE_NAME}")

    intervention = str(sub.iloc[0]["Intervention"])
    print(f"=== {FILE_NAME} | {intervention} ===")

    cycles = src.build_cycle_rows_force_tip(FILE_NAME, intervention)
    if not cycles:
        raise RuntimeError("No cycles produced for intact holding case.")

    results = [analyze_cycle(rec) for rec in cycles]

    out_png = OUT_DIR / "intact_holding_all_cycles_force_tip_linear_region_stiffness.png"
    out_csv = OUT_DIR / "intact_holding_all_cycles_force_tip_linear_region_stiffness_metrics.csv"

    plot_cycle_grid(results, FILE_NAME, intervention, out_png)
    save_metrics(results, out_csv)

    df = pd.read_csv(out_csv)
    valid = df[df["valid"]]
    if not valid.empty:
        k_mean = float(valid["k_linear_n_per_mm"].mean())
        k_std = float(valid["k_linear_n_per_mm"].std(ddof=0))
        print(
            f"valid cycles: {len(valid)}/{len(df)} | "
            f"k mean={k_mean:.3f} N/mm, std={k_std:.3f} N/mm"
        )


if __name__ == "__main__":
    main()
