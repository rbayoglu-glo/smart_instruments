from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_spreader_secant_stiffness as ass
import lamina_spreader as ls
import plot_theta_moment_time_all_interventions as base


OUT_DIR = Path(__file__).parent / "output" / "intact_holding_cycle_traces"
FILE_NAME = "2 - Intact Holding Longer"

M_SECANT_NM = (1.0, 4.0)
F_SECANT_N = np.asarray([ls.moment_to_force_n(m) for m in M_SECANT_NM], dtype=float)


def marker_tip_at_force_targets(
    tip_rel_mm: np.ndarray,
    force_dyn_n: np.ndarray,
    force_targets_n: np.ndarray,
) -> dict:
    """Interpolate tip displacement at target forces on the final loading climb."""
    if tip_rel_mm.size < 2 or force_dyn_n.size < 2:
        return {
            "tip_at_targets_mm": np.full(force_targets_n.shape, np.nan, dtype=float),
            "k_1to4_n_per_mm": np.nan,
            "marker_reason": "too_few_points",
        }

    x_all = np.asarray(tip_rel_mm, dtype=float)
    f_all = np.asarray(force_dyn_n, dtype=float)
    keep = np.isfinite(x_all) & np.isfinite(f_all)
    x_all = x_all[keep]
    f_all = f_all[keep]
    if x_all.size < 2:
        return {
            "tip_at_targets_mm": np.full(force_targets_n.shape, np.nan, dtype=float),
            "k_1to4_n_per_mm": np.nan,
            "marker_reason": "all_nonfinite",
        }

    pk = int(np.nanargmax(f_all))
    if pk < 1:
        return {
            "tip_at_targets_mm": np.full(force_targets_n.shape, np.nan, dtype=float),
            "k_1to4_n_per_mm": np.nan,
            "marker_reason": "peak_too_early",
        }

    low_target = float(np.nanmin(force_targets_n))
    high_target = float(np.nanmax(force_targets_n))

    below = np.flatnonzero(f_all[: pk + 1] <= (low_target + 1e-9))
    if below.size == 0:
        return {
            "tip_at_targets_mm": np.full(force_targets_n.shape, np.nan, dtype=float),
            "k_1to4_n_per_mm": np.nan,
            "marker_reason": "no_flow_crossing_before_peak",
        }
    i_start = int(below[-1])

    x_seg = x_all[i_start : pk + 1]
    f_seg = f_all[i_start : pk + 1]
    if x_seg.size < 2 or float(np.nanmax(f_seg)) < high_target:
        return {
            "tip_at_targets_mm": np.full(force_targets_n.shape, np.nan, dtype=float),
            "k_1to4_n_per_mm": np.nan,
            "marker_reason": "fhigh_not_reached_on_final_climb",
        }

    hull = f_seg >= np.maximum.accumulate(f_seg) - 1e-9
    x = x_seg[hull]
    f = f_seg[hull]
    if x.size < 2:
        return {
            "tip_at_targets_mm": np.full(force_targets_n.shape, np.nan, dtype=float),
            "k_1to4_n_per_mm": np.nan,
            "marker_reason": "hull_too_short",
        }

    order = np.argsort(f)
    f = f[order]
    x = x[order]

    f_u, first = np.unique(f, return_index=True)
    x_u = x[first]
    if f_u.size < 2:
        return {
            "tip_at_targets_mm": np.full(force_targets_n.shape, np.nan, dtype=float),
            "k_1to4_n_per_mm": np.nan,
            "marker_reason": "unique_f_too_short",
        }

    if low_target < float(np.min(f_u)) or high_target > float(np.max(f_u)):
        return {
            "tip_at_targets_mm": np.full(force_targets_n.shape, np.nan, dtype=float),
            "k_1to4_n_per_mm": np.nan,
            "marker_reason": "window_outside_final_climb",
        }

    x_targets = np.interp(force_targets_n, f_u, x_u)
    dx_1to4 = float(x_targets[-1] - x_targets[0])
    if not np.isfinite(dx_1to4) or dx_1to4 <= 1e-9:
        return {
            "tip_at_targets_mm": x_targets,
            "k_1to4_n_per_mm": np.nan,
            "marker_reason": "dx_too_small",
        }

    k_1to4 = float((force_targets_n[-1] - force_targets_n[0]) / dx_1to4)
    return {
        "tip_at_targets_mm": x_targets,
        "k_1to4_n_per_mm": k_1to4,
        "marker_reason": "ok",
    }


def build_cycle_rows_force_tip(file_name: str, intervention: str) -> list[dict]:
    kin = ls.compute_theta(
        ls.parse_poses(base.ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID)
    )
    force = ls.load_force(base.ai.find_force_csv(file_name))
    lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))
    d, _, fmax = ls.sync(kin, force, lag)

    f = d["force"].to_numpy()
    th = d["theta_dist"].to_numpy()

    rows: list[dict] = []
    runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
    for n, (s, e) in enumerate(runs, 1):
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue

        i0, i1, pk = rb
        ramp = np.arange(i0, i1)
        if ramp.size < 4:
            continue

        f0 = float(f[i0])
        f_dyn = f[ramp] - f0
        if float(np.nanmax(f_dyn)) <= 0:
            continue

        early_n = min(ass.FORCE_ZERO_MAX_SAMPLES, len(ramp))
        early = np.arange(early_n)
        zero_idx = early[np.abs(f_dyn[early]) <= ass.FORCE_ZERO_BAND_N]

        if zero_idx.size >= ass.FORCE_ZERO_MIN_SAMPLES:
            th_ref = float(np.nanmedian(th[ramp[zero_idx]]))
        else:
            iref0 = max(0, i0 - ass.THETA_REF_WINDOW + 1)
            th_ref = float(np.nanmedian(th[iref0:i0 + 1]))

        if not np.isfinite(th_ref):
            th_ref = float(th[i0])

        theta_rel = th[ramp] - th_ref
        tip_rel_mm = ls.theta_deg_to_tip_mm(theta_rel)
        mark = marker_tip_at_force_targets(tip_rel_mm, f_dyn, F_SECANT_N)

        rows.append(
            {
                "file_name": file_name,
                "intervention": intervention,
                "cycle": int(n),
                "force_dyn_N": f_dyn,
                "tip_rel_mm": tip_rel_mm,
                "F_peak_dyn_N": float(np.nanmax(f_dyn)),
                "tip_span_mm": float(np.ptp(tip_rel_mm)),
                "tip_at_peak_mm": float(tip_rel_mm[int(np.nanargmax(f_dyn))]),
                "tip_at_markers_mm": mark["tip_at_targets_mm"],
                "k_1to4_n_per_mm": float(mark["k_1to4_n_per_mm"]) if np.isfinite(mark["k_1to4_n_per_mm"]) else np.nan,
                "marker_reason": str(mark["marker_reason"]),
            }
        )

    return rows


def plot_force_tip_grid(rows: list[dict], file_name: str, intervention: str, out_png: Path) -> None:
    n = len(rows)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(5.8 * ncols, 4.0 * (nrows + 1)), constrained_layout=True)
    gs = fig.add_gridspec(nrows + 1, ncols, hspace=0.35, wspace=0.25)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)]
    axk = fig.add_subplot(gs[nrows, :])

    for i, rec in enumerate(rows):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]

        x = rec["tip_rel_mm"]
        y = rec["force_dyn_N"]
        ax.plot(x, y, color="tab:blue", lw=1.5)
        ax.axvline(0.0, color="0.7", lw=0.8)
        ax.axhline(0.0, color="0.7", lw=0.8)

        # Moment-equivalent force guides for 1 and 4 Nm.
        for m_nm, f_n in zip(M_SECANT_NM, F_SECANT_N):
            ax.axhline(f_n, color="tab:red", lw=0.8, ls="--", alpha=0.35)

        i_pk = int(np.nanargmax(y))
        ax.scatter([x[i_pk]], [y[i_pk]], s=26, color="tab:red", zorder=5)
        ax.text(x[i_pk], y[i_pk], f" peak={y[i_pk]:.1f}N", color="tab:red", fontsize=7,
                ha="left", va="bottom")

        x_marks = rec["tip_at_markers_mm"]
        for j, (m_nm, f_n) in enumerate(zip(M_SECANT_NM, F_SECANT_N)):
            if np.isfinite(x_marks[j]):
                ax.scatter([x_marks[j]], [f_n], s=24, marker="D", color="tab:red", zorder=6)
                ax.text(x_marks[j], f_n, f" {m_nm:.0f}Nm", color="tab:red", fontsize=7,
                        ha="left", va="bottom")

        if np.isfinite(rec["k_1to4_n_per_mm"]):
            k_text = f"k(1-4Nm)={rec['k_1to4_n_per_mm']:.2f} N/mm"
        else:
            k_text = f"k(1-4Nm)=N/A ({rec['marker_reason']})"
        ax.text(
            0.02,
            0.98,
            k_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.75"},
        )

        ax.set_title(
            f"cyc {int(rec['cycle'])} | peakF={rec['F_peak_dyn_N']:.1f}N | tip span={rec['tip_span_mm']:.2f}mm",
            fontsize=8,
        )
        ax.set_xlabel("tip displacement from ramp-start ref [mm]")
        ax.set_ylabel("force without preload [N]")
        ax.grid(alpha=0.25)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    cycles = np.asarray([int(rec["cycle"]) for rec in rows], dtype=int)
    k_vals = np.asarray([float(rec["k_1to4_n_per_mm"]) if np.isfinite(rec["k_1to4_n_per_mm"]) else np.nan for rec in rows], dtype=float)
    axk.plot(cycles, k_vals, color="tab:blue", lw=1.4, marker="o", ms=4)

    valid = np.isfinite(k_vals)
    if np.any(valid):
        k_mean = float(np.nanmean(k_vals))
        k_std = float(np.nanstd(k_vals, ddof=0))
        axk.axhline(k_mean, color="tab:red", ls="--", lw=1.2, label=f"mean={k_mean:.2f} N/mm")
        axk.axhline(k_mean + k_std, color="tab:red", ls=":", lw=1.0)
        axk.axhline(k_mean - k_std, color="tab:red", ls=":", lw=1.0)
        axk.fill_between(cycles, k_mean - k_std, k_mean + k_std, color="tab:red", alpha=0.10, label=f"std={k_std:.2f}")
        axk.text(
            0.01,
            0.97,
            f"mean={k_mean:.2f}, std={k_std:.2f} N/mm (valid {int(valid.sum())}/{len(k_vals)})",
            transform=axk.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "0.75"},
        )

    axk.set_title("Cycle-wise k(1-4Nm) from first to last cycle", fontsize=10)
    axk.set_xlabel("cycle number")
    axk.set_ylabel("k_1to4 [N/mm]")
    axk.set_xticks(cycles)
    axk.grid(alpha=0.25)
    axk.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"{file_name} ({intervention}): preload-free force vs tip displacement for all loading cycles\n"
        "No exclusion filtering applied; red diamond markers show 1 Nm and 4 Nm points",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"saved {out_png}")


def save_metrics(rows: list[dict], out_csv: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "file_name": rec["file_name"],
                "intervention": rec["intervention"],
                "cycle": rec["cycle"],
                "F_peak_dyn_N": rec["F_peak_dyn_N"],
                "tip_span_mm": rec["tip_span_mm"],
                "tip_at_peak_mm": rec["tip_at_peak_mm"],
                "tip_at_1Nm_mm": rec["tip_at_markers_mm"][0],
                "tip_at_4Nm_mm": rec["tip_at_markers_mm"][1],
                "k_1to4_n_per_mm": rec["k_1to4_n_per_mm"],
                "marker_reason": rec["marker_reason"],
            }
            for rec in rows
        ]
    )
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

    rows = build_cycle_rows_force_tip(FILE_NAME, intervention)
    if not rows:
        raise RuntimeError("No cycles produced for intact holding case.")

    out_png = OUT_DIR / "intact_holding_all_cycles_force_tip_displacement.png"
    out_csv = OUT_DIR / "intact_holding_all_cycles_force_tip_displacement_metrics.csv"

    plot_force_tip_grid(rows, FILE_NAME, intervention, out_png)
    save_metrics(rows, out_csv)


if __name__ == "__main__":
    main()
