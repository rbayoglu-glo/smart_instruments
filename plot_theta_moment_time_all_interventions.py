#%%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_interventions as ai
import analyze_spreader_secant_stiffness as ass
import lamina_spreader as ls
import spreader_cycle_pipeline as spc


OUT_DIR = Path(__file__).parent / "output" / "theta_moment_time_cycle_grids"
OUT_DIR_INTACT = Path(__file__).parent / "output" / "intact_holding_cycle_traces"

M_GUIDES_NM = (1.0, 4.0)
M_LOW = float(M_GUIDES_NM[0])
M_HIGH = float(M_GUIDES_NM[1])


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def old_secant(theta_rel_deg: np.ndarray, moment_nm: np.ndarray, m_low: float, m_high: float):
    """Legacy secant extraction for side-by-side diagnostics in plot titles."""
    if theta_rel_deg.size < 2 or moment_nm.size < 2:
        return np.nan, np.nan, np.nan, np.nan

    hull = moment_nm >= np.maximum.accumulate(moment_nm) - 1e-9
    m = moment_nm[hull]
    a = theta_rel_deg[hull]
    if m.size < 2:
        return np.nan, np.nan, np.nan, np.nan

    order = np.argsort(m)
    m = m[order]
    a = a[order]

    m_u, first = np.unique(m, return_index=True)
    a_u = a[first]
    if m_u.size < 2:
        return np.nan, np.nan, np.nan, np.nan

    if m_low < float(np.min(m_u)) or m_high > float(np.max(m_u)):
        return np.nan, np.nan, np.nan, np.nan

    th_low = float(np.interp(m_low, m_u, a_u))
    th_high = float(np.interp(m_high, m_u, a_u))
    dtheta = th_high - th_low
    if not np.isfinite(dtheta) or dtheta <= 1e-10:
        return np.nan, th_low, th_high, dtheta

    k_sec = float((m_high - m_low) / dtheta)
    return k_sec, th_low, th_high, dtheta


def marker_points_with_time(
    theta_rel_deg: np.ndarray,
    moment_nm: np.ndarray,
    t_rel_s: np.ndarray,
    m_low: float,
    m_high: float,
) -> dict:
    """Interpolate theta/time at target moments on final loading climb."""
    if theta_rel_deg.size < 2 or moment_nm.size < 2 or t_rel_s.size < 2:
        return {
            "theta_at_m_low_deg": np.nan,
            "theta_at_m_high_deg": np.nan,
            "t_at_m_low_s": np.nan,
            "t_at_m_high_s": np.nan,
            "marker_reason": "too_few_points",
        }

    m_all = np.asarray(moment_nm, dtype=float)
    a_all = np.asarray(theta_rel_deg, dtype=float)
    t_all = np.asarray(t_rel_s, dtype=float)
    keep = np.isfinite(m_all) & np.isfinite(a_all) & np.isfinite(t_all)
    m_all = m_all[keep]
    a_all = a_all[keep]
    t_all = t_all[keep]
    if m_all.size < 2:
        return {
            "theta_at_m_low_deg": np.nan,
            "theta_at_m_high_deg": np.nan,
            "t_at_m_low_s": np.nan,
            "t_at_m_high_s": np.nan,
            "marker_reason": "all_nonfinite",
        }

    pk = int(np.nanargmax(m_all))
    if pk < 1:
        return {
            "theta_at_m_low_deg": np.nan,
            "theta_at_m_high_deg": np.nan,
            "t_at_m_low_s": np.nan,
            "t_at_m_high_s": np.nan,
            "marker_reason": "peak_too_early",
        }

    below = np.flatnonzero(m_all[:pk + 1] <= (m_low + 1e-9))
    if below.size == 0:
        return {
            "theta_at_m_low_deg": np.nan,
            "theta_at_m_high_deg": np.nan,
            "t_at_m_low_s": np.nan,
            "t_at_m_high_s": np.nan,
            "marker_reason": "no_mlow_crossing_before_peak",
        }
    i_start = int(below[-1])

    m_seg = m_all[i_start:pk + 1]
    a_seg = a_all[i_start:pk + 1]
    t_seg = t_all[i_start:pk + 1]
    if m_seg.size < 2 or float(np.nanmax(m_seg)) < m_high:
        return {
            "theta_at_m_low_deg": np.nan,
            "theta_at_m_high_deg": np.nan,
            "t_at_m_low_s": np.nan,
            "t_at_m_high_s": np.nan,
            "marker_reason": "mhigh_not_reached_on_final_climb",
        }

    hull = m_seg >= np.maximum.accumulate(m_seg) - 1e-9
    m = m_seg[hull]
    a = a_seg[hull]
    tt = t_seg[hull]
    if m.size < 2:
        return {
            "theta_at_m_low_deg": np.nan,
            "theta_at_m_high_deg": np.nan,
            "t_at_m_low_s": np.nan,
            "t_at_m_high_s": np.nan,
            "marker_reason": "hull_too_short",
        }

    order = np.argsort(m)
    m = m[order]
    a = a[order]
    tt = tt[order]

    m_u, first = np.unique(m, return_index=True)
    a_u = a[first]
    t_u = tt[first]
    if m_u.size < 2:
        return {
            "theta_at_m_low_deg": np.nan,
            "theta_at_m_high_deg": np.nan,
            "t_at_m_low_s": np.nan,
            "t_at_m_high_s": np.nan,
            "marker_reason": "unique_m_too_short",
        }

    if m_low < float(np.min(m_u)) or m_high > float(np.max(m_u)):
        return {
            "theta_at_m_low_deg": np.nan,
            "theta_at_m_high_deg": np.nan,
            "t_at_m_low_s": np.nan,
            "t_at_m_high_s": np.nan,
            "marker_reason": "window_outside_final_climb",
        }

    th_low = float(np.interp(m_low, m_u, a_u))
    th_high = float(np.interp(m_high, m_u, a_u))
    t_low = float(np.interp(m_low, m_u, t_u))
    t_high = float(np.interp(m_high, m_u, t_u))
    return {
        "theta_at_m_low_deg": th_low,
        "theta_at_m_high_deg": th_high,
        "t_at_m_low_s": t_low,
        "t_at_m_high_s": t_high,
        "marker_reason": "ok",
    }


def build_cycle_rows(file_name: str, intervention: str) -> list[dict]:
    case = spc.load_case(file_name)
    t = case["t"]
    f_dyn = case["f_dyn"]
    th_filt = case["th_filt"]

    rows: list[dict] = []
    for n, c in enumerate(case["cycles"], 1):
        i0 = int(c["i0"])
        pk = int(c["pk"])
        ramp = np.arange(i0, pk + 1)
        if ramp.size < 4:
            continue

        f_dyn_ramp = f_dyn[ramp]
        if not np.any(np.isfinite(f_dyn_ramp)) or float(np.nanmax(f_dyn_ramp)) <= 0:
            continue

        theta_rel = th_filt[ramp]
        moment_rel = ls.force_to_moment_nm(f_dyn_ramp)
        t_rel = t[ramp] - t[ramp[0]]

        old_k, _, _, old_dtheta = old_secant(theta_rel, moment_rel, M_LOW, M_HIGH)
        new_k, _, _, new_dtheta, new_reason = ass.secant_on_rising_branch(
            theta_rel, moment_rel, M_LOW, M_HIGH
        )
        mark = marker_points_with_time(theta_rel, moment_rel, t_rel, M_LOW, M_HIGH)

        rows.append(
            {
                "file_name": file_name,
                "intervention": intervention,
                "cycle": int(n),
                "t_rel": t_rel,
                "theta_rel": theta_rel,
                "moment_rel": moment_rel,
                "M_peak_dyn_Nm": float(np.nanmax(moment_rel)),
                "theta_span_ramp_deg": float(np.ptp(theta_rel)),
                "old_dtheta_deg": float(old_dtheta) if np.isfinite(old_dtheta) else np.nan,
                "new_dtheta_deg": float(new_dtheta) if np.isfinite(new_dtheta) else np.nan,
                "old_k_sec": float(old_k) if np.isfinite(old_k) else np.nan,
                "new_k_sec": float(new_k) if np.isfinite(new_k) else np.nan,
                "new_reason": str(new_reason),
                "theta_at_m_low_deg": float(mark["theta_at_m_low_deg"]) if np.isfinite(mark["theta_at_m_low_deg"]) else np.nan,
                "theta_at_m_high_deg": float(mark["theta_at_m_high_deg"]) if np.isfinite(mark["theta_at_m_high_deg"]) else np.nan,
                "t_at_m_low_s": float(mark["t_at_m_low_s"]) if np.isfinite(mark["t_at_m_low_s"]) else np.nan,
                "t_at_m_high_s": float(mark["t_at_m_high_s"]) if np.isfinite(mark["t_at_m_high_s"]) else np.nan,
                "marker_reason": str(mark["marker_reason"]),
            }
        )

    return rows


def plot_cycle_grid(rows: list[dict], file_name: str, intervention: str, out_png: Path) -> None:
    n = len(rows)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.5 * nrows), squeeze=False)

    for i, rec in enumerate(rows):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]

        tt = rec["t_rel"]
        th = rec["theta_rel"]
        mm = rec["moment_rel"]

        ax.plot(tt, th, color="tab:orange", lw=1.4)
        ax.set_ylabel("theta [deg]", color="tab:orange")
        ax.tick_params(axis="y", labelcolor="tab:orange")
        ax.grid(alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(tt, mm, color="tab:blue", lw=1.0)
        ax2.set_ylabel("moment [Nm]", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        tl = rec["t_at_m_low_s"]
        thl = rec["theta_at_m_low_deg"]
        thh = rec["theta_at_m_high_deg"]
        tht = rec["t_at_m_high_s"]
        if np.isfinite(tl) and np.isfinite(thl):
            ax.scatter([tl], [thl], s=24, marker="o", color="tab:red", zorder=6)
            ax.text(tl, thl, f" θ@{M_LOW:.0f}={thl:.2f}", color="tab:red", fontsize=7,
                ha="left", va="bottom")
        if np.isfinite(tht) and np.isfinite(thh):
            ax.scatter([tht], [thh], s=24, marker="o", color="tab:red", zorder=6)
            ax.text(tht, thh, f" θ@{M_HIGH:.0f}={thh:.2f}", color="tab:red", fontsize=7,
                ha="left", va="bottom")

        # Secant reference region guides in red for quick visual alignment.
        for g in M_GUIDES_NM:
            ax2.axhline(g, color="tab:red", lw=1.0, ls="--", alpha=0.85)

        if np.isfinite(tl):
            ax2.scatter([tl], [M_LOW], s=22, marker="D", color="tab:red", zorder=6)
            ax2.text(tl, M_LOW, f" {M_LOW:.0f}Nm", color="tab:red", fontsize=7,
                     ha="left", va="bottom")
        if np.isfinite(tht):
            ax2.scatter([tht], [M_HIGH], s=22, marker="D", color="tab:red", zorder=6)
            ax2.text(tht, M_HIGH, f" {M_HIGH:.0f}Nm", color="tab:red", fontsize=7,
                     ha="left", va="bottom")

        ax.set_title(
            f"cyc {int(rec['cycle'])} | old dth={rec['old_dtheta_deg']:.3f} | "
            f"new dth={rec['new_dtheta_deg']:.3f}\n{rec['new_reason']}",
            fontsize=8,
        )

        if r == nrows - 1:
            ax.set_xlabel("time in ramp [s]")

    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    fig.suptitle(
        f"{file_name} ({intervention}): theta_rel and moment_rel vs time for each loading cycle\n"
        f"dashed red moment guides at {M_GUIDES_NM[0]:.0f} Nm and {M_GUIDES_NM[1]:.0f} Nm",
        fontsize=13,
    )
    fig.tight_layout()
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
                "M_peak_dyn_Nm": rec["M_peak_dyn_Nm"],
                "theta_span_ramp_deg": rec["theta_span_ramp_deg"],
                "old_dtheta_deg": rec["old_dtheta_deg"],
                "new_dtheta_deg": rec["new_dtheta_deg"],
                "old_k_sec": rec["old_k_sec"],
                "new_k_sec": rec["new_k_sec"],
                "new_reason": rec["new_reason"],
                "theta_at_m_low_deg": rec["theta_at_m_low_deg"],
                "theta_at_m_high_deg": rec["theta_at_m_high_deg"],
                "t_at_m_low_s": rec["t_at_m_low_s"],
                "t_at_m_high_s": rec["t_at_m_high_s"],
                "marker_reason": rec["marker_reason"],
            }
            for rec in rows
        ]
    )
    df.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR_INTACT.mkdir(parents=True, exist_ok=True)

    tracker = ai.tracker_cases().copy()
    tracker["File Name"] = tracker["File Name"].astype(str).str.strip()
    tracker["Intervention"] = tracker["Intervention"].astype(str).str.strip()

    for file_name in ass.CASE_ORDER:
        sub = tracker[tracker["File Name"] == file_name]
        if sub.empty:
            print(f"skip (not found in tracker): {file_name}")
            continue

        intervention = str(sub.iloc[0]["Intervention"])
        print(f"\n=== {file_name} | {intervention} ===")

        rows = build_cycle_rows(file_name, intervention)
        if not rows:
            print("no cycles after filtering")
            continue

        stem = sanitize(file_name)
        out_png = OUT_DIR / f"{stem}_all_cycles_theta_moment_time.png"
        out_csv = OUT_DIR / f"{stem}_all_cycles_theta_moment_time_metrics.csv"
        plot_cycle_grid(rows, file_name, intervention, out_png)
        save_metrics(rows, out_csv)

        # Keep the intact-holding legacy output path updated too.
        if file_name == "2 - Intact Holding Longer":
            intact_png = OUT_DIR_INTACT / "intact_holding_all_cycles_theta_moment_time.png"
            intact_csv = OUT_DIR_INTACT / "intact_holding_all_cycles_theta_moment_time_metrics.csv"
            plot_cycle_grid(rows, file_name, intervention, intact_png)
            save_metrics(rows, intact_csv)


if __name__ == "__main__":
    main()
