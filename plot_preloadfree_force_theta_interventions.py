from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_interventions as ai
import lamina_spreader as ls


OUT_DIR = Path(__file__).parent / "output"
SHORT_RAMP_SAMPLES = 20
GRID_POINTS = 180
MOMENT_MARKERS_NM = (2.0, 5.0)

CASE_ORDER = [
    "2 - Intact Holding Longer",
    "3 - PUBF Left",
    "4 - FUF Left",
    "5 - FBF",
    "6 - Posterior Release",
    "7 - SPO",
]

STYLE = {
    "intact, but holding": {"color": "#000000", "marker": "o"},
    "PUBF, left": {"color": "#0072B2", "marker": "s"},
    "FUF, left": {"color": "#E69F00", "marker": "^"},
    "FBF": {"color": "#009E73", "marker": "D"},
    "Posterior Release": {"color": "#D55E00", "marker": "v"},
    "SPO": {"color": "#CC79A7", "marker": "P"},
}


def rising_unique_curve(force_dyn: np.ndarray, theta_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hull = force_dyn >= np.maximum.accumulate(force_dyn) - 1e-9
    f = force_dyn[hull]
    th = theta_rel[hull]
    if f.size < 2:
        return np.array([]), np.array([])

    order = np.argsort(f)
    f = f[order]
    th = th[order]

    f_u, first = np.unique(f, return_index=True)
    th_u = th[first]
    good = np.isfinite(f_u) & np.isfinite(th_u)
    f_u = f_u[good]
    th_u = th_u[good]

    if f_u.size < 2:
        return np.array([]), np.array([])
    return f_u, th_u


def cycle_curves_for_case(file_name: str, intervention: str) -> list[dict]:
    kin = ls.compute_theta(
        ls.parse_poses(ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID)
    )
    force = ls.load_force(ai.find_force_csv(file_name))
    lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))
    d, _, fmax = ls.sync(kin, force, lag)

    f = d["force"].to_numpy()
    th = d["theta_dist"].to_numpy()

    curves = []
    runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
    for n, (s, e) in enumerate(runs, 1):
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0, i1, pk = rb
        ramp = np.arange(i0, i1)
        if ramp.size < 3:
            continue

        f0 = float(f[i0])
        f_dyn = f[ramp] - f0

        n_ref = min(SHORT_RAMP_SAMPLES, ramp.size)
        th_ref = float(np.nanmedian(th[ramp[:n_ref]]))
        if not np.isfinite(th_ref):
            th_ref = float(th[i0])

        th_rel = th[ramp] - th_ref
        f_u, th_u = rising_unique_curve(f_dyn, th_rel)
        if f_u.size < 2:
            continue

        curves.append(
            {
                "file_name": file_name,
                "intervention": intervention,
                "cycle": n,
                "f_start_dyn_N": float(f_dyn[0]),
                "theta_start_rel_deg": float(th_rel[0]),
                "f_peak_dyn_N": float(np.nanmax(f_dyn)),
                "f_u": f_u,
                "th_u": th_u,
            }
        )

    return curves


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    tracker = ai.tracker_cases()
    tracker = tracker[tracker["File Name"].isin(CASE_ORDER)].copy()

    curves_all = []
    print("Computing preload-free force-theta curves with short-ramp theta median reference...")
    for _, r in tracker.iterrows():
        file_name = str(r["File Name"]).strip()
        intervention = str(r["Intervention"]).strip()
        print(f"\n=== {file_name} | {intervention} ===")
        curves_all.extend(cycle_curves_for_case(file_name, intervention))

    if not curves_all:
        raise RuntimeError("No valid cycles found.")

    peaks = np.array([c["f_peak_dyn_N"] for c in curves_all], dtype=float)
    grid_max = float(np.nanpercentile(peaks, 90))
    if not np.isfinite(grid_max) or grid_max <= 1e-6:
        grid_max = float(np.nanmax(peaks))
    grid = np.linspace(0.0, grid_max, GRID_POINTS)

    rows = []
    fig, ax = plt.subplots(figsize=(9.8, 6.0))

    interventions = list(tracker["Intervention"].astype(str).str.strip())
    for label in interventions:
        cur = [c for c in curves_all if c["intervention"] == label]
        if not cur:
            continue

        mat = np.full((len(cur), grid.size), np.nan, dtype=float)
        for i, c in enumerate(cur):
            f_u = c["f_u"]
            th_u = c["th_u"]
            upto = grid <= float(np.nanmax(f_u))
            if np.count_nonzero(upto) < 2:
                continue
            mat[i, upto] = np.interp(grid[upto], f_u, th_u)

            rows.append(
                {
                    "intervention": label,
                    "cycle": c["cycle"],
                    "f_start_dyn_N": c["f_start_dyn_N"],
                    "theta_start_rel_deg": c["theta_start_rel_deg"],
                    "f_peak_dyn_N": c["f_peak_dyn_N"],
                }
            )

        n_valid = np.sum(np.isfinite(mat), axis=0)
        keep = n_valid >= 2
        med = np.full(grid.size, np.nan, dtype=float)
        q25 = np.full(grid.size, np.nan, dtype=float)
        q75 = np.full(grid.size, np.nan, dtype=float)
        if np.any(keep):
            med[keep] = np.nanmedian(mat[:, keep], axis=0)
            q25[keep] = np.nanpercentile(mat[:, keep], 25, axis=0)
            q75[keep] = np.nanpercentile(mat[:, keep], 75, axis=0)

        st = STYLE.get(label, {"color": "#444444", "marker": "o"})
        ax.plot(grid[keep], med[keep], color=st["color"], lw=2.2, label=label)
        ax.fill_between(grid[keep], q25[keep], q75[keep], color=st["color"], alpha=0.12)

        f0_med = float(np.nanmedian([c["f_start_dyn_N"] for c in cur]))
        th0_med = float(np.nanmedian([c["theta_start_rel_deg"] for c in cur]))
        ax.scatter(
            [f0_med],
            [th0_med],
            s=58,
            marker=st["marker"],
            facecolor="white",
            edgecolor=st["color"],
            linewidth=1.1,
            zorder=4,
        )

    ax.axhline(0.0, color="0.6", lw=0.9)
    ax.axvline(0.0, color="0.6", lw=0.9)

    # Moment-force conversion uses ls.D_O (effective contact-point distance).
    for m_nm in MOMENT_MARKERS_NM:
        f_eq_n = float(ls.moment_to_force_n(m_nm))
        ax.axvline(f_eq_n, color="0.35", lw=1.0, ls="--", alpha=0.9)
        ax.text(
            f_eq_n,
            0.985,
            f"{m_nm:.0f} Nm ({f_eq_n:.1f} N)",
            rotation=90,
            ha="right",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
            color="0.25",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        )

    ax.set_xlabel("distraction force without preload [N]")
    ax.set_ylabel("theta relative to short-ramp median [deg]")
    ax.set_title(
        "All interventions: preload-free force vs theta\n"
        f"theta reference = median of first {SHORT_RAMP_SAMPLES} ramp samples"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_png = OUT_DIR / "spreader_preloadfree_force_theta_all_interventions.png"
    fig.savefig(out_png, dpi=160)
    print(f"saved {out_png}")

    out_csv = OUT_DIR / "spreader_preloadfree_force_theta_rampstart_points.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
