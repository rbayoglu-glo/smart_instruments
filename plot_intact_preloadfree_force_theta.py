from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analyze_interventions as ai
import lamina_spreader as ls


OUT_DIR = Path(__file__).parent / "output"
FILE_NAME = "2 - Intact Holding Longer"
LABEL = "intact, but holding"
SHORT_RAMP_SAMPLES = 20


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    kin = ls.compute_theta(
        ls.parse_poses(ai.find_robot_folder(FILE_NAME) / "pf" / "poses.txt", ls.SPREADER_UUID)
    )
    force = ls.load_force(ai.find_force_csv(FILE_NAME))
    lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))
    d, _, fmax = ls.sync(kin, force, lag)

    t = d["t_sync"].to_numpy()
    f = d["force"].to_numpy()
    th = d["theta_dist"].to_numpy()

    f_dyn = np.full_like(f, np.nan, dtype=float)
    th_dyn = np.full_like(th, np.nan, dtype=float)
    t_start = []
    f_start = []
    th_start = []

    runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
    for s, e in runs:
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0, i1, _ = rb
        ramp = np.arange(i0, i1)

        f0 = float(f[i0])
        f_dyn_ramp = f[ramp] - f0

        n_ref = min(SHORT_RAMP_SAMPLES, len(ramp))
        th_ref = float(np.nanmedian(th[ramp[:n_ref]]))
        if not np.isfinite(th_ref):
            th_ref = float(th[i0])

        f_dyn[ramp] = f_dyn_ramp
        th_dyn[ramp] = th[ramp] - th_ref

        t_start.append(float(t[i0]))
        f_start.append(float(f_dyn_ramp[0]))
        th_start.append(float(th_dyn[ramp[0]]))

    fig, ax1 = plt.subplots(figsize=(12.2, 4.0))
    ax2 = ax1.twinx()

    h_force, = ax1.plot(t, f_dyn, color="tab:blue", lw=1.4, label="distraction force (preload-free)")
    h_theta, = ax2.plot(t, th_dyn, color="tab:orange", lw=1.0, alpha=0.9, label="theta (short-ramp-median referenced)")

    h_start_force = ax1.scatter(t_start, f_start, s=28, marker="o", facecolor="white", edgecolor="tab:blue", lw=1.1, zorder=4, label="ramp start (force)")
    h_start_theta = ax2.scatter(t_start, th_start, s=26, marker="D", facecolor="white", edgecolor="tab:orange", lw=1.0, zorder=4, label="ramp start (theta)")

    ax1.axhline(0.0, color="0.6", lw=0.8)
    ax2.axhline(0.0, color="0.7", lw=0.8, ls=":")

    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("distraction force [N]", color="tab:blue")
    ax2.set_ylabel("theta [deg]", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title("Intact Holding: preload-free distraction force and short-ramp-median theta")
    ax1.grid(alpha=0.3)

    handles = [h_force, h_theta, h_start_force, h_start_theta]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper right", fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "intact_preloadfree_force_theta_vs_time.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
