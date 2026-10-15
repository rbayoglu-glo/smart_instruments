from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analyze_l4l5_intact_prior_method as A
from lamina_spreader_lr import break_tracking_gaps

OUT = Path(__file__).resolve().parent / "output" / "Step_04_baseline_sampling_highlight.png"


def main() -> None:
    names = list(A.TRIALS)
    results = {n: A.process_trial(n) for n in names}

    fig, axes = plt.subplots(
        len(names), 2, figsize=(16.0, 7.0), squeeze=False,
        gridspec_kw={"width_ratios": (4.0, 1.5)},
    )
    for (ax, zoom_ax), name in zip(axes, names):
        info, _ = results[name]
        fs, fd = info["force_t_sync"], info["force_dynamic"]
        t, thd, th0 = info["theta_t"], info["theta_dynamic"], info["theta0_deg"]

        ws, we = info["force_baseline_window_s"]
        ax.axvspan(ws, we, color="tab:green", alpha=0.25, label="force baseline window")
        in_win = (fs >= ws) & (fs < we)
        h_fb = ax.scatter(fs[in_win], fd[in_win], s=6, color="tab:green", zorder=5, label="force baseline samples")
        h_force, = ax.plot(fs, fd, color="tab:blue", lw=1.0, label="force")
        ax.set_ylabel("force [N]", color="tab:blue")
        ax.set_ylim(-25.0, 150.0)
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        tb, thb = break_tracking_gaps(t, thd)
        h_theta, = ax2.plot(tb, thb, color="tab:orange", lw=0.95, label="theta")
        ax2.axhline(0.0, color="tab:orange", lw=0.9, ls="--", alpha=0.8)
        ax2.set_ylabel("theta [deg]", color="tab:orange")
        ax2.set_ylim(-4.0, 7.0)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        in_th = (t >= ws) & (t < we) & np.isfinite(thd)
        h_used = ax2.scatter(
            t[in_th], thd[in_th], s=38, color="tab:purple", edgecolor="white",
            linewidth=0.6, zorder=9, label="theta0 samples (pre-cycle window)",
        )

        near_window = (t >= ws - 0.5) & (t <= we + 0.5)
        zoom_t, zoom_theta = break_tracking_gaps(t[near_window], thd[near_window])
        zoom_ax.plot(zoom_t, zoom_theta, color="tab:orange", lw=1.0)
        zoom_ax.axvspan(ws, we, color="tab:purple", alpha=0.12)
        zoom_ax.axhline(0.0, color="tab:purple", lw=0.8, ls="--")
        zoom_ax.scatter(
            t[in_th], thd[in_th], s=28, color="tab:purple", edgecolor="white",
            linewidth=0.6, zorder=5, label="theta0 samples (pre-cycle window)",
        )
        zoom_ax.set_xlim(ws - 0.5, we + 0.5)
        zoom_ax.set_title(
            f"theta0: {int(in_th.sum())} highlighted samples\n"
            f"Window [{ws:.2f}, {we:.2f}) s", loc="left", fontsize=9,
        )
        zoom_ax.set_xlabel("time [s]")
        zoom_ax.set_ylabel("baseline-subtracted theta [deg]")
        zoom_ax.grid(alpha=0.25)

        starts = np.asarray([c["i0"] for c in info["candidates"]], dtype=int)
        h_sf = ax.scatter(fs[starts], fd[starts], s=28, marker="o", facecolor="white", edgecolor="tab:red", zorder=6, label="ramp start (force)")
        h_st = ax2.scatter(fs[starts], info["theta_on_force"][starts], s=28, marker="D", facecolor="white", edgecolor="tab:red", zorder=6, label="ramp start (theta)")
        for c in info["candidates"]:
            idx = int(c["i0"])
            ax2.annotate(
                f"C{c['cycle_id']}",
                (fs[idx], info["theta_on_force"][idx]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="tab:red",
                zorder=7,
            )

        ax.set_title(
            f"{name} | force baseline {info['force_baseline_N']:+.2f} N from [{ws:.2f}, {we:.2f}) s "
            f"({int(in_win.sum())} samples)\n"
            f"theta0 {th0:+.3f} deg from {int(in_th.sum())} samples in the same window",
            loc="left", fontsize=9,
        )
        ax.set_xlabel("time [s]")
        if ax is axes[0, 0]:
            hs = [h_fb, h_force, h_theta, h_used, h_sf, h_st]
            ax.legend(hs, [h.get_label() for h in hs], loc="upper left", fontsize=7)

    fig.suptitle(
        "Step 04 baseline sampling: force window (green), theta0 samples (purple)\n"
        "Right panels show the exact theta0 samples at baseline-window scale",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
