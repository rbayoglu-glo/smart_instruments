from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_l4l5_intact_prior_method as analysis
from intact_level_processing_config import OUTPUT_DIR, get_level_config
from lamina_spreader_lr import break_tracking_gaps

LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5")
OUT = OUTPUT_DIR / "Step_04_intraop_Lumbar_baseline_sampling_highlights.png"


def configure(level: str) -> None:
    config = get_level_config(level)
    analysis.FORCE_DIR = config["force_dir"]
    analysis.TRACKING_DIR = config["tracking_dir"]
    analysis.TRIALS = config["trials"]


def main() -> None:
    records = []
    for level in LEVELS:
        configure(level)
        for trial_name in analysis.TRIALS:
            records.append((level, trial_name, analysis.process_trial(trial_name)[0]))

    fig, axes = plt.subplots(
        len(records),
        3,
        figsize=(20.0, 24.0),
        squeeze=False,
        gridspec_kw={"width_ratios": (4.0, 1.5, 1.5)},
    )
    for row, ((level, name, info), (ax, force_zoom_ax, theta_zoom_ax)) in enumerate(zip(records, axes)):
        fs, fd = info["force_t_sync"], info["force_dynamic"]
        t, theta_dynamic = info["theta_t"], info["theta_dynamic"]
        ws, we = info["force_baseline_window_s"]

        ax.axvspan(ws, we, color="tab:green", alpha=0.25, label="force baseline window")
        in_force_window = (fs >= ws) & (fs < we)
        h_force_window = ax.scatter(
            fs[in_force_window],
            fd[in_force_window],
            s=6,
            color="tab:green",
            zorder=5,
            label="force baseline samples",
        )
        h_force, = ax.plot(fs, fd, color="tab:blue", lw=1.0, label="force")
        ax.set_ylabel("force [N]", color="tab:blue")
        ax.set_ylim(-25.0, 150.0)
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        theta_t_brk, theta_brk = break_tracking_gaps(t, theta_dynamic)
        h_theta, = ax2.plot(theta_t_brk, theta_brk, color="tab:orange", lw=0.95, label="theta")
        ax2.axhline(0.0, color="tab:orange", lw=0.9, ls="--", alpha=0.8)
        ax2.set_ylabel("theta [deg]", color="tab:orange")
        ax2.set_ylim(-4.0, 7.0)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        near_force_window = (fs >= ws - 0.5) & (fs <= we + 0.5)
        force_zoom_ax.plot(
            fs[near_force_window],
            fd[near_force_window],
            color="tab:blue",
            lw=1.0,
        )
        force_zoom_ax.axvspan(ws, we, color="tab:green", alpha=0.16)
        force_zoom_ax.axhline(0.0, color="tab:green", lw=0.8, ls="--")
        force_zoom_ax.scatter(
            fs[in_force_window],
            fd[in_force_window],
            s=16,
            color="tab:green",
            edgecolor="white",
            linewidth=0.4,
            zorder=5,
        )
        force_zoom_ax.set_xlim(ws - 0.5, we + 0.5)
        local_force = fd[near_force_window]
        local_force = local_force[np.isfinite(local_force)]
        if local_force.size:
            force_pad = max(0.5, 0.15 * float(np.ptp(local_force)))
            force_zoom_ax.set_ylim(
                float(np.nanmin(local_force)) - force_pad,
                float(np.nanmax(local_force)) + force_pad,
            )
        force_zoom_ax.set_title(
            f"force0: {int(in_force_window.sum())} samples\n"
            f"Window [{ws:.2f}, {we:.2f}) s",
            loc="left",
            fontsize=9,
        )
        force_zoom_ax.set_ylabel("force [N]")
        force_zoom_ax.grid(alpha=0.25)

        in_theta_window = (t >= ws) & (t < we) & np.isfinite(theta_dynamic)
        h_theta_window = ax2.scatter(
            t[in_theta_window],
            theta_dynamic[in_theta_window],
            s=38,
            color="tab:purple",
            edgecolor="white",
            linewidth=0.6,
            zorder=9,
            label="theta0 samples",
        )

        near_window = (t >= ws - 0.5) & (t <= we + 0.5)
        zoom_t, zoom_theta = break_tracking_gaps(t[near_window], theta_dynamic[near_window])
        theta_zoom_ax.plot(zoom_t, zoom_theta, color="tab:orange", lw=1.0)
        theta_zoom_ax.axvspan(ws, we, color="tab:purple", alpha=0.12)
        theta_zoom_ax.axhline(0.0, color="tab:purple", lw=0.8, ls="--")
        theta_zoom_ax.scatter(
            t[in_theta_window],
            theta_dynamic[in_theta_window],
            s=28,
            color="tab:purple",
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        theta_zoom_ax.set_xlim(ws - 0.5, we + 0.5)
        theta_zoom_ax.set_title(
            f"theta0: {int(in_theta_window.sum())} samples\n"
            f"Window [{ws:.2f}, {we:.2f}) s",
            loc="left",
            fontsize=9,
        )
        theta_zoom_ax.set_ylabel("theta [deg]")
        theta_zoom_ax.grid(alpha=0.25)

        starts = np.asarray([candidate["i0"] for candidate in info["candidates"]], dtype=int)
        h_start_force = ax.scatter(
            fs[starts],
            fd[starts],
            s=28,
            marker="o",
            facecolor="white",
            edgecolor="tab:red",
            zorder=6,
            label="ramp start (force)",
        )
        h_start_theta = ax2.scatter(
            fs[starts],
            info["theta_on_force"][starts],
            s=28,
            marker="D",
            facecolor="white",
            edgecolor="tab:red",
            zorder=6,
            label="ramp start (theta)",
        )
        for candidate in info["candidates"]:
            index = int(candidate["i0"])
            ax2.annotate(
                f"C{candidate['cycle_id']}",
                (fs[index], info["theta_on_force"][index]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="tab:red",
                zorder=7,
            )

        ax.set_title(
            f"{level} | {name.rsplit(' ', 1)[-1]} | "
            f"force0 {info['force_baseline_N']:+.2f} N, theta0 {info['theta0_deg']:+.3f} deg | "
            f"window [{ws:.2f}, {we:.2f}) s",
            loc="left",
            fontsize=9,
        )
        ax.set_xlabel("time [s]")
        force_zoom_ax.set_xlabel("time [s]")
        theta_zoom_ax.set_xlabel("time [s]")
        if row == 0:
            handles = [h_force_window, h_force, h_theta, h_theta_window, h_start_force, h_start_theta]
            ax.legend(handles, [handle.get_label() for handle in handles], loc="upper left", fontsize=7)

    fig.suptitle(
        "Step 04 baseline sampling: all lumbar intraoperative levels and sides\n"
        "Full traces are followed by force-baseline and theta0 sample-selection zooms.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
