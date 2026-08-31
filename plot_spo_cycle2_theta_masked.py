from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analyze_interventions as ai
import lamina_spreader as ls
import plot_sync_check_all_interventions_stacked as rawsync
import plot_sync_check_all_interventions_stacked_filtered as filt


OUT_DIR = Path(__file__).parent / "output"
FILE_NAME = "7 - SPO"
THETA_FORCE_MASK_N = 5.0
CYCLE_NUMBER = 2
PAD_SEC = 2.0


def load_case_with_cycles() -> tuple[dict, list[dict]]:
    kin = ls.compute_theta(
        ls.parse_poses(ai.find_robot_folder(FILE_NAME) / "pf" / "poses.txt", ls.SPREADER_UUID)
    )
    force = ls.load_force(ai.find_force_csv(FILE_NAME))
    lag0 = ls.find_lag(force, kin)
    lag, _ = ls.refine_lag(kin, force, lag0)
    d, _, fmax = ls.sync(kin, force, lag)

    t = d["t_sync"].to_numpy(float)
    f = d["force"].to_numpy(float)
    th = d["theta_dist"].to_numpy(float)

    if t.size >= 2:
        dt = float(np.nanmedian(np.diff(t)))
    else:
        dt = np.nan
    fs_hz = float(1.0 / dt) if np.isfinite(dt) and dt > 0 else np.nan

    f_dsp = filt.despike_mad(f, filt.FORCE_DESPIKE_WINDOW, filt.FORCE_DESPIKE_Z)
    f_dsp[~np.isfinite(f)] = np.nan
    f_filt = filt.butter_lowpass_filter(f_dsp, fs_hz, filt.FORCE_LOWPASS_CUTOFF_HZ, filt.BUTTER_ORDER)
    th_filt = filt.filter_theta(th, fs_hz)

    runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
    cands = []
    for s, e in runs:
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0, _, pk = rb
        cands.append(
            {
                "s": int(s),
                "e": int(e),
                "i0": int(i0),
                "pk": int(pk),
                "fpk": float(f[pk]),
                "dur": int(e - s),
            }
        )

    cands = rawsync.dedupe_peak_candidates(cands, t)
    cands = rawsync.filter_pseudoramps(cands, t, th)
    cands = sorted(cands, key=lambda c: c["pk"])

    case = {
        "lag": float(lag),
        "fs_hz": fs_hz,
        "t": t,
        "f": f,
        "th": th,
        "f_filt": f_filt,
        "th_filt": th_filt,
    }
    return case, cands


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    case, cycles = load_case_with_cycles()
    if len(cycles) < CYCLE_NUMBER:
        raise RuntimeError(f"Requested cycle {CYCLE_NUMBER} not found. Available cycles: {len(cycles)}")

    cyc = cycles[CYCLE_NUMBER - 1]
    t = case["t"]
    f = case["f"]
    th = case["th"]
    f_filt = case["f_filt"]
    th_filt = case["th_filt"]

    i_s = int(cyc["s"])
    i_e = int(cyc["e"])

    t_start = float(t[i_s] - PAD_SEC)
    t_end = float(t[max(i_s, i_e - 1)] + PAD_SEC)
    win = (t >= t_start) & (t <= t_end)

    th_raw_masked = th.copy()
    th_filt_masked = th_filt.copy()
    mask_loaded = np.isfinite(f) & (f >= THETA_FORCE_MASK_N)
    th_raw_masked[~mask_loaded] = np.nan
    th_filt_masked[~mask_loaded] = np.nan

    fig, ax = plt.subplots(2, 1, figsize=(12.5, 6.7), sharex=True)

    ax[0].plot(t[win], f[win], color="0.7", lw=1.0, label="force raw")
    ax[0].plot(t[win], f_filt[win], color="tab:blue", lw=1.3, label="force filtered")
    ax[0].axhline(THETA_FORCE_MASK_N, color="tab:red", ls="--", lw=1.0, alpha=0.9, label="theta mask threshold (5 N)")
    ax[0].set_ylabel("force [N]")
    ax[0].set_title(f"SPO cycle {CYCLE_NUMBER}: force window used for theta masking")
    ax[0].grid(alpha=0.25)
    ax[0].legend(fontsize=8, loc="best")

    ax[1].plot(t[win], th[win], color="0.8", lw=1.0, label="theta raw (unmasked)")
    ax[1].plot(t[win], th_filt[win], color="#f4b183", lw=1.0, label="theta filtered (unmasked)")
    ax[1].plot(t[win], th_raw_masked[win], color="0.35", lw=1.2, label="theta raw masked (force>=5N)")
    ax[1].plot(t[win], th_filt_masked[win], color="tab:orange", lw=1.4, label="theta filtered masked (force>=5N)")
    ax[1].set_ylabel("theta [deg]")
    ax[1].set_xlabel("time [s]")
    ax[1].set_title("Theta with force-based masking")
    ax[1].grid(alpha=0.25)
    ax[1].legend(fontsize=8, loc="best")

    fs_txt = f"{case['fs_hz']:.2f} Hz" if np.isfinite(case["fs_hz"]) else "N/A"
    fig.suptitle(
        f"SPO cycle {CYCLE_NUMBER} raw vs filtered with theta masking | lag {case['lag']:+.2f} s | fs {fs_txt}\n"
        f"theta hidden where force < {THETA_FORCE_MASK_N:.1f} N",
        fontsize=12,
    )
    fig.tight_layout()

    out = OUT_DIR / "spo_cycle2_raw_vs_filtered_theta_masked_force_ge5N.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    print(f"saved {out}")
    print(f"cycle_start_s={float(t[i_s]):.3f}")
    print(f"cycle_end_s={float(t[max(i_s, i_e - 1)]):.3f}")


if __name__ == "__main__":
    main()
