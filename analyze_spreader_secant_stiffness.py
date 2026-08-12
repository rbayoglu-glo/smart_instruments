#%%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_interventions as ai
import lamina_spreader as ls


OUT_DIR = Path(__file__).parent / "output"

# Positive-branch secant window for spreader loading ramps.
SECANT_M_LOW_NM = 2.0
SECANT_M_HIGH_NM = 5.0

# Baseline Intact excluded, Intact Holding retained.
CASE_ORDER = [
    "2 - Intact Holding Longer",
    "3 - PUBF Left",
    "4 - FUF Left",
    "5 - FBF",
    "6 - Posterior Release",
    "7 - SPO",
]


def secant_on_rising_branch(theta_rel_deg: np.ndarray, moment_nm: np.ndarray,
                            m_low: float, m_high: float) -> tuple[float, float, float]:
    """Compute secant stiffness k = dM/dtheta on rising branch only.

    Returns k_sec [Nm/deg], theta_low [deg], theta_high [deg].
    """
    if theta_rel_deg.size < 2 or moment_nm.size < 2:
        return np.nan, np.nan, np.nan

    # Keep the rising hull so M(theta) is single-valued on the loading segment.
    hull = moment_nm >= np.maximum.accumulate(moment_nm) - 1e-9
    m = moment_nm[hull]
    a = theta_rel_deg[hull]
    if m.size < 2:
        return np.nan, np.nan, np.nan

    order = np.argsort(m)
    m = m[order]
    a = a[order]

    m_u, first = np.unique(m, return_index=True)
    a_u = a[first]
    if m_u.size < 2:
        return np.nan, np.nan, np.nan

    if m_low < float(np.min(m_u)) or m_high > float(np.max(m_u)):
        return np.nan, np.nan, np.nan

    th_low = float(np.interp(m_low, m_u, a_u))
    th_high = float(np.interp(m_high, m_u, a_u))
    dtheta = th_high - th_low
    if not np.isfinite(dtheta) or dtheta <= 1e-10:
        return np.nan, th_low, th_high

    k_sec = float((m_high - m_low) / dtheta)
    return k_sec, th_low, th_high


def cycle_secant_table(file_name: str, intervention: str) -> pd.DataFrame:
    kin = ls.compute_theta(ls.parse_poses(ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID))
    force = ls.load_force(ai.find_force_csv(file_name))
    lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))
    d, theta0, fmax = ls.sync(kin, force, lag)

    f = d["force"].to_numpy()
    th = d["theta_dist"].to_numpy()
    mom = d["moment"].to_numpy()

    rows = []
    curves = []
    runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
    for n, (s, e) in enumerate(runs, 1):
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0, i1, pk = rb
        ramp = np.arange(i0, i1)

        f_eng = min(ls.ENGAGE_FORCE_N, 0.60 * f[pk])
        hit = ramp[f[ramp] >= f_eng]
        if len(hit) == 0:
            continue
        iref = int(hit[0])

        theta_rel = th[ramp] - th[iref]
        m_rel = mom[ramp]
        k_sec, th_lo, th_hi = secant_on_rising_branch(
            theta_rel, m_rel, SECANT_M_LOW_NM, SECANT_M_HIGH_NM
        )

        rows.append(
            {
                "file_name": file_name,
                "intervention": intervention,
                "cycle": n,
                "lag_s": float(lag),
                "theta0_deg": float(theta0),
                "F_peak_N": float(f[pk]),
                "M_peak_Nm": float(mom[pk]),
                "F_eng_N": float(f_eng),
                "M_low_Nm": float(SECANT_M_LOW_NM),
                "M_high_Nm": float(SECANT_M_HIGH_NM),
                "theta_low_deg": float(th_lo) if np.isfinite(th_lo) else np.nan,
                "theta_high_deg": float(th_hi) if np.isfinite(th_hi) else np.nan,
                "k_sec_Nm_per_deg": float(k_sec) if np.isfinite(k_sec) else np.nan,
                "valid_secant": bool(np.isfinite(k_sec)),
            }
        )

        curves.append(
            {
                "cycle": n,
                "theta_rel_deg": theta_rel,
                "moment_nm": m_rel,
                "theta_low_deg": float(th_lo) if np.isfinite(th_lo) else np.nan,
                "theta_high_deg": float(th_hi) if np.isfinite(th_hi) else np.nan,
                "k_sec_Nm_per_deg": float(k_sec) if np.isfinite(k_sec) else np.nan,
            }
        )

    plot_moment_theta_cycles(file_name, intervention, curves)

    return pd.DataFrame(rows)


def rank_corr(summary: pd.DataFrame) -> float:
    """Correlation between ranking order and surgical file order."""
    r = summary.dropna(subset=["k_sec_med_Nm_per_deg"]).sort_values(
        "k_sec_med_Nm_per_deg", ascending=False
    )
    names = [x for x in r["file_name"] if x in CASE_ORDER]
    if len(names) <= 2:
        return np.nan
    x = [CASE_ORDER.index(n) for n in names]
    y = list(range(len(names)))
    return float(np.corrcoef(x, y)[0, 1])


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def plot_moment_theta_cycles(file_name: str, intervention: str, curves: list[dict]) -> None:
    """Save one figure per intervention with one subplot per cycle (M vs theta)."""
    if not curves:
        return

    n = len(curves)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.1 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, c in zip(axes, curves):
        theta = c["theta_rel_deg"]
        moment = c["moment_nm"]
        ax.plot(theta, moment, lw=1.6, color="tab:blue")

        if np.isfinite(c["theta_low_deg"]) and np.isfinite(c["theta_high_deg"]):
            ax.plot(
                [c["theta_low_deg"], c["theta_high_deg"]],
                [SECANT_M_LOW_NM, SECANT_M_HIGH_NM],
                color="tab:orange",
                lw=2.0,
            )
            ax.scatter(
                [c["theta_low_deg"], c["theta_high_deg"]],
                [SECANT_M_LOW_NM, SECANT_M_HIGH_NM],
                color="tab:orange",
                s=18,
                zorder=3,
            )

        title = (
            f"cycle {c['cycle']}  |  k_sec={c['k_sec_Nm_per_deg']:.2f}"
            if np.isfinite(c["k_sec_Nm_per_deg"])
            else f"cycle {c['cycle']}  |  k_sec=N/A"
        )
        ax.set_title(title, fontsize=9)
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.axvline(0.0, color="0.7", lw=0.8)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    for ax in axes[-ncols:]:
        ax.set_xlabel(r"$\theta_{eng}$ [deg]")
    for ax in axes[::ncols]:
        ax.set_ylabel(r"pivot moment $M$ [N$\cdot$m]")

    fig.suptitle(
        f"{file_name} ({intervention}) - moment vs theta by cycle\n"
        f"theta referenced to engagement, secant window [{SECANT_M_LOW_NM:.1f}, {SECANT_M_HIGH_NM:.1f}] Nm"
    )
    fig.tight_layout()
    out = OUT_DIR / f"spreader_moment_theta_cycles_{sanitize(file_name)}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def plot_summary(summary: pd.DataFrame, out: Path) -> None:
    df = summary.sort_values("k_sec_med_Nm_per_deg", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    y = df["k_sec_med_Nm_per_deg"].to_numpy(float)
    ylo = y - df["k_sec_q25_Nm_per_deg"].to_numpy(float)
    yhi = df["k_sec_q75_Nm_per_deg"].to_numpy(float) - y

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bars = ax.bar(x, y, color="tab:blue", alpha=0.85)
    ax.errorbar(x, y, yerr=[ylo, yhi], fmt="none", ecolor="black", capsize=4, lw=1)

    for i, (b, n_valid, n_total) in enumerate(
        zip(bars, df["n_valid"].to_numpy(int), df["n_cycles"].to_numpy(int))
    ):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{n_valid}/{n_total}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["intervention"].to_list(), rotation=20, ha="right")
    ax.set_ylabel("secant stiffness k_sec [Nm/deg]")
    ax.set_title(
        f"Spreader secant stiffness ranking (engagement-referenced)\n"
        f"M in [{SECANT_M_LOW_NM:.1f}, {SECANT_M_HIGH_NM:.1f}] Nm; labels = valid/total cycles"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    tracker = ai.tracker_cases()
    tracker = tracker[tracker["File Name"].isin(CASE_ORDER)].copy()

    frames = []
    print("Computing spreader secant stiffness per cycle...")
    for _, r in tracker.iterrows():
        file_name = str(r["File Name"]).strip()
        intervention = str(r["Intervention"]).strip()
        print(f"\n=== {file_name} | {intervention} ===")
        frames.append(cycle_secant_table(file_name, intervention))

    cyc = pd.concat(frames, ignore_index=True)

    summary = (
        cyc.groupby(["file_name", "intervention"], as_index=False)
        .agg(
            n_cycles=("cycle", "count"),
            n_valid=("valid_secant", "sum"),
            k_sec_med_Nm_per_deg=("k_sec_Nm_per_deg", "median"),
            k_sec_mean_Nm_per_deg=("k_sec_Nm_per_deg", "mean"),
            k_sec_q25_Nm_per_deg=("k_sec_Nm_per_deg", lambda s: float(np.nanpercentile(s, 25))),
            k_sec_q75_Nm_per_deg=("k_sec_Nm_per_deg", lambda s: float(np.nanpercentile(s, 75))),
            M_peak_med_Nm=("M_peak_Nm", "median"),
            F_peak_max_N=("F_peak_N", "max"),
        )
    )
    summary["rank_k_sec"] = summary["k_sec_med_Nm_per_deg"].rank(ascending=False, method="dense")
    summary = summary.sort_values("rank_k_sec").reset_index(drop=True)

    rho = rank_corr(summary)
    print("\n" + "=" * 72)
    print("Secant stiffness ranking (higher k_sec = stiffer)")
    print(f"Window: [{SECANT_M_LOW_NM:.1f}, {SECANT_M_HIGH_NM:.1f}] Nm")
    print(f"Rank correlation vs surgical file order: {rho:+.2f}")
    print("=" * 72)
    print(
        summary[
            [
                "file_name",
                "intervention",
                "n_valid",
                "n_cycles",
                "k_sec_med_Nm_per_deg",
                "k_sec_q25_Nm_per_deg",
                "k_sec_q75_Nm_per_deg",
                "rank_k_sec",
            ]
        ].to_string(index=False)
    )

    cyc_path = OUT_DIR / "spreader_secant_cycles.csv"
    sum_path = OUT_DIR / "spreader_secant_summary.csv"
    fig_path = OUT_DIR / "spreader_secant_ranking.png"

    cyc.to_csv(cyc_path, index=False)
    summary.to_csv(sum_path, index=False)
    plot_summary(summary, fig_path)

    print(f"\nSaved: {cyc_path}")
    print(f"Saved: {sum_path}")


if __name__ == "__main__":
    main()
