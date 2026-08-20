#%%
"""Lamina spreader analog of the MTS theta-at-target-moment ranking.

The MTS ranking works because it reads the angle reached at a FIXED moment on the
loading branch, which compares every specimen state at the same load instead of
at whatever peak the operator happened to reach.  The spreader equivalent is the
same idea with the pivot moment M = F * d_o:

    higher angle at the same moment  ->  more compliant  ->  less stiff

Two angle references are reported, because the spreader has an artefact the MTS
rig does not:
  theta_ramp : measured from the foot of the loading ramp (F ~ 0). Direct analog
               of the MTS neutral position, but it also contains tip seating.
  theta_eng  : measured from tip engagement (F = ENGAGE_FORCE_N), which removes
               the seating offset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_interventions as ai
import lamina_spreader as ls

OUT_DIR = Path(__file__).parent / "output"
TEST_LABEL = "L231147 L5-S1"

# moment levels to probe, N.m about the spreader pivot
TARGET_MOMENTS_NM = (2.0, 3.0, 4.0, 5.0)
# level used for the headline ranking
HEADLINE_MOMENT_NM = 5.0

# order the resections were performed, for checking monotonicity
SURGICAL_ORDER = ["Intact Holding", "PUBF Left", "FUF Left", "FBF",
                  "Posterior Release", "SPO"]

CASES = [
    ("2 - Intact Holding Longer", "Intact Holding"),
    ("3 - PUBF Left", "PUBF Left"),
    ("4 - FUF Left", "FUF Left"),
    ("5 - FBF", "FBF"),
    ("6 - Posterior Release", "Posterior Release"),
    ("7 - SPO", "SPO"),
]

# Colorblind-safe, high-contrast intervention styles for visual comparison.
INTERVENTION_STYLE = {
    "Intact Holding": {"color": "#000000", "ls": "-", "marker": "o"},
    "PUBF Left": {"color": "#0072B2", "ls": "-", "marker": "s"},
    "FUF Left": {"color": "#E69F00", "ls": "-", "marker": "^"},
    "FBF": {"color": "#009E73", "ls": "-", "marker": "D"},
    "Posterior Release": {"color": "#D55E00", "ls": "-", "marker": "v"},
    "SPO": {"color": "#CC79A7", "ls": "-", "marker": "P"},
}


def angle_at_moment(theta: np.ndarray, moment: np.ndarray, target: float) -> float:
    """Angle where the rising branch first reaches `target` moment."""
    if theta.size < 2 or target > np.nanmax(moment) or target < np.nanmin(moment):
        return np.nan
    # keep the rising hull so the branch is single valued in moment
    hull = moment >= np.maximum.accumulate(moment) - 1e-9
    m, a = moment[hull], theta[hull]
    order = np.argsort(m)
    m, a = m[order], a[order]
    m_u, first = np.unique(m, return_index=True)
    if m_u.size < 2:
        return np.nan
    return float(np.interp(target, m_u, a[first]))


def cycle_table(file_name: str, label: str) -> pd.DataFrame:
    kin = ls.compute_theta(ls.parse_poses(
        ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID))
    force = ls.load_force(ai.find_force_csv(file_name))
    lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))
    d, _, fmax = ls.sync(kin, force, lag)

    f = d["force"].to_numpy()
    th = d["theta_dist"].to_numpy()
    mom = d["moment"].to_numpy()

    rows = []
    for n, (s, e) in enumerate(ls.segment_cycles(f, ls.CYCLE_THRESH * fmax), 1):
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0, i1, pk = rb
        ramp = np.arange(i0, i1)

        f_eng = min(ls.ENGAGE_FORCE_N, 0.60 * f[pk])
        hit = ramp[f[ramp] >= f_eng]
        if len(hit) == 0:
            continue

        row = {
            "cycle": n,
            "M_peak_Nm": float(mom[pk]),
            "F_peak_N": float(f[pk]),
            "F_eng_N": float(f_eng),
        }
        for tgt in TARGET_MOMENTS_NM:
            row[f"th_ramp@{tgt}"] = angle_at_moment(
                th[ramp] - th[i0], mom[ramp], tgt)
            row[f"th_eng@{tgt}"] = angle_at_moment(
                th[ramp] - th[int(hit[0])], mom[ramp], tgt)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.insert(0, "intervention", label)
    out.insert(0, "file_name", file_name)
    return out


def median_curve(cyc: pd.DataFrame, grid: np.ndarray, ref: str) -> np.ndarray:
    """Median loading curve on a common moment grid, for plotting."""
    cols = [c for c in cyc.columns if c.startswith(f"th_{ref}@")]
    levels = np.array([float(c.split("@")[1]) for c in cols])
    order = np.argsort(levels)
    vals = cyc[[cols[i] for i in order]].to_numpy()
    med = np.nanmedian(vals, axis=0)
    return np.interp(grid, levels[order], med, left=np.nan, right=np.nan)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    frames = []
    for file_name, label in CASES:
        print(f"\n=== {file_name} | {label} ===")
        frames.append(cycle_table(file_name, label))
    cyc = pd.concat(frames, ignore_index=True)
    cyc.to_csv(OUT_DIR / "spreader_theta_at_moment_cycles.csv", index=False)

    agg = {"M_peak_Nm": "median", "F_peak_N": "max", "F_eng_N": "median", "cycle": "count"}
    for tgt in TARGET_MOMENTS_NM:
        agg[f"th_ramp@{tgt}"] = "median"
        agg[f"th_eng@{tgt}"] = "median"
    summ = cyc.groupby(["file_name", "intervention"], as_index=False).agg(agg)
    summ = summ.rename(columns={"cycle": "n_cycles"}).sort_values("file_name")

    # how many cycles actually reached each level
    for tgt in TARGET_MOMENTS_NM:
        reach = (cyc.assign(ok=cyc[f"th_ramp@{tgt}"].notna())
                 .groupby("file_name")["ok"].sum().rename(f"n_reach@{tgt}"))
        summ = summ.merge(reach, on="file_name", how="left")
    summ.to_csv(OUT_DIR / "spreader_theta_at_moment_summary.csv", index=False)

    for ref, name in [("ramp", "from ramp start (F~0)"),
                      ("eng", f"from tip engagement ({ls.ENGAGE_FORCE_N:.0f} N)")]:
        print("\n" + "=" * 72)
        print(f"Angle at target pivot moment, theta measured {name}")
        print("(Lower angle = stiffer)")
        print("=" * 72)
        for tgt in TARGET_MOMENTS_NM:
            col = f"th_{ref}@{tgt}"
            r = summ[["intervention", col, f"n_reach@{tgt}", "n_cycles"]].dropna(subset=[col])
            r = r.sort_values(col).reset_index(drop=True)
            order = [x for x in r["intervention"] if x in SURGICAL_ORDER]
            rho = np.nan
            if len(order) > 2:
                rho = np.corrcoef([SURGICAL_ORDER.index(x) for x in order],
                                  list(range(len(order))))[0, 1]
            print(f"\nM = {tgt:.1f} Nm  (F = {ls.moment_to_force_n(tgt):.0f} N)   "
                  f"rank corr vs surgical order: {rho:+.2f}")
            for i, row in r.iterrows():
                print(f"  {i + 1:>2}. {row['intervention']:<18} "
                      f"theta = {row[col]:.3f} deg   "
                      f"[{int(row[f'n_reach@{tgt}'])}/{int(row['n_cycles'])} cycles]")

    print("\n" + "=" * 72)
    print("Peak reached per intervention")
    print("=" * 72)
    print(summ[["intervention", "n_cycles", "F_peak_N", "M_peak_Nm"]]
          .sort_values("F_peak_N", ascending=False).to_string(index=False))

    # median loading curves, tip-engagement reference only
    m_eng_nominal = ls.force_to_moment_nm(ls.ENGAGE_FORCE_N)
    grid = np.linspace(min(m_eng_nominal, min(TARGET_MOMENTS_NM)),
                       max(TARGET_MOMENTS_NM), 80)
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.8))
    for _, label in CASES:
        sub = cyc[cyc["intervention"] == label]
        if sub.empty:
            continue
        st = INTERVENTION_STYLE.get(label, {"color": "#444444", "ls": "-", "marker": "o"})
        curve = median_curve(sub, grid, "eng")
        ax.plot(
            grid,
            curve,
            lw=2.4,
            color=st["color"],
            ls=st["ls"],
            label=label,
        )
        # Emphasize the measured target moments to help direct comparisons.
        cols = [f"th_eng@{t}" for t in TARGET_MOMENTS_NM]
        pts = sub[cols].median().to_numpy(float)
        ax.plot(
            np.array(TARGET_MOMENTS_NM),
            pts,
            linestyle="None",
            marker=st["marker"],
            ms=5.8,
            color=st["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        f_eng_med = float(summ.loc[summ["intervention"] == label, "F_eng_N"].iloc[0])
        m_eng = ls.force_to_moment_nm(f_eng_med)
        # Engagement reference point: theta_eng is exactly zero at M_eng.
        ax.plot(m_eng, 0.0, marker=st["marker"], ms=5.5, color=st["color"],
                mfc="white", mew=1.0)
    ax.axvline(HEADLINE_MOMENT_NM, color="0.5", ls="--", lw=1)
    ax.axvline(m_eng_nominal, color="0.6", ls=":", lw=1)
    ax.set(
        xlabel=r"pivot moment $M = F\,d_o$ [N$\cdot$m]",
        ylabel=r"distraction angle $\theta$ [deg]",
        title=f"theta from tip engagement ({ls.ENGAGE_FORCE_N:.0f} N)",
    )
    ax.text(m_eng_nominal + 0.03, ax.get_ylim()[0] + 0.03 * np.ptp(ax.get_ylim()),
            r"$M_{eng}$", color="0.4", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Lamina spreader median loading curves - {TEST_LABEL}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spreader_theta_at_moment.png", dpi=150)
    print(f"\nsaved {OUT_DIR / 'spreader_theta_at_moment.png'}")


if __name__ == "__main__":
    main()
