from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_interventions as ai
import lamina_spreader as ls


OUT_DIR = Path(__file__).parent / "output"

CASE_ORDER = [
    "2 - Intact Holding Longer",
    "3 - PUBF Left",
    "4 - FUF Left",
    "5 - FBF",
    "6 - Posterior Release",
    "7 - SPO",
]


def analyze_case(file_name: str, intervention: str) -> tuple[list[dict], dict | None, dict]:
    force_csv = ai.find_force_csv(file_name)

    raw = pd.read_csv(force_csv)
    raw_first200_med = float(pd.to_numeric(raw[ls.FORCE_COL], errors="coerce").iloc[:200].median())

    kin = ls.compute_theta(ls.parse_poses(ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID))
    force = ls.load_force(force_csv)
    lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))
    d, theta0, fmax = ls.sync(kin, force, lag)

    f = d["force"].to_numpy()
    th = d["theta_dist"].to_numpy()
    mom = d["moment"].to_numpy()
    t = d["t_sync"].to_numpy()

    rows = []
    first_curve = None
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

        theta_dist = th[ramp]
        theta_eng = th[ramp] - th[iref]
        moment = mom[ramp]

        rows.append(
            {
                "file_name": file_name,
                "intervention": intervention,
                "cycle": n,
                "F_start_N": float(f[i0]),
                "F_eng_N": float(f_eng),
                "F_peak_N": float(f[pk]),
                "theta_dist_start_deg": float(th[i0]),
                "theta_dist_at_eng_deg": float(th[iref]),
                "theta_eng_start_deg": float(th[i0] - th[iref]),
                "M_start_Nm": float(mom[i0]),
                "M_eng_Nm": float(mom[iref]),
                "lag_s": float(lag),
                "theta0_deg": float(theta0),
                "ramp_start_time_s": float(t[i0]),
                "eng_time_s": float(t[iref]),
            }
        )

        if first_curve is None:
            first_curve = {
                "file_name": file_name,
                "intervention": intervention,
                "cycle": n,
                "theta_dist": theta_dist,
                "theta_eng": theta_eng,
                "moment": moment,
                "theta_dist_at_eng": float(th[iref]),
                "M_eng": float(mom[iref]),
            }

    baseline = {
        "file_name": file_name,
        "intervention": intervention,
        "raw_first200_median": raw_first200_med,
        "corrected_first200_median": float(force["force"].iloc[:200].median()),
    }

    return rows, first_curve, baseline


def plot_preload_boxplot(df: pd.DataFrame, labels: list[str], out: Path) -> None:
    vals = [df.loc[df["intervention"] == lb, "F_start_N"].dropna().to_numpy() for lb in labels]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bp = ax.boxplot(vals, labels=labels, patch_artist=True, showmeans=True)
    for box in bp["boxes"]:
        box.set(facecolor="#8ecae6", alpha=0.8)

    ax.axhline(ls.ENGAGE_FORCE_N, color="tab:red", ls="--", lw=1.2, label=f"engagement threshold ({ls.ENGAGE_FORCE_N:.0f} N)")
    ax.set_ylabel("preload at ramp start F_start [N]")
    ax.set_title("Preload magnitude by intervention (cycle-wise)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def plot_preload_vs_engagement(df: pd.DataFrame, labels: list[str], out: Path) -> None:
    g = (
        df.groupby("intervention", as_index=False)
        .agg(F_start_med=("F_start_N", "median"), F_eng_med=("F_eng_N", "median"), n=("cycle", "count"))
    )
    g = g.set_index("intervention").reindex(labels).reset_index()

    x = np.arange(len(g))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - w / 2, g["F_start_med"], width=w, color="#219ebc", label="median F_start")
    ax.bar(x + w / 2, g["F_eng_med"], width=w, color="#ffb703", label="median F_eng")

    for i, n in enumerate(g["n"].to_numpy(int)):
        ax.text(x[i], max(g["F_start_med"].iloc[i], g["F_eng_med"].iloc[i]) + 0.35,
                f"n={n}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(g["intervention"], rotation=20, ha="right")
    ax.set_ylabel("force [N]")
    ax.set_title("Preload vs engagement force by intervention")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def plot_theta_reference_shift(first_curves: list[dict], out: Path) -> None:
    curves = [c for c in first_curves if c is not None]
    if not curves:
        return

    n = len(curves)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, c in zip(axes, curves):
        ax.plot(c["theta_dist"], c["moment"], lw=1.8, color="tab:blue", label=r"$M$ vs $\theta_{dist}$")
        ax.plot(c["theta_eng"], c["moment"], lw=1.8, color="tab:orange", label=r"$M$ vs $\theta_{eng}$")

        ax.scatter([c["theta_dist_at_eng"]], [c["M_eng"]], color="tab:blue", s=22)
        ax.scatter([0.0], [c["M_eng"]], color="tab:orange", s=22)

        ax.axvline(0.0, color="0.6", ls=":", lw=1)
        ax.axhline(0.0, color="0.6", ls=":", lw=1)
        ax.set_title(f"{c['intervention']} (cycle {c['cycle']})", fontsize=9)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    for ax in axes[-ncols:]:
        ax.set_xlabel("theta [deg]")
    for ax in axes[::ncols]:
        ax.set_ylabel("moment [N.m]")

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Reference shift visualization: theta_dist vs theta_eng (first valid cycle)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def plot_first200_baseline(baseline_df: pd.DataFrame, labels: list[str], out: Path) -> None:
    df = baseline_df.set_index("intervention").reindex(labels).reset_index()

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x, df["raw_first200_median"], color="#90be6d", alpha=0.85)
    ax.axhline(0.0, color="0.5", lw=1)

    ax.set_xticks(x)
    ax.set_xticklabels(df["intervention"], rotation=20, ha="right")
    ax.set_ylabel("raw first-200 median [sensor units]")
    ax.set_title("Force baseline term used in correction (one value per file, not per cycle)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    tracker = ai.tracker_cases()
    tracker = tracker[tracker["File Name"].isin(CASE_ORDER)].copy()

    all_rows = []
    first_curves = []
    baselines = []

    print("Computing preload/theta-reference diagnostics...")
    for _, r in tracker.iterrows():
        file_name = str(r["File Name"]).strip()
        intervention = str(r["Intervention"]).strip()
        print(f"\n=== {file_name} | {intervention} ===")
        rows, first_curve, baseline = analyze_case(file_name, intervention)
        all_rows.extend(rows)
        first_curves.append(first_curve)
        baselines.append(baseline)

    cyc = pd.DataFrame(all_rows)
    base = pd.DataFrame(baselines)

    labels = [str(tracker.loc[tracker["File Name"] == fn, "Intervention"].iloc[0]).strip() for fn in CASE_ORDER if fn in set(tracker["File Name"]) ]

    cyc_path = OUT_DIR / "preload_theta_reference_cycles.csv"
    base_path = OUT_DIR / "preload_theta_reference_baseline.csv"
    cyc.to_csv(cyc_path, index=False)
    base.to_csv(base_path, index=False)

    plot_preload_boxplot(cyc, labels, OUT_DIR / "preload_magnitude_boxplot.png")
    plot_preload_vs_engagement(cyc, labels, OUT_DIR / "preload_vs_engagement.png")
    plot_theta_reference_shift(first_curves, OUT_DIR / "theta_reference_shift_first_cycle.png")
    plot_first200_baseline(base, labels, OUT_DIR / "force_first200_baseline_per_file.png")

    print(f"\nSaved: {cyc_path}")
    print(f"Saved: {base_path}")


if __name__ == "__main__":
    main()
