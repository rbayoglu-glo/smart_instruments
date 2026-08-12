#%%
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_interventions as ai
import lamina_spreader as ls


BASE = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab")
TRACKER_XLSX = BASE / "Force_Data" / "Intervention Tracker.xlsx"
OUT_DIR = Path(__file__).parent / "output"


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def intervention_cases() -> pd.DataFrame:
    """Exclude baseline Intact only; keep Intact Holding rows."""
    df = pd.read_excel(TRACKER_XLSX, sheet_name="8-20")
    df = df[df["File Name"].notna()].copy()
    df = df[df["File Name"].str.match(r"^[0-9]+\s-\s")].copy()

    # Remove only the baseline intact case, but keep intact-holding trials.
    inter_norm = df["Intervention"].fillna("").str.strip().str.lower()
    file_norm = df["File Name"].fillna("").str.strip().str.lower()
    is_baseline_intact = inter_norm.eq("intact") | file_norm.str.match(r"^1\s*-\s*intact$")
    df = df[~is_baseline_intact].copy()

    return df[["Session", "Intervention", "File Name"]].sort_values("Session")


def plot_sync_for_case(file_name: str, intervention: str) -> dict:
    force_csv = ai.find_force_csv(file_name)
    poses_txt = ai.find_robot_folder(file_name) / "pf" / "poses.txt"

    kin = ls.compute_theta(ls.parse_poses(poses_txt, ls.SPREADER_UUID))
    force = ls.load_force(force_csv)
    lag0 = ls.find_lag(force, kin)
    lag, scores = ls.refine_lag(kin, force, lag0)

    d, _, fmax = ls.sync(kin, force, lag)
    runs = ls.segment_cycles(d["force"].to_numpy(), ls.CYCLE_THRESH * fmax)

    # zoom around first loading ramp for quick quality inspection
    if runs:
        s0, e0 = runs[0]
        t0 = float(d["t_sync"].iloc[s0])
        t1 = float(d["t_sync"].iloc[min(e0, len(d) - 1)])
        lo, hi = max(0.0, t0 - 3.0), min(float(force["t_s"].max()), t1 + 6.0)
    else:
        lo, hi = 0.0, min(40.0, float(force["t_s"].max()))

    fig, bx = plt.subplots(3, 1, figsize=(14, 10))
    for a, (x0, x1) in zip(bx[:2], [(0.0, float(force["t_s"].max())), (lo, hi)]):
        a.plot(force["t_s"], force["force"], lw=0.8, color="tab:blue")
        a.set(xlabel="time [s]", ylabel="distraction force [N]", xlim=(x0, x1))
        a2 = a.twinx()
        a2.plot(d["t_sync"], d["theta_dist"], lw=0.9, color="tab:orange")
        a2.set_ylabel(r"$\theta$ [deg]", color="tab:orange")
        a.grid(alpha=0.3)

    bx[0].set_title(f"{file_name} ({intervention})  -  sync check (lag {lag:+.2f} s)")
    bx[2].plot(scores[:, 0], scores[:, 1], lw=1)
    bx[2].axvline(lag, color="tab:red", ls="--")
    bx[2].set(xlabel="lag [s]", ylabel="mean ramp $R^2$", title="lag sensitivity")
    bx[2].grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / f"sync_check_{sanitize(file_name)}_{sanitize(intervention)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")

    return {
        "file_name": file_name,
        "intervention": intervention,
        "lag_s": float(lag),
        "n_cycles": int(len(runs)),
        "plot": out.name,
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    cases = intervention_cases()

    rows = []
    print("Generating sync-check plots for interventions (intact rows excluded)...")
    for _, r in cases.iterrows():
        file_name = str(r["File Name"]).strip()
        intervention = str(r["Intervention"]).strip()
        print(f"\n=== {file_name} | {intervention} ===")
        rows.append(plot_sync_for_case(file_name, intervention))

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / "sync_check_interventions_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nDone.")
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
