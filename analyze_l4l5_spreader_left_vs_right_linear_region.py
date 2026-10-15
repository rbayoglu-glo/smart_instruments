"""L4-L5 lamina spreader, segment data: left vs right linear-region stiffness.

Scope: motion-segment spreader sessions only (intraop data excluded). Sessions,
level, side, intervention and computer start time come from the
"L4-5, Single FSU" sheet of displacement_and_force_correspondence.xlsx.

Method: the global linear-region pipeline is reused unchanged
(spreader_cycle_pipeline.load_case -> build_cycle_rows_force_tip ->
analyze_intact_linear_region_stiffness.analyze_cycle). Only the file lookups are
redirected from the 8-20 tracker layout to the segment folders.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_interventions as ai
import analyze_intact_linear_region_stiffness as lr
import analyze_linear_region_stiffness_all_interventions as alr
import lamina_spreader as ls
import plot_intact_holding_all_cycles_force_tip_displacement as src

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
SEG_ROOT = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\UTN24_26016\segment\spreader")
META_XLSX = SEG_ROOT / "displacement_and_force_correspondence.xlsx"
META_SHEET = "L4-5, Single FSU"
FORCE_DIR = SEG_ROOT / "Force" / "lamspreader"
DISP_DIR = SEG_ROOT / "Displacement" / "gns-sensorhub"
MATCH_TOL_S = 300
OUT_DIR = Path(__file__).parent / "output" / "l4l5_spreader_left_vs_right_linear_region"

# maps the pipeline's file_name key to the segment force CSV and tracking folder
FILE_MAP: dict[str, dict] = {}


def _patched_find_force_csv(file_name: str) -> Path:
    return FILE_MAP[file_name]["force"]


def _patched_find_robot_folder(file_name: str) -> Path:
    return FILE_MAP[file_name]["folder"]


# redirect the resolvers used by spreader_cycle_pipeline.load_case
ai.find_force_csv = _patched_find_force_csv
ai.find_robot_folder = _patched_find_robot_folder


# sensor-glitch rule (global, applied identically to every session):
# a sample is a glitch if it deviates from the centred rolling median (WIN samples,
# 0.11 s at 100 Hz) by more than DEV_LIMIT_N. Clean segment sessions deviate <= 5 N;
# glitch bursts deviate by tens to hundreds of N.
GLITCH_WIN = 11
DEV_LIMIT_N = 20.0
ABS_LIMIT_N = 200.0  # gross-glitch bound applied before the local-deviation pass
MAX_GLITCH_PASSES = 10


def _repair_force(t: np.ndarray, raw: np.ndarray, base: float) -> tuple[np.ndarray, int, int]:
    """Return force samples (baseline removed, same length as t) with glitch/overflow samples
    replaced by linear interpolation in time. Also returns (n_missing, n_glitch)."""
    valid = ~np.isnan(raw)
    within_abs = np.abs(raw - base) <= ABS_LIMIT_N
    keep = valid & within_abs
    for _ in range(MAX_GLITCH_PASSES):
        filled = np.interp(t, t[keep], raw[keep]) - base
        med = pd.Series(filled).rolling(GLITCH_WIN, center=True, min_periods=3).median().to_numpy()
        new_keep = valid & within_abs & (np.abs(raw - base - med) <= DEV_LIMIT_N)
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
    n_missing = int((~valid).sum())
    n_glitch = int(valid.sum() - keep.sum())
    return np.interp(t, t[keep], raw[keep]) - base, n_missing, n_glitch


def _load_force_segment(path: Path) -> pd.DataFrame:
    """Same logic as lamina_spreader.load_force, with segment-CSV fixes applied globally:
    - values use thousands separators (e.g. "-1,314.09897"): thousands=','
    - saturated samples are written as the text 'Overflow' -> treated as missing
    - sensor glitch bursts -> removed by _repair_force (see DEV_LIMIT_N)
    Removed samples are linearly interpolated in time so the time base stays intact.
    """
    df = pd.read_csv(path, thousands=",")
    t = df["Time"].to_numpy() / 1e6
    raw = pd.to_numeric(df[ls.FORCE_COL], errors="coerce").to_numpy(dtype=float)
    base = float(np.nanmedian(raw[:200]))
    x, n_missing, n_glitch = _repair_force(t, raw, base)
    df["t_s"] = t
    df["force"] = ls.FORCE_SIGN * x
    print(f"  repaired {n_missing} Overflow/missing + {n_glitch} glitch samples "
          f"(interpolated) in {path.name}")
    print(f"force: {len(df)} samples, {df['t_s'].iloc[-1]:.1f} s, "
          f"peak {df['force'].max():.2f} N")
    return df[["t_s", "force"]]


# pipeline calls ls.load_force at call time; override for segment CSVs only
ls.load_force = _load_force_segment


# --------------------------------------------------------------------------- #
# metadata and session matching
# --------------------------------------------------------------------------- #
def read_metadata() -> tuple[str, pd.DataFrame]:
    raw = pd.read_excel(META_XLSX, sheet_name=META_SHEET, header=None)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", str(raw.iloc[1, 0]))
    if m is None:
        raise ValueError("computer date not found in metadata sheet")
    comp_date = m.group(1)

    rows = []
    for _, r in raw.iloc[3:].iterrows():
        if pd.isna(r.iloc[2]) or pd.isna(r.iloc[6]):
            continue
        session = int(r.iloc[6])
        rows.append(
            {
                "session": session,
                "level": str(r.iloc[1]).strip(),
                "side": str(r.iloc[2]).strip(),
                "intervention": " ".join(str(r.iloc[3]).split()),
                "computer_time": str(r.iloc[5]).strip(),
                "key": f"L4-5 Session {session}",
            }
        )
    return comp_date, pd.DataFrame(rows)


def find_tracking_folder(comp_date: str, comp_time: str) -> tuple[int, Path]:
    """Nearest tracking folder (with pf/poses.txt) to the computer start time."""
    hh, mm, ss = (int(x) for x in comp_time.split(":"))
    target = hh * 3600 + mm * 60 + ss

    cands = []
    for p in DISP_DIR.rglob(f"{comp_date}_*"):
        m = re.fullmatch(rf"{comp_date}_\[(\d\d)-(\d\d)-(\d\d)\]", p.name)
        if m is None or not p.is_dir() or not (p / "pf" / "poses.txt").exists():
            continue
        sec = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        cands.append((abs(sec - target), p))

    if not cands:
        raise FileNotFoundError(f"no tracking folder on {comp_date}")
    cands.sort(key=lambda c: c[0])
    if cands[0][0] > MATCH_TOL_S:
        raise RuntimeError(f"no tracking folder within {MATCH_TOL_S} s of {comp_time}")
    if len(cands) > 1 and cands[1][0] == cands[0][0]:
        raise RuntimeError(f"ambiguous tracking folder for {comp_time}")
    return cands[0]


def empty_summary(key: str, intervention: str) -> dict:
    return {
        "file_name": key,
        "intervention": intervention,
        "n_cycles": 0,
        "n_valid": 0,
        "k_mean_n_per_mm": np.nan,
        "k_median_n_per_mm": np.nan,
        "k_std_n_per_mm": np.nan,
        "r2_mean": np.nan,
    }


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_left_vs_right(cyc: pd.DataFrame, pairs: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(pairs) + 3.0), 5.2))
    x = np.arange(len(pairs))
    valid = cyc[cyc["valid"].astype(bool)]
    rng = np.random.default_rng(0)

    for side, off, color in (("Left", -0.13, "tab:blue"), ("Right", 0.13, "tab:orange")):
        for i, row in enumerate(pairs.itertuples(index=False)):
            sess = row.left_session if side == "Left" else row.right_session
            k = valid.loc[valid["session"] == sess, "k_linear_n_per_mm"].to_numpy(dtype=float)
            if k.size:
                jit = rng.uniform(-0.04, 0.04, size=k.size)
                ax.scatter(x[i] + off + jit, k, s=18, color=color, alpha=0.45, zorder=2)
            med = row.left_k_median if side == "Left" else row.right_k_median
            if np.isfinite(med):
                ax.scatter([x[i] + off], [med], s=70, marker="D", color=color,
                           edgecolor="k", zorder=3,
                           label=side if i == 0 else None)
                ax.text(x[i] + off, med, f" {med:.2f}", fontsize=8, va="center", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(pairs["intervention"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("linear-region stiffness [N/mm]")
    ax.set_title("L4-L5 spreader (motion segment): left vs right, per cycle and median")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    comp_date, meta = read_metadata()
    print(f"computer date: {comp_date} | sessions: {len(meta)}")

    used: dict[Path, int] = {}
    force_cols = set()
    for _, r in meta.iterrows():
        force = FORCE_DIR / f"Session #{r['session']}.csv"
        if not force.exists():
            raise FileNotFoundError(force)
        with force.open() as fh:
            hdr = [c.strip() for c in fh.readline().split(",")]
        spreader_cols = [c for c in hdr if c.startswith("Spreader (")]
        if len(spreader_cols) != 1:
            raise RuntimeError(f"{force.name}: expected one spreader column, found {spreader_cols}")
        force_cols.add(spreader_cols[0])
        dt, folder = find_tracking_folder(comp_date, r["computer_time"])
        if folder in used:
            raise RuntimeError(f"folder {folder} matched sessions {used[folder]} and {r['session']}")
        used[folder] = int(r["session"])
        FILE_MAP[r["key"]] = {"force": force, "folder": folder}
        print(f"session {r['session']:>3} {r['side']:<5} {r['computer_time']} "
              f"dt={dt:>3d}s -> {folder.relative_to(SEG_ROOT)}")

    if len(force_cols) != 1:
        raise RuntimeError(f"force column differs across sessions: {sorted(force_cols)}")
    ls.FORCE_COL = force_cols.pop()
    print(f"force column: {ls.FORCE_COL}")

    cycle_frames = []
    summary_rows = []
    for _, r in meta.iterrows():
        sess, side, key, interv = int(r["session"]), r["side"], r["key"], r["intervention"]
        print(f"\n=== session {sess} | {side} | {interv} ===")

        rows = src.build_cycle_rows_force_tip(key, interv)
        if not rows:
            print("no cycles produced")
            summary_rows.append({**empty_summary(key, interv), "session": sess, "side": side})
            continue

        results = [lr.analyze_cycle(rec) for rec in rows]

        png = OUT_DIR / f"session_{sess}_{side}_cycles.png"
        csv = OUT_DIR / f"session_{sess}_{side}_cycle_metrics.csv"
        lr.plot_cycle_grid(results, f"Session {sess} | {side}", interv, png)
        lr.save_metrics(results, csv)

        df = pd.read_csv(csv)
        df.insert(0, "side", side)
        df.insert(0, "intervention", interv)
        df.insert(0, "session", sess)
        cycle_frames.append(df)

        s = alr.summarize_case(df, key, interv)
        s.update({"session": sess, "side": side})
        summary_rows.append(s)
        print(f"valid cycles: {s['n_valid']}/{s['n_cycles']} | "
              f"k median={s['k_median_n_per_mm']:.3f} N/mm, std={s['k_std_n_per_mm']:.3f}")

    cyc = pd.concat(cycle_frames, ignore_index=True)
    cyc.to_csv(OUT_DIR / "all_sessions_cycle_metrics.csv", index=False)

    summ = pd.DataFrame(summary_rows)
    summ = summ[["session", "side", "intervention", "n_cycles", "n_valid",
                 "k_mean_n_per_mm", "k_median_n_per_mm", "k_std_n_per_mm", "r2_mean"]]
    summ.to_csv(OUT_DIR / "session_summary.csv", index=False)

    # Left/Right pairs: the sheet lists each intervention as consecutive Left, Right rows
    pair_rows = []
    m_sorted = meta.reset_index(drop=True)
    for i in range(0, len(m_sorted) - 1, 2):
        left, right = m_sorted.iloc[i], m_sorted.iloc[i + 1]
        if left["side"] != "Left" or right["side"] != "Right":
            raise RuntimeError(f"unexpected side order at sessions {left['session']}/{right['session']}")
        sl = summ[summ["session"] == left["session"]].iloc[0]
        sr = summ[summ["session"] == right["session"]].iloc[0]
        pair_rows.append(
            {
                "intervention": left["intervention"],
                "left_session": int(left["session"]),
                "right_session": int(right["session"]),
                "left_n_valid": int(sl["n_valid"]),
                "left_k_median": sl["k_median_n_per_mm"],
                "left_k_mean": sl["k_mean_n_per_mm"],
                "left_k_std": sl["k_std_n_per_mm"],
                "right_n_valid": int(sr["n_valid"]),
                "right_k_median": sr["k_median_n_per_mm"],
                "right_k_mean": sr["k_mean_n_per_mm"],
                "right_k_std": sr["k_std_n_per_mm"],
                "right_over_left_median": sr["k_median_n_per_mm"] / sl["k_median_n_per_mm"]
                if np.isfinite(sl["k_median_n_per_mm"]) and sl["k_median_n_per_mm"] != 0 else np.nan,
            }
        )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT_DIR / "left_vs_right_by_intervention.csv", index=False)
    plot_left_vs_right(cyc, pairs, OUT_DIR / "left_vs_right_linear_region_stiffness.png")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nLeft vs right by intervention (N/mm):")
    print(pairs.round(3).to_string(index=False))
    print(f"\nsaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
