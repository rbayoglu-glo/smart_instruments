from pathlib import Path
import re

import numpy as np
import pandas as pd

import analyze_intact_linear_region_stiffness as lr
import analyze_spreader_secant_stiffness as ass
import plot_intact_holding_all_cycles_force_tip_displacement as src
import plot_theta_moment_time_all_interventions as base


OUT_DIR = Path(__file__).parent / "output" / "linear_region_stiffness_cycle_grids"
FORCE_YLIM_N = (-25.0, 125.0)
CSV_TRANSITION_FLAGS = Path(__file__).parent / "output" / "sync_check_transition_flags.csv"


def _clean_intervention(name: str) -> str:
    txt = str(name).strip().replace("_", " ")
    txt = re.sub(r"^\d+\s*-\s*", "", txt)
    txt = " ".join(txt.split())
    low = txt.lower()
    if "intact" in low and "holding" in low:
        return "Intact"
    if low == "intact":
        return "Intact"
    if "pubf" in low or low.startswith("puf"):
        return "PUF"
    if low.startswith("fuf"):
        return "FUF"
    return txt


def _display_case_name(file_name: str, intervention: str) -> str:
    m = re.match(r"^(\d+)\s*-\s*", str(file_name).strip())
    if m:
        return f"{m.group(1)} - {_clean_intervention(intervention)}"
    return _clean_intervention(intervention)


def _to_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "yes", "y"])


def load_transition_flag_map() -> dict[tuple[str, int], bool]:
    if not CSV_TRANSITION_FLAGS.exists():
        return {}

    df = pd.read_csv(CSV_TRANSITION_FLAGS)
    req = {"file_name", "cycle", "transition_flag"}
    if df.empty or not req.issubset(df.columns):
        return {}

    tf = _to_bool(df["transition_flag"])
    out: dict[tuple[str, int], bool] = {}
    for r, is_flag in zip(df.itertuples(index=False), tf.to_numpy(bool)):
        key = (str(r.file_name).strip(), int(r.cycle))
        out[key] = bool(is_flag)
    return out


def apply_transition_exclusions(results: list[dict], file_name: str, flag_map: dict[tuple[str, int], bool]) -> list[dict]:
    out = []
    for res in results:
        key = (str(file_name).strip(), int(res.get("cycle", -1)))
        if flag_map.get(key, False):
            rr = dict(res)
            rr["ok"] = False
            rr["reason"] = "transition_flagged_excluded"
            out.append(rr)
        else:
            out.append(res)
    return out


def summarize_case(df: pd.DataFrame, file_name: str, intervention: str) -> dict:
    valid = df[df["valid"]]
    if valid.empty:
        return {
            "file_name": file_name,
            "intervention": intervention,
            "n_cycles": int(len(df)),
            "n_valid": 0,
            "k_mean_n_per_mm": np.nan,
            "k_median_n_per_mm": np.nan,
            "k_std_n_per_mm": np.nan,
            "r2_mean": np.nan,
        }

    return {
        "file_name": file_name,
        "intervention": intervention,
        "n_cycles": int(len(df)),
        "n_valid": int(len(valid)),
        "k_mean_n_per_mm": float(valid["k_linear_n_per_mm"].mean()),
        "k_median_n_per_mm": float(valid["k_linear_n_per_mm"].median()),
        "k_std_n_per_mm": float(valid["k_linear_n_per_mm"].std(ddof=0)),
        "r2_mean": float(valid["r2"].mean()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transition_flags = load_transition_flag_map()

    tracker = base.ai.tracker_cases().copy()
    tracker["File Name"] = tracker["File Name"].astype(str).str.strip()
    tracker["Intervention"] = tracker["Intervention"].astype(str).str.strip()

    summary_rows = []
    cycle_frames = []
    case_payloads = []

    for file_name in ass.CASE_ORDER:
        sub = tracker[tracker["File Name"] == file_name]
        if sub.empty:
            print(f"skip (not found in tracker): {file_name}")
            continue

        intervention = str(sub.iloc[0]["Intervention"])
        print(f"\n=== {file_name} | {intervention} ===")

        cycles = src.build_cycle_rows_force_tip(file_name, intervention)
        if not cycles:
            print("no cycles produced")
            continue

        results = [lr.analyze_cycle(rec) for rec in cycles]
        results = apply_transition_exclusions(results, file_name, transition_flags)

        case_payloads.append(
            {
                "file_name": file_name,
                "intervention": intervention,
                "results": results,
            }
        )

    if not case_payloads:
        print("no cases produced")
        return

    for payload in case_payloads:
        file_name = payload["file_name"]
        intervention = payload["intervention"]
        results = payload["results"]

        case_xmin = np.inf
        case_xmax = -np.inf
        for res in results:
            x_raw = np.asarray(res.get("x_raw", np.array([])), dtype=float)
            if x_raw.size == 0 or not np.any(np.isfinite(x_raw)):
                continue
            case_xmin = min(case_xmin, float(np.nanmin(x_raw)))
            case_xmax = max(case_xmax, float(np.nanmax(x_raw)))

        if np.isfinite(case_xmin) and np.isfinite(case_xmax) and case_xmax > case_xmin:
            x_span = case_xmax - case_xmin
            x_pad = max(1.0, 0.03 * x_span)
            case_xlim_mm = (case_xmin - x_pad, case_xmax + x_pad)
        else:
            case_xlim_mm = None

        case_ylim_n = FORCE_YLIM_N

        stem = base.sanitize(file_name)
        out_png = OUT_DIR / f"{stem}_all_cycles_force_tip_linear_region_stiffness.png"
        out_csv = OUT_DIR / f"{stem}_all_cycles_force_tip_linear_region_stiffness_metrics.csv"

        lr.plot_cycle_grid(
            results,
            _display_case_name(file_name, intervention),
            _clean_intervention(intervention),
            out_png,
            xlim_mm=case_xlim_mm,
            ylim_n=case_ylim_n,
        )
        lr.save_metrics(results, out_csv)

        df_case = pd.read_csv(out_csv)
        df_case.insert(0, "intervention", intervention)
        df_case.insert(0, "file_name", file_name)
        cycle_frames.append(df_case)

        s = summarize_case(df_case, file_name, intervention)
        summary_rows.append(s)
        if s["n_valid"] > 0:
            print(
                f"valid cycles: {s['n_valid']}/{s['n_cycles']} | "
                f"k mean={s['k_mean_n_per_mm']:.3f} N/mm, "
                f"std={s['k_std_n_per_mm']:.3f} N/mm"
            )
        else:
            print(f"valid cycles: 0/{s['n_cycles']}")

        n_non_neutral = int((df_case["reason"] == "non_neutral_start_excluded").sum())
        if n_non_neutral > 0:
            print(f"{file_name} excluded non-neutral-start cycles: {n_non_neutral}")

        n_transition = int((df_case["reason"] == "transition_flagged_excluded").sum())
        if n_transition > 0:
            print(f"{file_name} excluded transition-flagged cycles: {n_transition}")

        print(
            f"{file_name} force y-limits [N]: "
            f"{case_ylim_n[0]:.2f} to {case_ylim_n[1]:.2f}"
        )
        if case_xlim_mm is not None:
            print(
                f"{file_name} displacement x-limits [mm]: "
                f"{case_xlim_mm[0]:.2f} to {case_xlim_mm[1]:.2f}"
            )

    if cycle_frames:
        cyc_all = pd.concat(cycle_frames, ignore_index=True)
        out_all_cycles = OUT_DIR / "all_interventions_linear_region_stiffness_cycles.csv"
        cyc_all.to_csv(out_all_cycles, index=False)
        print(f"saved {out_all_cycles}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary["rank_k_median"] = summary["k_median_n_per_mm"].rank(ascending=False, method="dense")
        summary = summary.sort_values("rank_k_median").reset_index(drop=True)

        out_summary = OUT_DIR / "all_interventions_linear_region_stiffness_summary.csv"
        summary.to_csv(out_summary, index=False)
        print(f"saved {out_summary}")

        print("\nLinear-region stiffness ranking (higher = stiffer):")
        print(
            summary[
                [
                    "file_name",
                    "intervention",
                    "n_valid",
                    "n_cycles",
                    "k_mean_n_per_mm",
                    "k_median_n_per_mm",
                    "k_std_n_per_mm",
                    "r2_mean",
                    "rank_k_median",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
