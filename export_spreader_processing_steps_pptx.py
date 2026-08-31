from datetime import date
from pathlib import Path
import re

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt


BASE = Path(__file__).parent
OUT = BASE / "output"
PPTX_PATH = OUT / f"spreader_processing_steps_{date.today().isoformat()}.pptx"

FIG_STEP1_RAW = OUT / "sync_check_all_interventions_stacked.png"
FIG_STEP2_RAMPSTART = OUT / "sync_check_all_interventions_stacked_filtered_rampstart.png"
FIG_STEP3_PRELOADFREE = OUT / "sync_check_all_interventions_stacked_filtered_rampstart_preloadfree_theta0corr.png"

DIR_LINEAR = OUT / "linear_region_stiffness_cycle_grids"
DIR_MOMENT = OUT / "moment_theta_cycle_grids"

LINEAR_SUFFIX = "_all_cycles_force_tip_linear_region_stiffness.png"
MOMENT_SUFFIX = "_all_cycles_moment_theta.png"

FIG_LINEAR_RESULTS = OUT / "linear_region_stiffness_cycle_grids" / "all_interventions_linear_region_stiffness_mean_std_bar_by_order.png"

FIG_MOMENT_RESULTS = OUT / "moment_theta_cycle_grids" / "all_interventions_moment_theta_stiffness_mean_std_bar_by_order.png"

CSV_FLAGS = OUT / "sync_check_transition_flags.csv"
CSV_LINEAR_SUMMARY = OUT / "linear_region_stiffness_cycle_grids" / "all_interventions_linear_region_stiffness_summary.csv"
CSV_MOMENT_SUMMARY = OUT / "moment_theta_cycle_grids" / "all_interventions_moment_theta_stiffness_summary.csv"

TITLE_FONT = Pt(32)
SUBTITLE_FONT = Pt(18)
BODY_FONT = Pt(19)
MONO_FONT = Pt(12)


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


def _to_bool_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "yes", "y"])


def set_title(shape, text: str, size: Pt = TITLE_FONT) -> None:
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.bold = True
    run.font.size = size


def add_bullets(slide, lines: list[str]) -> None:
    box = slide.shapes.add_textbox(Inches(0.8), Inches(1.45), Inches(11.9), Inches(5.7))
    tf = box.text_frame
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = BODY_FONT


def add_mono_block(slide, title: str, lines: list[str]) -> None:
    set_title(slide.shapes.title, title, size=Pt(26))
    box = slide.shapes.add_textbox(Inches(0.65), Inches(0.95), Inches(12.1), Inches(5.95))
    tf = box.text_frame
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = "Consolas"
        p.font.size = MONO_FONT


def add_image_slide(prs: Presentation, title: str, image_path: Path, caption: str | None = None) -> None:
    if not image_path.exists():
        return

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(slide.shapes.title, title, size=Pt(26))
    slide.shapes.add_picture(str(image_path), Inches(0.5), Inches(0.9), width=Inches(12.3))

    if caption:
        cap = slide.shapes.add_textbox(Inches(0.6), Inches(6.9), Inches(12.1), Inches(0.4))
        tf = cap.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.text = caption
        p.font.size = Pt(11)


def _case_sort_key(image_path: Path) -> tuple[int, str]:
    stem = image_path.stem
    head = stem.split("___", 1)[0].strip()
    try:
        n = int(head)
    except ValueError:
        n = 999
    return n, stem


def _case_label_from_file_name(file_name: str, suffix: str) -> str:
    base = file_name
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    base = base.replace("___", " - ").replace("_", " ")
    return _clean_intervention(base)


def add_case_chart_slides(
    prs: Presentation,
    title_prefix: str,
    chart_dir: Path,
    suffix: str,
    caption_prefix: str,
) -> None:
    if not chart_dir.exists():
        return

    chart_paths = sorted(chart_dir.glob(f"*{suffix}"), key=_case_sort_key)
    if not chart_paths:
        return

    for chart_path in chart_paths:
        case_label = _case_label_from_file_name(chart_path.name, suffix)
        add_image_slide(
            prs,
            f"{title_prefix}: {case_label}",
            chart_path,
            f"{caption_prefix} {case_label}.",
        )


def add_cycle_eligibility_slide(prs: Presentation) -> None:
    if not CSV_FLAGS.exists():
        return

    df = pd.read_csv(CSV_FLAGS)
    if df.empty or "transition_flag" not in df.columns or "intervention" not in df.columns:
        return

    df["intervention_clean"] = df["intervention"].map(_clean_intervention)
    ineligible = _to_bool_series(df["transition_flag"]) if "transition_flag" in df.columns else pd.Series(False, index=df.index)

    grp = (
        df.assign(ineligible=ineligible)
        .groupby("intervention_clean", as_index=False)
        .agg(
            n_cycles=("cycle", "count"),
            n_ineligible=("ineligible", "sum"),
        )
        .sort_values("intervention_clean")
        .reset_index(drop=True)
    )
    grp["n_eligible"] = grp["n_cycles"] - grp["n_ineligible"]
    total_ineligible = int(grp["n_ineligible"].sum())

    fit_excl = pd.DataFrame()
    if CSV_LINEAR_SUMMARY.exists() and CSV_MOMENT_SUMMARY.exists():
        ldf = pd.read_csv(CSV_LINEAR_SUMMARY)
        mdf = pd.read_csv(CSV_MOMENT_SUMMARY)
        l_ok = {"intervention", "n_cycles", "n_valid"}.issubset(ldf.columns)
        m_ok = {"intervention", "n_cycles", "n_valid"}.issubset(mdf.columns)
        if l_ok and m_ok:
            ldf = ldf.copy()
            mdf = mdf.copy()
            ldf["intervention_clean"] = ldf["intervention"].map(_clean_intervention)
            mdf["intervention_clean"] = mdf["intervention"].map(_clean_intervention)
            ldf["force_fit_excluded"] = (ldf["n_cycles"] - ldf["n_valid"]).astype(int)
            mdf["moment_fit_excluded"] = (mdf["n_cycles"] - mdf["n_valid"]).astype(int)
            fit_excl = ldf[["intervention_clean", "force_fit_excluded"]].merge(
                mdf[["intervention_clean", "moment_fit_excluded"]],
                on="intervention_clean",
                how="outer",
            ).fillna(0)
            fit_excl["force_fit_excluded"] = fit_excl["force_fit_excluded"].astype(int)
            fit_excl["moment_fit_excluded"] = fit_excl["moment_fit_excluded"].astype(int)
            fit_excl = fit_excl.sort_values("intervention_clean").reset_index(drop=True)

    lines = [
        "Step 4: Cycle Eligibility (Transition-Flag Filtering)",
        "",
        "Intervention              n_cycles   eligible   ineligible",
        "---------------------------------------------------------",
    ]
    for _, r in grp.iterrows():
        lines.append(
            f"{str(r['intervention_clean']):<24} {int(r['n_cycles']):>8}   {int(r['n_eligible']):>8}   {int(r['n_ineligible']):>10}"
        )

    lines += [
        "",
        "Rule: transition-flagged cycles are excluded from stiffness fitting.",
        "Note: ineligible here only counts transition_flag=True cycles.",
    ]

    if total_ineligible == 0:
        lines.append("This run has no transition-flagged cycles, so ineligible=0 for all interventions.")

    if not fit_excl.empty:
        lines += [
            "",
            "Additional exclusions can occur later at fit stage (separate from transition flags):",
            "Intervention              force-fit excluded   moment-fit excluded",
            "--------------------------------------------------------------",
        ]
        for _, r in fit_excl.iterrows():
            lines.append(
                f"{str(r['intervention_clean']):<24} {int(r['force_fit_excluded']):>18}   {int(r['moment_fit_excluded']):>19}"
            )

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    add_mono_block(slide, "Step 4: Mark And Exclude Ineligible Cycles", lines)


def add_linear_results_table_slide(prs: Presentation) -> None:
    if not CSV_LINEAR_SUMMARY.exists():
        return

    df = pd.read_csv(CSV_LINEAR_SUMMARY)
    if df.empty:
        return

    df["intervention_clean"] = df["intervention"].map(_clean_intervention)
    df = df.sort_values("rank_k_median").reset_index(drop=True)

    lines = [
        "Step 6A Results: Linear-Region Force-Tip Stiffness",
        "",
        "Rank  Intervention            median [N/mm]    mean [N/mm]   n_valid/n_cycles",
        "--------------------------------------------------------------------------",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{int(r['rank_k_median']):>4}  {str(r['intervention_clean']):<22} {float(r['k_median_n_per_mm']):>11.3f}    {float(r['k_mean_n_per_mm']):>11.3f}      {int(r['n_valid'])}/{int(r['n_cycles'])}"
        )

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    add_mono_block(slide, "Linear Stiffness Ranking", lines)


def add_moment_results_table_slide(prs: Presentation) -> None:
    if not CSV_MOMENT_SUMMARY.exists():
        return

    df = pd.read_csv(CSV_MOMENT_SUMMARY)
    if df.empty:
        return

    df["intervention_clean"] = df["intervention"].map(_clean_intervention)
    df = df.sort_values("k_median_nm_per_deg", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    lines = [
        "Step 6B Results: Moment-Theta Rotational Stiffness",
        "",
        "Rank  Intervention            median [Nm/deg]   mean [Nm/deg]  n_valid/n_cycles",
        "---------------------------------------------------------------------------",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{int(r['rank']):>4}  {str(r['intervention_clean']):<22} {float(r['k_median_nm_per_deg']):>13.3f}   {float(r['k_mean_nm_per_deg']):>12.3f}     {int(r['n_valid'])}/{int(r['n_cycles'])}"
        )

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    add_mono_block(slide, "Rotational Stiffness Ranking", lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Title
    s = prs.slides.add_slide(prs.slide_layouts[0])
    set_title(s.shapes.title, "Spreader Data Processing And Results")
    subtitle = s.placeholders[1].text_frame
    subtitle.clear()
    p = subtitle.paragraphs[0]
    p.text = f"Updated {date.today().isoformat()}"
    p.font.size = SUBTITLE_FONT

    # Workflow summary
    s = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(s.shapes.title, "Workflow Overview", size=Pt(26))
    add_bullets(
        s,
        [
            "1. Sync-check raw force and raw theta traces across interventions.",
            "2. Apply force/theta filters only (no theta0 extraction yet).",
            "3. Apply preload removal + theta0 baseline extraction + ramp-start detection with theta snap.",
            "4. Mark transition-flagged cycles as ineligible.",
            "5. Fit force-tip linear region using displacement >= 0 mm points (some cycles can fail fit criteria).",
            "6. Fit moment-theta linear region using theta >= 0 deg points (some cycles can fail fit criteria).",
        ],
    )

    add_image_slide(
        prs,
        "Step 1: Raw Force-Theta Sync Check",
        FIG_STEP1_RAW,
        "All interventions stacked before filtering/corrections.",
    )

    add_image_slide(
        prs,
        "Step 2: Filtering Only (No Theta0 Yet)",
        FIG_STEP2_RAMPSTART,
        "Filtered force/theta signals only; no theta0 extraction or preload removal in this step.",
    )

    add_image_slide(
        prs,
        "Step 3: Preload Removal + Theta0 + Ramp-Start Snap",
        FIG_STEP3_PRELOADFREE,
        "Signals after preload removal, theta0 baseline extraction, and ramp-start detection with theta snap.",
    )

    add_cycle_eligibility_slide(prs)

    add_case_chart_slides(
        prs,
        "Step 5A: Force-Tip Linear Region Fitting",
        DIR_LINEAR,
        LINEAR_SUFFIX,
        "Cycle-wise linear windows and per-cycle stiffness (fit uses displacement >= 0 mm; shared force y-limits) for",
    )

    add_image_slide(
        prs,
        "Step 5B: Linear Stiffness Results Across Interventions",
        FIG_LINEAR_RESULTS,
        "Mean ± SD from valid cycles (force-tip linear-region stiffness).",
    )

    add_linear_results_table_slide(prs)

    add_case_chart_slides(
        prs,
        "Step 6A: Moment-Theta Processing",
        DIR_MOMENT,
        MOMENT_SUFFIX,
        "Cycle-wise rotational stiffness processing (fit uses theta >= 0 deg; shared theta limits) for",
    )

    add_image_slide(
        prs,
        "Step 6B: Rotational Stiffness Results Across Interventions",
        FIG_MOMENT_RESULTS,
        "Mean ± SD from valid cycles (moment-theta stiffness).",
    )

    add_moment_results_table_slide(prs)

    s = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(s.shapes.title, "Appended Update: Fit-Point Rules", size=Pt(26))
    add_bullets(
        s,
        [
            "This version appends re-analysis results with explicit fit-point thresholds.",
            "Displacement stiffness fitting: use only displacement >= 0 mm points.",
            "Rotational stiffness fitting: use only theta >= 0 deg points.",
            "All rankings/tables in this deck reflect these updated fit-selection rules.",
        ],
    )

    s = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(s.shapes.title, "Appendix: Filters Used", size=Pt(26))
    add_bullets(
        s,
        [
            "Median filter (theta, kernel=5): replaces each sample with local median; removes isolated spikes while preserving step/ramp edges better than mean smoothing.",
            "Butterworth low-pass (order=2): force cutoff=1.5 Hz, theta cutoff=1.2 Hz; attenuates high-frequency noise while preserving low-frequency cycle shape and phase reasonably well.",
            "MAD despiking before low-pass: force window=11, z=3.5; theta window=15, z=3.5; robust outlier suppression using rolling median absolute deviation.",
            "Pseudo-ramp rejection: keep cycles only if run_samples>=30, rise_time>=1.2 s, and theta_rise>=0.2 deg.",
            "Duplicate-peak merge: close peaks are clustered and only strongest candidate retained (time-gap bounds 2.5 to 10.0 s).",
            "Theta0 calibration filter: estimate zero-angle from low-force samples (<5% Fmax) but exclude low-force blocks with |median theta|>1.0 deg.",
            "Ramp-start theta snap: move force-based ramp start to nearest theta~0 in early ramp window (1.5 s), unless start is already non-neutral (|theta|>1.5 deg).",
            "Linear-fit point constraints: displacement fits use x>=0 mm; moment-theta fits use theta>=0 deg; flat low-slope segments removed before regression.",
        ],
    )

    s = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(s.shapes.title, "Key Takeaways", size=Pt(26))
    add_bullets(
        s,
        [
            "Step-wise processing narrows from raw traces to quality-controlled cycles.",
            "Results are reported in two complementary stiffness spaces:",
            "force-tip (N/mm) and moment-theta (Nm/deg).",
            "Rankings are based on median stiffness across valid cycles.",
            "Fit-point constraints are displacement >= 0 mm and theta >= 0 deg.",
        ],
    )

    prs.save(PPTX_PATH)
    print(f"saved {PPTX_PATH}")


if __name__ == "__main__":
    main()
