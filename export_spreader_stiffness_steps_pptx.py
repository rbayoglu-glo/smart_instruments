from datetime import date
from pathlib import Path
import re

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt


BASE = Path(__file__).parent
OUT = BASE / "output"
PPTX_PATH = OUT / f"spreader_stiffness_steps_slides_{date.today().isoformat()}.pptx"

FIG_RAW_FORCE_TIME = OUT / "sync_check_all_interventions_stacked.png"
FIG_PRELOADFREE_TIME = OUT / "sync_check_all_interventions_stacked_preloadfree.png"
FIG_RAMP_STARTS = OUT / "spreader_ramp_start_force_across_cycles_interventions.png"
FIG_FORCE_DISP = OUT / "cycles_force_vs_tip.png"
FIG_MOMENT_THETA = OUT / "cycles_moment_vs_theta.png"

CSV_FLAGS = OUT / "sync_check_transition_flags.csv"
CSV_SUMMARY_EXCL = OUT / "spreader_secant_summary_exclude_transition.csv"

TITLE_FONT = Pt(32)
SUBTITLE_FONT = Pt(18)
BODY_FONT = Pt(20)
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
    box = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.8), Inches(5.6))
    tf = box.text_frame
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = BODY_FONT


def add_image_slide(prs: Presentation, title: str, image_path: Path, caption: str | None = None) -> None:
    if not image_path.exists():
        return

    s = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(s.shapes.title, title, size=Pt(26))
    s.shapes.add_picture(str(image_path), Inches(0.5), Inches(0.9), width=Inches(12.3))

    if caption:
        cap = s.shapes.add_textbox(Inches(0.6), Inches(6.95), Inches(12.1), Inches(0.35))
        tf = cap.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.text = caption
        p.font.size = Pt(11)


def add_ineligible_summary_slide(prs: Presentation) -> None:
    if not CSV_FLAGS.exists():
        return

    df = pd.read_csv(CSV_FLAGS)
    if df.empty or ("transition_flag" not in df.columns):
        return

    df["intervention_clean"] = df["intervention"].map(_clean_intervention)
    ineligible = _to_bool_series(df["transition_flag"])

    grp = (
        df.assign(ineligible=ineligible).groupby("intervention_clean", as_index=False)
        .agg(
            n_cycles=("cycle", "count"),
            n_ineligible=("ineligible", "sum"),
        )
        .sort_values("intervention_clean")
        .reset_index(drop=True)
    )
    grp["n_eligible"] = grp["n_cycles"] - grp["n_ineligible"]
    total_ineligible = int(grp["n_ineligible"].sum())

    lines = [
        "Cycle eligibility summary (preload-free transition flags)",
        "",
        "Intervention                n_cycles   eligible   ineligible",
        "------------------------------------------------------------",
    ]
    for _, r in grp.iterrows():
        lines.append(
            f"{str(r['intervention_clean']):<26} {int(r['n_cycles']):>8}   {int(r['n_eligible']):>8}   {int(r['n_ineligible']):>10}"
        )

    lines += [
        "",
        "Rule: ineligible here only means transition_flag=True.",
    ]
    if total_ineligible == 0:
        lines.append("This run has no transition-flagged cycles, so ineligible=0 for all interventions.")

    s = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(s.shapes.title, "Step 4: Mark Ineligible Cycles", size=Pt(26))

    box = s.shapes.add_textbox(Inches(0.7), Inches(1.0), Inches(12.1), Inches(5.8))
    tf = box.text_frame
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = "Consolas"
        p.font.size = MONO_FONT


def add_spreader_ranking_slide(prs: Presentation) -> None:
    if not CSV_SUMMARY_EXCL.exists():
        return

    df = pd.read_csv(CSV_SUMMARY_EXCL)
    if df.empty:
        return

    df["intervention_clean"] = df["intervention"].map(_clean_intervention)
    df = df.sort_values("rank_k_sec").reset_index(drop=True)

    lines = [
        "Spreader stiffness ranking (after excluding ineligible cycles)",
        "",
        "Rank  Intervention                median k_sec [Nm/deg]   n_valid/n_cycles",
        "--------------------------------------------------------------------------",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{int(r['rank_k_sec']):>4}  {str(r['intervention_clean']):<24} {float(r['k_sec_med_Nm_per_deg']):>10.3f}            {int(r['n_valid'])}/{int(r['n_cycles'])}"
        )

    s = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(s.shapes.title, "Final Spreader Stiffness Ranking", size=Pt(26))

    box = s.shapes.add_textbox(Inches(0.7), Inches(1.0), Inches(12.1), Inches(5.8))
    tf = box.text_frame
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = "Consolas"
        p.font.size = MONO_FONT


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # 1) Title
    s = prs.slides.add_slide(prs.slide_layouts[0])
    set_title(s.shapes.title, "Spreader Stiffness: Step-by-Step Workflow")
    subtitle = s.placeholders[1].text_frame
    subtitle.clear()
    p = subtitle.paragraphs[0]
    p.text = f"Updated {date.today().isoformat()}"
    p.font.size = SUBTITLE_FONT

    # 2) Workflow overview
    s = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(s.shapes.title, "Workflow Overview", size=Pt(26))
    add_bullets(
        s,
        [
            "1. Plot raw force/theta sync traces across interventions.",
            "2. Apply filtering and preload/theta0 corrections.",
            "3. Detect and mark ramp starts.",
            "4. Mark transition-flagged cycles as ineligible.",
            "5. Fit force-tip and moment-theta stiffness (some cycles can fail fit criteria).",
            "6. Report rankings from valid fitted cycles.",
        ],
    )

    # 3) raw force vs time
    add_image_slide(
        prs,
        "Step 1: Raw Force vs Time",
        FIG_RAW_FORCE_TIME,
        "Raw distraction force trace with synchronized angle per intervention.",
    )

    # 4) preload-free force vs time
    add_image_slide(
        prs,
        "Step 2: Force Without Preload vs Time",
        FIG_PRELOADFREE_TIME,
        "Preload-corrected force; dashed lines show secant force references.",
    )

    # 5) ramp starts
    add_image_slide(
        prs,
        "Step 3: Ramp Start Detection",
        FIG_RAMP_STARTS,
        "Ramp-start preload force tracked across cycles/interventions.",
    )

    # 6) ineligible cycles summary
    add_ineligible_summary_slide(prs)

    # 7) force vs displacement
    add_image_slide(
        prs,
        "Step 5: Force vs Tip Displacement",
        FIG_FORCE_DISP,
        "Blue segment is the selected linear region used for stiffness fit.",
    )

    # 8) moment vs theta
    add_image_slide(
        prs,
        "Step 6: Moment vs Theta",
        FIG_MOMENT_THETA,
        "Final rotational stiffness k_rot computed from the selected linear region.",
    )

    # 9) ranking
    add_spreader_ranking_slide(prs)

    prs.save(PPTX_PATH)
    print(f"saved {PPTX_PATH}")


if __name__ == "__main__":
    main()
