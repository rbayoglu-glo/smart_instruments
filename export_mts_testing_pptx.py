from datetime import date
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt


BASE = Path(__file__).parent
OUT = BASE / "output"
PPTX_PATH = OUT / f"mts_testing_data_slides_{date.today().isoformat()}.pptx"

INTERVENTION_FIGS = [
    "Intact_stiffness.png",
    "PUF_Left_stiffness.png",
    "FUF_Left_stiffness.png",
    "FBF_stiffness.png",
    "Posterior_Release_stiffness.png",
    "SPO_stiffness.png",
]

FIG_FLEXION_PANELS = OUT / "mts_flexion_0to5_secant_fit_panels.png"
FIG_FLEXION_BAR = OUT / "mts_flexion_0to5_secant_stiffness_bar.png"
FIG_FLEXION_METHODS = OUT / "mts_flexion_stiffness_four_methods_bar.png"
FIG_EXTENSION_BAR = OUT / "mts_extension_0to5_secant_stiffness_bar.png"
FIG_EXTENSION_METHODS = OUT / "mts_extension_stiffness_four_methods_bar.png"
FIG_COEFF = OUT / "fe_coefficient_illustration.png"
CSV_FLEXION = OUT / "mts_flexion_0to5_secant_stiffness.csv"

TITLE_FONT = Pt(32)
SUBTITLE_FONT = Pt(18)
BODY_FONT = Pt(20)
MONO_FONT = Pt(13)


def set_title(shape, text: str, size: Pt = TITLE_FONT) -> None:
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.bold = True
    run.font.size = size


def add_bullets(slide, lines: list[str]) -> None:
    box = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.8), Inches(5.5))
    tf = box.text_frame
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = BODY_FONT


def add_image_slide(prs: Presentation, title: str, image_path: Path) -> None:
    if not image_path.exists():
        return
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(slide.shapes.title, title, size=Pt(26))
    slide.shapes.add_picture(str(image_path), Inches(0.5), Inches(0.9), width=Inches(12.3))


def add_ranking_slide(prs: Presentation) -> None:
    if not CSV_FLEXION.exists():
        return

    df = pd.read_csv(CSV_FLEXION)
    df = df[df["valid"] == True].copy()
    if df.empty:
        return

    df = df.sort_values("k_abs_nm_per_deg", ascending=False).reset_index(drop=True)

    lines = [
        "MTS Flexion Ranking (native negative axes, 0 to -5 Nm secant)",
        "",
        "Rank  Intervention          k [Nm/deg]   R2      n_points",
        "----------------------------------------------------------",
    ]
    for i, row in df.iterrows():
        lines.append(
            f"{i+1:>4}  {row['Intervention']:<20} {row['k_abs_nm_per_deg']:>8.3f}   {row['r2']:>5.3f}   {int(row['n_points_window']):>7}"
        )

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(slide.shapes.title, "MTS Flexion Stiffness Ranking", size=Pt(26))

    box = slide.shapes.add_textbox(Inches(0.7), Inches(1.0), Inches(12.0), Inches(5.9))
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

    # Slide 1: title
    s = prs.slides.add_slide(prs.slide_layouts[0])
    set_title(s.shapes.title, "MTS Testing Data Summary")
    subtitle = s.placeholders[1].text_frame
    subtitle.clear()
    p = subtitle.paragraphs[0]
    p.text = f"Updated {date.today().isoformat()} | Axes flipped: Angle on X, Moment on Y"
    p.font.size = SUBTITLE_FONT

    # Slide 2: method snapshot
    s = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(s.shapes.title, "Method Snapshot", size=Pt(26))
    add_bullets(
        s,
        [
            "Input: MTS 6DOF Book*.xlsx grouped as FE, LB, AR per intervention",
            "Cycle basis: third cycle, centered avg(load/unload)",
            "Model: M(theta) = C1*theta + C2*theta^2 + C3*theta^3",
            "Plot orientation: angle on x-axis, moment on y-axis",
            "Flexion branch display: native negative theta and negative moment",
            "Flexion stiffness window: 0 to -5 Nm secant",
        ],
    )

    add_ranking_slide(prs)

    # One slide per intervention (FE/LB/AR in each figure)
    for fig_name in INTERVENTION_FIGS:
        image_path = OUT / fig_name
        title = fig_name.replace("_stiffness.png", "").replace("_", " ")
        add_image_slide(prs, f"Intervention: {title}", image_path)

    # Flexion 1-5 summaries
    add_image_slide(prs, "MTS FE (Native Axes): Extension 0-5 and Flexion 0 to -5 Nm Secant Fit Panels", FIG_FLEXION_PANELS)
    add_image_slide(prs, "MTS Flexion (Native Negative Axes): 0 to -5 Nm Secant Stiffness Bar", FIG_FLEXION_BAR)
    add_image_slide(prs, "MTS Flexion (Native Negative Axes): Four-Method Stiffness Comparison", FIG_FLEXION_METHODS)
    add_image_slide(prs, "MTS Extension (Native Positive Axes): 0 to 5 Nm Secant Stiffness Bar", FIG_EXTENSION_BAR)
    add_image_slide(prs, "MTS Extension (Native Positive Axes): Four-Method Stiffness Comparison", FIG_EXTENSION_METHODS)

    # Coefficient illustration
    add_image_slide(prs, "Coefficient Illustration (FE)", FIG_COEFF)

    prs.save(PPTX_PATH)
    print(f"saved {PPTX_PATH}")


if __name__ == "__main__":
    main()
