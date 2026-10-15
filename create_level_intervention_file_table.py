#%%
"""Create a table of source files used for every level and intervention."""

from pathlib import Path
import re

import pandas as pd

import plot_l4l5_all_interventions_fe_cycle3_mean as l4_fe
import plot_l4l5_all_interventions_lb_ar_cycle3_centered_mean as l4_lb_ar
import plot_levels_all_interventions_fe_cycles_centered_mean_stacked as other_fe
import plot_levels_all_interventions_lb_ar_cycles_centered_mean_stacked as other_lb_ar


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "all_levels_interventions_source_files.csv"
OUT_MD = OUT_DIR / "all_levels_interventions_source_files.md"
LEVELS = ("L4-L5", "L2-L3", "T12-L1", "T10-T11", "T8-T9", "T6-T7", "T4-T5", "T2-T3")


def numeric_book_number(path: Path) -> int:
    return int(re.search(r"(\d+)", path.stem).group(1))


def numeric_opto_number(path: Path) -> int:
    return int(re.search(r"_(\d+)_cal", path.name).group(1))


def sorted_motor_files(folder: Path) -> list[Path]:
    return sorted((folder / "Motor").glob("Book*.xlsx"), key=numeric_book_number)


def sorted_opto_files(folder: Path) -> list[Path]:
    return sorted((folder / "Optotrak").glob("*.xls"), key=numeric_opto_number)


def l4_complete_cycles(folder: Path, config: dict) -> str:
    motor_time, moment, _ = l4_lb_ar.load_motor(folder, config)
    opto_time, angle, _ = l4_lb_ar.load_optotrak(folder, config)
    boundaries = l4_lb_ar.fe_cycle_boundaries(motor_time, moment)
    aligned, valid, lag, _ = l4_lb_ar.align_angle_to_motor(
        motor_time,
        moment,
        opto_time,
        angle,
        boundaries[1] - boundaries[0],
    )
    available = []
    for cycle_number, (cycle_start, cycle_end) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        cycle_mask = (motor_time >= cycle_start) & (motor_time < cycle_end)
        keep = valid & cycle_mask & pd.notna(aligned)
        has_full_coverage = (
            cycle_start >= opto_time[0] + lag
            and cycle_end <= opto_time[-1] + lag
            and int(keep.sum()) == int(cycle_mask.sum())
            and int(keep.sum()) >= 20
        )
        if has_full_coverage:
            available.append(cycle_number)
    return ",".join(map(str, available))


def l4_rows() -> list[dict]:
    level_dir = l4_fe.LEVEL_DIR
    rows = []
    for folder in l4_lb_ar.intervention_folders():
        name = re.sub(r"^\d+\.\s*", "", folder.name)
        motor_files = sorted_motor_files(folder)
        opto_files = sorted_opto_files(folder)
        fe_config = l4_fe.FE_FILE_OVERRIDES.get(folder.name, {})
        lb_config = dict(l4_lb_ar.AXES["LB"])
        lb_config.update(l4_lb_ar.FILE_OVERRIDES.get("LB", {}).get(folder.name, {}))
        ar_config = l4_lb_ar.AXES["AR"]
        lb_complete_cycles = l4_complete_cycles(folder, lb_config)
        ar_complete_cycles = l4_complete_cycles(folder, ar_config)
        rows.append({
            "level": "L4-L5",
            "intervention": name,
            "fe_motor_file": motor_files[fe_config.get("motor_index", 0)].name,
            "fe_optotrak_file": opto_files[fe_config.get("opto_index", 0)].name,
            "lb_motor_file": motor_files[lb_config["motor_index"]].name,
            "lb_optotrak_file": opto_files[lb_config["opto_index"]].name,
            "ar_motor_file": motor_files[ar_config["motor_index"]].name,
            "ar_optotrak_file": opto_files[ar_config["opto_index"]].name,
            "fe_complete_cycles": "1,2,3",
            "lb_complete_cycles": lb_complete_cycles,
            "ar_complete_cycles": ar_complete_cycles,
        })
    return rows


def cases_by_name(cases: list[dict]) -> dict[str, dict]:
    return {case["name"]: case for case in cases}


def other_level_rows(level: str) -> list[dict]:
    fe_cases = cases_by_name(other_fe.load_cases(level))
    lb_cases = cases_by_name(other_lb_ar.load_cases(level, "LB"))
    ar_cases = cases_by_name(other_lb_ar.load_cases(level, "AR"))
    names = list(fe_cases)
    rows = []
    for name in names:
        fe_case = fe_cases[name]
        lb_case = lb_cases[name]
        ar_case = ar_cases[name]
        rows.append({
            "level": level,
            "intervention": name,
            "fe_motor_file": fe_case["motor_file"],
            "fe_optotrak_file": fe_case["opto_file"],
            "lb_motor_file": lb_case["motor_file"],
            "lb_optotrak_file": lb_case["opto_file"],
            "ar_motor_file": ar_case["motor_file"],
            "ar_optotrak_file": ar_case["opto_file"],
            "fe_complete_cycles": ",".join(map(str, fe_case["available_cycles"])),
            "lb_complete_cycles": ",".join(map(str, lb_case["available_cycles"])),
            "ar_complete_cycles": ",".join(map(str, ar_case["available_cycles"])),
        })
    return rows


def markdown_table(data: pd.DataFrame) -> str:
    columns = list(data.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in data.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = l4_rows()
    for level in LEVELS[1:]:
        rows.extend(other_level_rows(level))
    data = pd.DataFrame(rows)
    data.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(
        "# Source files used by level and intervention\n\n"
        + markdown_table(data),
        encoding="utf-8",
    )
    print(f"saved {OUT_CSV}")
    print(f"saved {OUT_MD}")
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()
