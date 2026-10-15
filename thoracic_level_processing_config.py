from pathlib import Path


DATA_ROOT = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\UTN24_26016")
SPREADER_ROOT = DATA_ROOT / "segment" / "spreader"
OUTPUT_DIR = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Data_analysis\output")

LEVEL_CONFIGS = {
    "T1-T2": {
        "T1-T2 Left": {"force_file": "Session #77.csv", "tracking_folder": "2026-12-14_[09-13-39]"},
        "T1-T2 Right": {"force_file": "Session #78.csv", "tracking_folder": "2026-12-14_[09-18-41]"},
    },
    "T2-T3": {
        "T2-T3 Left": {"force_file": "Session #75.csv", "tracking_folder": "2026-12-14_[08-57-08]"},
        "T2-T3 Right": {"force_file": "Session #76.csv", "tracking_folder": "2026-12-14_[09-03-09]"},
    },
    "T3-T4": {
        "T3-T4 Left": {"force_file": "Session #73.csv", "tracking_folder": "2026-12-14_[08-47-55]"},
        "T3-T4 Right": {"force_file": "Session #74.csv", "tracking_folder": "2026-12-14_[08-52-59]"},
    },
    "T4-T5": {
        "T4-T5 Left": {"force_file": "Session #71.csv", "tracking_folder": "2026-12-14_[08-35-09]"},
        "T4-T5 Right": {"force_file": "Session #72.csv", "tracking_folder": "2026-12-14_[08-39-47]"},
    },
    "T5-T6": {
        "T5-T6 Left": {"force_file": "Session #68.csv", "tracking_folder": "2026-12-14_[08-12-09]"},
        "T5-T6 Right": {"force_file": "Session #69.csv", "tracking_folder": "2026-12-14_[08-17-28]"},
    },
    "T6-T7": {
        "T6-T7 Left": {"force_file": "Session #66.csv", "tracking_folder": "2026-12-14_[07-46-42]"},
        "T6-T7 Right": {"force_file": "Session #67.csv", "tracking_folder": "2026-12-14_[07-53-49]"},
    },
    "T7-T8": {
        "T7-T8 Left": {"force_file": "Session #64.csv", "tracking_folder": "2026-12-14_[07-35-21]"},
        "T7-T8 Right": {"force_file": "Session #65.csv", "tracking_folder": "2026-12-14_[07-41-35]"},
    },
    "T8-T9": {
        "T8-T9 Left": {"force_file": "Session #62.csv", "tracking_folder": "2026-12-14_[07-15-41]"},
        "T8-T9 Right": {"force_file": "Session #63.csv", "tracking_folder": "2026-12-14_[07-27-37]"},
    },
    "T9-T10": {
        "T9-T10 Left": {"force_file": "Session #60.csv", "tracking_folder": "2026-12-14_[07-02-11]"},
        "T9-T10 Right": {"force_file": "Session #61.csv", "tracking_folder": "2026-12-14_[07-09-01]"},
    },
    "T10-T11": {
        "T10-T11 Left": {"force_file": "Session #58.csv", "tracking_folder": "2026-12-14_[06-38-54]"},
        "T10-T11 Right": {"force_file": "Session #59.csv", "tracking_folder": "2026-12-14_[06-44-11]"},
    },
    "T11-T12": {
        "T11-T12 Left": {"force_file": "Session #56.csv", "tracking_folder": "2026-12-14_[06-18-12]"},
        "T11-T12 Right": {"force_file": "Session #57.csv", "tracking_folder": "2026-12-14_[06-28-23]"},
    },
    "T12-L1": {
        "T12-L1 Left": {"force_file": "Session #53.csv", "tracking_folder": "2026-12-14_[05-54-12]"},
        "T12-L1 Right": {"force_file": "Session #55.csv", "tracking_folder": "2026-12-14_[06-12-53]"},
    },
}

GROUPS = {
    "T1-T2_to_T4-T5": ("T1-T2", "T2-T3", "T3-T4", "T4-T5"),
    "T5-T6_to_T8-T9": ("T5-T6", "T6-T7", "T7-T8", "T8-T9"),
    "T9-T10_to_T12-L1": ("T9-T10", "T10-T11", "T11-T12", "T12-L1"),
}

RAMPSTART_GROUPS = {
    "T1-T2_to_T4-T5": GROUPS["T1-T2_to_T4-T5"],
    "T5-T6_to_T8-T9": GROUPS["T5-T6_to_T8-T9"],
    "T9-T10_to_T12-L1": GROUPS["T9-T10_to_T12-L1"],
}

def get_level_config(level_name: str) -> dict:
    try:
        trials = LEVEL_CONFIGS[level_name]
    except KeyError as exc:
        available = ", ".join(sorted(LEVEL_CONFIGS))
        raise ValueError(f"Unknown thoracic level {level_name!r}; available: {available}") from exc
    return {
        "force_dir": SPREADER_ROOT / "Force" / "lamspreader",
        "tracking_dir": SPREADER_ROOT / "Displacement" / "gns-sensorhub",
        "trials": trials,
        "output_dir": OUTPUT_DIR,
    }
