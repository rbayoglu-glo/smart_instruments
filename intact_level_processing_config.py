from pathlib import Path


DATA_ROOT = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\UTN24_26016")
SPREADER_ROOT = DATA_ROOT / "segment" / "spreader"
LUMBAR_ACQUISITION_DATE = "2026-09-04"
OUTPUT_DIR = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Data_analysis\output")

LEVEL_CONFIGS = {
    "L1-L2": {
        "force_dir": SPREADER_ROOT / "Force" / "lamspreader",
        "tracking_dir": SPREADER_ROOT / "Displacement" / "gns-sensorhub",
        "trials": {
            "L1-L2 Left": {
                "force_file": "L1-L2 Left.csv",
                "tracking_folder": "2026-11-08_[14-24-04]",
            },
            "L1-L2 Right": {
                "force_file": "L1-L2 Right.csv",
                "tracking_folder": "2026-11-08_[14-41-43]",
            },
        },
    },
    "L2-L3": {
        "force_dir": SPREADER_ROOT / "Force" / "lamspreader",
        "tracking_dir": SPREADER_ROOT / "Displacement" / "gns-sensorhub",
        "trials": {
            "L2-L3 Left": {
                "force_file": "L2-L3 Left.csv",
                "tracking_folder": "2026-11-08_[14-55-39]",
            },
            "L2-L3 Right": {
                "force_file": "L2-L3 Right.csv",
                "tracking_folder": "2026-11-08_[15-11-40]",
            },
        },
    },
    "L3-L4": {
        "force_dir": SPREADER_ROOT / "Force" / "lamspreader",
        "tracking_dir": SPREADER_ROOT / "Displacement" / "gns-sensorhub",
        "trials": {
            "L3-L4 Left": {
                "force_file": "L3-L4 Left.csv",
                "tracking_folder": "2026-11-08_[15-27-44]",
            },
            "L3-L4 Right": {
                "force_file": "L3-L4 Right.csv",
                "tracking_folder": "2026-11-08_[15-37-44]",
            },
        },
    },
    "L4-L5": {
        "force_dir": SPREADER_ROOT / "Force" / "lamspreader",
        "tracking_dir": SPREADER_ROOT / "Displacement" / "gns-sensorhub",
        "trials": {
            "L4-L5 Left": {
                "force_file": "L4-L5 Left.csv",
                "tracking_folder": "2026-11-08_[15-50-03]",
            },
            "L4-L5 Right": {
                "force_file": "L4-L5 Right.csv",
                "tracking_folder": "2026-11-08_[16-06-03]",
            },
        },
    },
}


def get_level_config(level_name: str) -> dict:
    try:
        config = LEVEL_CONFIGS[level_name]
    except KeyError as exc:
        available = ", ".join(sorted(LEVEL_CONFIGS))
        raise ValueError(f"Unknown level {level_name!r}; available levels: {available}") from exc

    if any(values["tracking_folder"] is None for values in config["trials"].values()):
        raise ValueError(
            f"{level_name}: verify UTN24_26016 lumbar tracking mappings for acquisition "
            f"{LUMBAR_ACQUISITION_DATE} before processing."
        )

    return {
        "force_dir": config["force_dir"],
        "tracking_dir": config["tracking_dir"],
        "trials": {name: dict(values) for name, values in config["trials"].items()},
        "output_dir": OUTPUT_DIR,
    }