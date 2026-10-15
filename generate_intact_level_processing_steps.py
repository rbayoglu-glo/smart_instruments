import argparse
from pathlib import Path

import analyze_l4l5_intact_prior_method as analysis
import highlight_step04
import plot_l4l5_intact_processing_steps as stages
from intact_level_processing_config import get_level_config
from thoracic_level_processing_config import get_level_config as get_thoracic_level_config


def configure(level_name: str) -> None:
    try:
        config = get_level_config(level_name)
        relative_time_mode = False
    except ValueError:
        config = get_thoracic_level_config(level_name)
        relative_time_mode = True
    force_dir = config["force_dir"]
    tracking_dir = config["tracking_dir"]
    trials = config["trials"]
    output_dir = config["output_dir"]

    analysis.LEVEL_NAME = level_name
    analysis.OUTPUT_PREFIX = "intraop_"
    analysis.FORCE_DIR = force_dir
    analysis.TRACKING_DIR = tracking_dir
    analysis.TRIALS = trials
    analysis.RELATIVE_TIME_MODE = relative_time_mode
    analysis.OUT_DIR = output_dir
    analysis.PLOT_OUT_DIR = output_dir

    if relative_time_mode:
        trials, rejected = analysis.split_synchronizable_trials()
        for name, correlation in rejected.items():
            print(f"EXCLUDED {name}: force/theta sync correlation {correlation:.2f} below gate")
        analysis.TRIALS = trials

    stages.LEVEL_NAME = level_name
    stages.OUTPUT_PREFIX = "intraop_"
    stages.FORCE_DIR = force_dir
    stages.TRACKING_DIR = tracking_dir
    stages.TRIALS = trials
    stages.RELATIVE_TIME_MODE = relative_time_mode
    stages.TRACKING_START_OFFSETS_S = {
        name: values.get("tracking_start_offset_s", 0.0)
        for name, values in trials.items()
    }
    stages.OUT_DIR = output_dir

    highlight_step04.OUT = output_dir / f"Step_04_intraop_{level_name}_baseline_sampling_highlight.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate generic intact-level processing steps.")
    parser.add_argument("--level", default="L3-L4")
    parser.add_argument("--steps", choices=("1-4", "1-6"), default="1-6")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    configure(args.level)
    if args.output_dir is not None:
        analysis.OUT_DIR = args.output_dir
        analysis.PLOT_OUT_DIR = args.output_dir
        stages.OUT_DIR = args.output_dir
        highlight_step04.OUT = args.output_dir / f"Step_04_intraop_{args.level}_baseline_sampling_highlight.png"
    stages.main()
    if args.steps == "1-6":
        highlight_step04.main()
        analysis.main()


if __name__ == "__main__":
    main()