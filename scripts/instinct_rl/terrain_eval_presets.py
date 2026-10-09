"""Evaluation geometry presets; explicit CLI options override preset defaults."""

import argparse


HARD_DEFAULTS = {
    "num_envs": 8,
    "num_steps": 8,
    "step_height_range": (0.10, 0.24),
    "tread_depth_range": (0.24, 0.42),
    "irregular": True,
    "irregular_fraction": 0.35,
    "dimension_variation": 0.40,
    "vary_both_dimensions": True,
    "curb_count": 3,
    "curb_height_range": (0.12, 0.24),
    "curb_depth_range": (0.80, 1.30),
    "curb_gap_range": (0.45, 0.90),
    "episode_length_s": 90.0,
    "video_view": "overview",
    "video_width": 1920,
    "video_height": 1080,
}


def apply_difficulty_defaults(parser, argv=None):
    """Select a preset before parsing the real CLI, retaining per-option overrides."""
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--difficulty", choices=("baseline", "hard"), default="baseline")
    selected, _ = selector.parse_known_args(argv)
    if selected.difficulty == "hard":
        parser.set_defaults(**HARD_DEFAULTS)
