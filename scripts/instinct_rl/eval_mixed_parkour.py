"""Frozen G1 policy validation with MP4 on stairs -> slopes -> curbs.

See eval_mixed_parkour.md. Uses the existing evaluator and saved training params.
"""

from eval_stairs import main


if __name__ == "__main__":
    main({"task": "Instinct-Parkour-Mixed-Amp-G1-v0", "terrain_mode": "mixed",
          "num_envs": 4, "num_steps": 4, "episodes_per_env": 3, "episode_length_s": 60.0})
