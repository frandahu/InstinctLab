"""Record the actual depth-reference implementation and experiment assumptions."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def save_visual_runtime(env_cfg, log_dir):
    directory = Path(__file__).resolve().parent
    output = Path(log_dir) / "params"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "variant": "B" if hasattr(env_cfg, "visual_foothold") else "A",
        "reference_source": "metric_camera_depth_only" if hasattr(env_cfg, "visual_foothold") else "none",
        "terrain_truth_in_planner": False,
        "world_robot_pose_in_planner": False,
        "contact_source": "simulated_upward_ankle_contact_force",
        "alignment_assumption": "short rigid stationary support; invalidate on unsupported transfer",
        "depth_crop_top_bottom_left_right": [18, 0, 16, 16],
        "depth_geometry_before_blur": True,
        "depth_delay_control_steps": 1,
        "step_dt": env_cfg.sim.dt * env_cfg.decimation,
        "settings": asdict(env_cfg.visual_foothold) if hasattr(env_cfg, "visual_foothold") else None,
        "scope": "Visual footprint estimate and commanded FK touchdown error; no pressure/slip certificate",
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(directory.glob("*.py"))},
    }
    urdf = directory.parents[2] / "urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
    metadata["urdf_sha256"] = hashlib.sha256(urdf.read_bytes()).hexdigest()
    (output / "visual_foothold_runtime.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[VISUAL] Runtime assumptions/source hashes saved to {output}", flush=True)
