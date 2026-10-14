"""Record the actual depth-reference implementation and experiment assumptions."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .terrain import terrain_wall_summary


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
        "reference_experiment": asdict(env_cfg.visual_reference) if hasattr(env_cfg, "visual_reference") else None,
        "terrain": terrain_wall_summary(env_cfg.scene.terrain.terrain_generator),
        "touchdown_measurement": "first-contact URDF shoe FK in contact anchor frame; confirmed later",
        "execution_error_definition": "first_contact_actual - final_projected_applied_reference",
        "bias_frame": "gravity-levelled torso heading XY at first contact",
        "bias_reset": "per environment, per episode; no cross-episode calibration",
        "scope": "Visual footprint estimate and commanded FK touchdown error; no pressure/slip certificate",
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(directory.glob("*.py"))},
    }
    urdf = directory.parents[2] / "urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
    metadata["urdf_sha256"] = hashlib.sha256(urdf.read_bytes()).hexdigest()
    (output / "visual_foothold_runtime.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    feedback = metadata["reference_experiment"]
    if feedback is not None:
        print(f"[VISUAL REFERENCE] mode={feedback['mode']}, alpha={feedback['ewma_alpha']}, "
              f"min_samples={feedback['min_samples']}, max_compensation_m={feedback['max_compensation_m']}", flush=True)
    print(f"[VISUAL TERRAIN] random_walls_enabled={metadata['terrain']['random_walls_enabled']}", flush=True)
    print(f"[VISUAL] Runtime assumptions/source hashes saved to {output}", flush=True)
