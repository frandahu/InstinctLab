"""Isaac adapter for the isolated visual foothold experiment."""

from pathlib import Path

import torch

from .controller import VisualFootholdController
from .kinematics import LegKinematics, quaternion_rotation
from .perception import depth_points


class VisualFootholdState:
    """Sample measurements once, and share them across observations and rewards."""

    def __init__(self, env):
        self.env = env
        self.settings = env.cfg.visual_foothold
        self.settings.validate(env.step_dt)
        self.robot = env.scene["robot"]
        self.camera = env.scene["camera"]
        cfg = self.camera.cfg
        if cfg.offset.convention != "world" or (cfg.pattern_cfg.height, cfg.pattern_cfg.width) != (36, 64):
            raise ValueError("Visual foothold v1 requires the existing 64x36 world-convention camera pattern")
        if abs(cfg.update_period - env.step_dt) > 1e-6:
            raise ValueError("Visual foothold v1 requires one camera frame per control step")
        urdf = Path(__file__).resolve().parents[3] / "urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
        self.fk = LegKinematics(urdf, list(self.robot.joint_names), env.device)
        self.contact_ids = []
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            ids, _ = env.scene["contact_forces"].find_bodies(name)
            if len(ids) != 1:
                raise ValueError(f"Expected one contact sensor body: {name}")
            self.contact_ids.append(ids[0])
        self.camera_r = quaternion_rotation(torch.tensor(cfg.offset.rot, device=env.device))
        self.camera_p = torch.tensor(cfg.offset.pos, device=env.device)
        self.core = VisualFootholdController(env.num_envs, 18*32, env.step_dt, self.settings, env.device)
        self.dirty = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_step = -1

    @torch.no_grad()
    def update(self):
        env = self.env
        tick = env.common_step_counter
        new_step = tick != self.last_step
        if not new_step and not self.dirty.any():
            return self.core
        fresh = self.dirty.clone()
        self.last_step = tick
        data = self.robot.data
        q, dq = data.joint_pos, data.joint_vel
        # Sensor-like measurement noise belongs before FK, not on its final output.
        if self.settings.joint_position_noise:
            q = q + (2*torch.rand_like(q)-1) * self.settings.joint_position_noise
        if self.settings.joint_velocity_noise:
            dq = dq + (2*torch.rand_like(dq)-1) * self.settings.joint_velocity_noise
        p, rotation, velocity = self.fk.compute(q, dq)
        raw = self.camera.data.output["distance_to_image_plane"][:, 18:, 16:48, 0].flatten(1)
        valid_raw = torch.isfinite(raw) & (raw > 0.1) & (raw < self.settings.depth_max - 0.01)
        depth = raw + torch.randn_like(raw) * self.settings.depth_noise_std
        # Local pattern directions only. Never consume camera.ray_hits_w or terrain geometry.
        rays = self.camera.ray_directions.reshape(env.num_envs, 36, 64, 3)[:, 18:, 16:48].flatten(1, 2)
        starts = self.camera.ray_starts.reshape(env.num_envs, 36, 64, 3)[:, 18:, 16:48].flatten(1, 2)
        points, valid = depth_points(depth, rays, starts, self.camera_r, self.camera_p, self.settings.depth_max)
        force = env.scene["contact_forces"].data.net_forces_w[:, self.contact_ids, 2]
        command = env.command_manager.get_command("base_velocity")
        if not all(torch.isfinite(value).all() for value in (q, dq, force, data.projected_gravity_b, command)):
            raise RuntimeError("Nonfinite joint, force, gravity or command input to visual foothold state")
        advance = torch.full_like(fresh, new_step) & ~fresh
        self.core.update(p, rotation, velocity, force, data.projected_gravity_b, command,
                         points, valid & valid_raw, advance, tick)
        self.dirty[:] = False
        return self.core

    def reset(self, ids):
        self.core.reset(ids)
        self.dirty[ids] = True


def state(env):
    if not hasattr(env, "visual_foothold_state"):
        env.visual_foothold_state = VisualFootholdState(env)
    return env.visual_foothold_state


def capture_before_reset(env):
    """Pre-reset lifecycle term; observes the step and never terminates an episode."""
    core = state(env).update()
    if getattr(env, "visual_recording", False):
        env.visual_snapshot = {
            "time": core.time.clone(), "observation": core.observation.clone(),
            "touchdown": core.last_touchdown.clone(), "error": core.last_error.clone(),
            "contact_state": core.contact.state.clone(),
            "stats_names": tuple(core.stats),
            "stats": torch.stack(tuple(core.stats.values()), -1).clone(),
            "command": env.command_manager.get_command("base_velocity").clone(),
        }
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def visual_observation(env):
    return state(env).update().observation.clone()


def visual_reward(env, term):
    return state(env).update().reward[term]
