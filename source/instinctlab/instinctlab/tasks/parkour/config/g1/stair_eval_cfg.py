"""Evaluation-only straight stairs, using the trained Parkour observation pipeline."""

from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch
import trimesh

from isaaclab.managers import ManagerTermBase, TerminationTermCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from instinctlab.tasks.parkour.mdp.commands.pose_velocity_command import PoseVelocityCommand


class StaircaseGenerator:
    """The custom TerrainImporter supports this generator through hacked_generator.

    One column per environment gives deterministic lane assignment. Target patches
    are world coordinates, matching PoseVelocityCommand's existing interface.
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.cases = cfg.cases
        origins = []
        targets = []
        meshes = []
        for case in self.cases:
            lane_y = (case["env_id"] - (len(self.cases) - 1) / 2) * cfg.size[1]
            origins.append([0.0, lane_y, case["start_height_m"]])
            targets.append([case["goal_x_m"], lane_y, case["end_height_m"]])
            if "profile" in case:
                vertices = np.asarray(case["mesh_vertices"], dtype=np.float64).copy()
                vertices[:, 1] += lane_y
                meshes.append(trimesh.Trimesh(vertices=vertices, faces=case["mesh_faces"], process=False))
                continue
            edge = case["stair_start_x_m"]
            spans = [(case["lane_min_x_m"], edge, case["start_height_m"])]
            for depth, height in zip(case["tread_depths_m"], case["surface_heights_m"]):
                spans.append((edge, edge + depth, height))
                edge += depth
            spans.append((edge, case["lane_max_x_m"], case["end_height_m"]))
            if case["direction"] == "flat":
                # One slab avoids artificial collision seams in the flat control.
                spans = [(case["lane_min_x_m"], case["lane_max_x_m"], 0.0)]
            for left, right, height in spans:
                bottom = -0.10
                mesh = trimesh.creation.box(extents=(right - left, case["width_m"], height - bottom))
                mesh.apply_translation(((left + right) / 2, lane_y, (height + bottom) / 2))
                meshes.append(mesh)
        self.terrain_mesh = trimesh.util.concatenate(meshes)
        self.terrain_origins = np.asarray(origins, dtype=np.float32)[None, :, :]
        self.flat_patches = {
            "target": torch.tensor(targets, dtype=torch.float32, device=device)[None, :, None, :]
        }


@configclass
class StaircaseGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = StaircaseGenerator
    cases: list = []


class StairGoalCommand(PoseVelocityCommand):
    """Use the same target-to-velocity controller, with a fixed landing target."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.max_command_b[env_ids, 0] = self.cfg.ranges.lin_vel_x[1]
        self.max_command_b[env_ids, 1] = self.cfg.ranges.lin_vel_y[1]
        self.max_command_b[env_ids, 2] = self.cfg.ranges.ang_vel_z[1]
        self.is_standing_env[env_ids] = False


def straight_velocity_command(heading, speed, yaw_limit, gain):
    """Body-frame forward command and yaw feedback toward world +X.

    This changes only commands given to the frozen actor; it never changes the
    robot's pose or policy actions. Lateral command is always zero.
    """
    result = torch.zeros(heading.shape[0], 3, device=heading.device, dtype=heading.dtype)
    result[:, 0] = speed
    error = torch.atan2(torch.sin(-heading), torch.cos(-heading))
    result[:, 2] = torch.clamp(gain * error, min=-yaw_limit, max=yaw_limit)
    return result


class StraightWalkCommand(StairGoalCommand):
    """Diagnostic flat walk: hold world heading instead of steering to a goal."""

    def _update_command(self):
        super()._update_command()
        self.vel_command_b[:] = straight_velocity_command(
            self.robot.data.heading_w, self.cfg.ranges.lin_vel_x[1],
            self.cfg.ranges.ang_vel_z[1], self.cfg.heading_control_stiffness,
        )


def _case_tensors(env):
    if not hasattr(env, "_stair_eval_cases"):
        generator = env.scene.terrain.terrain_generator
        cases = [generator.cases[i] for i in env.scene.terrain.terrain_types.cpu().tolist()]
        env._stair_eval_cases = cases
        env._stair_eval_bounds = {
            key: torch.tensor([case[key] for case in cases], dtype=torch.float32, device=env.device)
            for key in (
                "width_m", "lane_min_x_m", "lane_max_x_m", "stair_end_x_m", "goal_x_m", "end_height_m",
                "summit_start_x_m", "summit_end_x_m", "summit_height_m",
            )
        }
        env._stair_requires_summit = torch.tensor(
            [case["requires_summit"] for case in cases], dtype=torch.bool, device=env.device
        )
        edges, levels, slopes, starts = [], [], [], []
        for case in cases:
            if "profile" in case:
                profile = case["profile"]
                edges.append([segment[1] for segment in profile])
                levels.append([segment[2] for segment in profile] + [case["end_height_m"]])
                slopes.append([(high - low) / (right - left) for left, right, low, high in profile] + [0.0])
                starts.append([segment[0] for segment in profile] + [profile[-1][1]])
            else:
                row = [case["stair_start_x_m"]]
                for depth in case["tread_depths_m"][:-1]:
                    row.append(row[-1] + depth)
                edges.append(row)
                levels.append([case["start_height_m"], *case["surface_heights_m"]])
                slopes.append([0.0] * len(levels[-1]))
                starts.append([0.0] * len(levels[-1]))
        # Suite lanes have different segment counts. Infinity keeps padding inactive.
        count = max(map(len, edges))
        for row, level, slope, start in zip(edges, levels, slopes, starts):
            missing = count - len(row)
            row.extend([float("inf")] * missing)
            level.extend([level[-1]] * missing)
            slope.extend([0.0] * missing)
            start.extend([0.0] * missing)
        env._stair_eval_edges = torch.tensor(edges, dtype=torch.float32, device=env.device)
        env._stair_eval_levels = torch.tensor(levels, dtype=torch.float32, device=env.device)
        env._stair_eval_slopes = torch.tensor(slopes, dtype=torch.float32, device=env.device)
        env._stair_eval_starts = torch.tensor(starts, dtype=torch.float32, device=env.device)
    return env._stair_eval_bounds


def ground_height(env, local_x):
    _case_tensors(env)
    index = (local_x[:, None] >= env._stair_eval_edges).sum(dim=-1)
    index = index[:, None]
    level = env._stair_eval_levels.gather(1, index).squeeze(1)
    slope = env._stair_eval_slopes.gather(1, index).squeeze(1)
    start = env._stair_eval_starts.gather(1, index).squeeze(1)
    return level + slope * (local_x - start).clamp(min=0.0)


def reset_stair_root(env, env_ids, centered_start=False):
    """Per-lane reset seeds do not depend on when other policies/lanes terminate."""
    _case_tensors(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "_stair_spawn_counts"):
        env._stair_spawn_counts = [0] * env.num_envs
        env._stair_spawn_offsets = torch.zeros(env.num_envs, 3, device=env.device)
    ids = env_ids.cpu().tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
    offsets = []
    for env_id in ids:
        rng = random.Random(env._stair_eval_cases[env_id]["case_seed"] + 65537 * env._stair_spawn_counts[env_id])
        offsets.append([0.0, 0.0, 0.0] if centered_start else [rng.uniform(-0.05, 0.05) for _ in range(3)])
        env._stair_spawn_counts[env_id] += 1
    offsets = torch.tensor(offsets, dtype=torch.float32, device=env.device).reshape(-1, 3)
    env._stair_spawn_offsets[env_ids] = offsets
    robot = env.scene["robot"]
    state = robot.data.default_root_state[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
    state[:, :2] += offsets[:, :2]
    zeros = torch.zeros(len(ids), device=env.device)
    yaw_delta = quat_from_euler_xyz(zeros, zeros, offsets[:, 2])
    state[:, 3:7] = quat_mul(yaw_delta, state[:, 3:7])
    state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)


def stair_root_height(env, minimum_height=0.5):
    robot = env.scene["robot"]
    local_x = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return robot.data.root_pos_w[:, 2] - ground_height(env, local_x) < minimum_height


def off_course(env):
    bounds = _case_tensors(env)
    xy = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return (
        (xy[:, 0] < bounds["lane_min_x_m"] + 0.1)
        | (xy[:, 0] > bounds["lane_max_x_m"] - 0.1)
        | (xy[:, 1].abs() > bounds["width_m"] / 2 - 0.1)
    )


def nonfinite_state(env):
    data = env.scene["robot"].data
    return ~(
        torch.isfinite(data.root_state_w).all(dim=-1)
        & torch.isfinite(data.joint_pos).all(dim=-1)
        & torch.isfinite(data.joint_vel).all(dim=-1)
    )


class StairSuccess(ManagerTermBase):
    """Capture exact post-physics states before automatic reset, including failures."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hold_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.summit_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.snapshot = None
        _case_tensors(env)
        checkpoints = [case.get("checkpoints", []) for case in env._stair_eval_cases]
        count = max((len(row) for row in checkpoints), default=0)
        self.checkpoint_valid = torch.tensor(
            [[True] * len(row) + [False] * (count - len(row)) for row in checkpoints],
            dtype=torch.bool, device=env.device,
        ).reshape(env.num_envs, count)
        self.checkpoint_reached = torch.zeros_like(self.checkpoint_valid)
        self.checkpoint_bounds = {
            key: torch.tensor(
                [[point[key] for point in row] + [0.0] * (count - len(row)) for row in checkpoints],
                dtype=torch.float32, device=env.device,
            ).reshape(env.num_envs, count)
            for key in ("start_x_m", "end_x_m", "height_m")
        }
        self.foot_ids, _ = env.scene["robot"].find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        self.contact_ids, _ = env.scene["contact_forces"].find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        if len(self.foot_ids) != 2 or len(self.contact_ids) != 2:
            raise RuntimeError("Stair evaluation requires both ankle links and their contact sensors")

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.hold_steps[ids] = 0
        self.summit_reached[ids] = False
        self.checkpoint_reached[ids] = False
        # Keep snapshot intact: the caller reads it after env.step has auto-reset.

    def __call__(self, env, hold_s=0.5, goal_radius=0.35):
        bounds = _case_tensors(env)
        robot = env.scene["robot"]
        pos = robot.data.root_pos_w.clone()
        pos[:, :2] -= env.scene.env_origins[:, :2]
        feet = robot.data.body_pos_w[:, self.foot_ids].clone()
        feet[:, :, :2] -= env.scene.env_origins[:, None, :2]
        forces = env.scene["contact_forces"].data.net_forces_w[:, self.contact_ids].norm(dim=-1)
        floor = ground_height(env, pos[:, 0])
        clearance = pos[:, 2] - floor
        gravity_z = robot.data.projected_gravity_b[:, 2]
        supported_upright = (
            (forces.max(dim=-1).values > 5.0) & (gravity_z < -math.cos(0.5)) & (clearance > 0.5)
        )
        on_summit = (
            (feet[:, :, 0] > bounds["summit_start_x_m"][:, None] + 0.05)
            & (feet[:, :, 0] < bounds["summit_end_x_m"][:, None] - 0.05)
            & (feet[:, :, 1].abs() < bounds["width_m"][:, None] / 2 - 0.1)
            & ((feet[:, :, 2] - bounds["summit_height_m"][:, None]).abs() < 0.20)
        ).all(dim=-1)
        on_summit &= (pos[:, 0] > bounds["summit_start_x_m"]) & (pos[:, 0] < bounds["summit_end_x_m"])
        self.summit_reached |= env._stair_requires_summit & on_summit & supported_upright
        # Each obstacle in a combined route requires a supported summit landing.
        points = self.checkpoint_bounds
        on_checkpoint = (
            (feet[:, :, 0, None] > points["start_x_m"][:, None, :] + 0.05)
            & (feet[:, :, 0, None] < points["end_x_m"][:, None, :] - 0.05)
            & (feet[:, :, 1, None].abs() < bounds["width_m"][:, None, None] / 2 - 0.1)
            & ((feet[:, :, 2, None] - points["height_m"][:, None, :]).abs() < 0.20)
        ).all(dim=1)
        on_checkpoint &= (pos[:, 0, None] > points["start_x_m"]) & (pos[:, 0, None] < points["end_x_m"])
        self.checkpoint_reached |= on_checkpoint & supported_upright[:, None] & self.checkpoint_valid
        route_passed = (self.checkpoint_reached | ~self.checkpoint_valid).all(dim=1)
        distance = torch.sqrt((pos[:, 0] - bounds["goal_x_m"]) ** 2 + pos[:, 1] ** 2)
        on_landing = (feet[:, :, 0] > bounds["stair_end_x_m"][:, None] + 0.1).all(dim=-1)
        on_landing &= (feet[:, :, 1].abs() < bounds["width_m"][:, None] / 2 - 0.1).all(dim=-1)
        feet_near_floor = (
            (feet[:, :, 2] - bounds["end_height_m"][:, None]).abs() < 0.20
        ).all(dim=-1)
        eligible = (
            (distance < goal_radius)
            & on_landing
            & feet_near_floor
            & supported_upright
            & (~env._stair_requires_summit | self.summit_reached)
            & route_passed
        )
        self.hold_steps[:] = torch.where(eligible, self.hold_steps + 1, 0)
        self.snapshot = {
            "step": env.episode_length_buf.clone(),
            "pos": pos,
            "feet": feet,
            "foot_forces": forces.clone(),
            "floor": floor.clone(),
            "clearance": clearance,
            "gravity_z": gravity_z.clone(),
            "velocity": robot.data.root_lin_vel_b.clone(),
            "velocity_w": robot.data.root_lin_vel_w.clone(),
            "heading_w": robot.data.heading_w.clone(),
            "joint_velocity_rms_rad_s": robot.data.joint_vel.square().mean(dim=-1).sqrt(),
            "applied_torque_rms_nm": robot.data.applied_torque.square().mean(dim=-1).sqrt(),
            "yaw_rate": robot.data.root_ang_vel_b[:, 2].clone(),
            "command": env.command_manager.get_command("base_velocity").clone(),
            "goal_distance": distance,
            "spawn_offsets": env._stair_spawn_offsets.clone(),
            "summit_reached": self.summit_reached.clone(),
            "checkpoints_passed": self.checkpoint_reached.sum(dim=1),
            "checkpoints_required": self.checkpoint_valid.sum(dim=1),
        }
        return self.hold_steps >= math.ceil(hold_s / env.step_dt)


def configure_stair_evaluation(env_cfg, cases, seed, speed, episode_length_s, hold_s,
                               command_mode="goal", centered_start=False):
    """Change terrain, resets and goals only; preserve trained sensors and action scales."""
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.seed = seed
    env_cfg.scene.num_envs = len(cases)
    env_cfg.episode_length_s = episode_length_s
    env_cfg.scene.terrain.terrain_type = "hacked_generator"
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.scene.terrain.terrain_generator = StaircaseGeneratorCfg(
        seed=seed,
        size=(max(case["lane_max_x_m"] for case in cases) + 1.2, cases[0]["width_m"] + 2.0),
        num_rows=1,
        num_cols=len(cases),
        border_width=0.0,
        curriculum=False,
        use_cache=False,
        sub_terrains={},
        cases=cases,
    )
    env_cfg.curriculum.terrain_levels = None
    command = env_cfg.commands.base_velocity
    if command_mode not in ("goal", "straight"):
        raise ValueError("Unknown evaluation command mode: " + command_mode)
    if command_mode == "straight" and any(case["direction"] != "flat" for case in cases):
        raise ValueError("Straight command control is only supported for flat evaluation")
    command.class_type = StraightWalkCommand if command_mode == "straight" else StairGoalCommand
    command.velocity_ranges = None
    command.random_velocity_terrain = None
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (episode_length_s + 1, episode_length_s + 1)
    command.ranges.lin_vel_x = (speed, speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (-1.0, 1.0)
    command.target_dis_threshold = 0.25
    command.debug_vis = False
    env_cfg.events.reset_base.func = reset_stair_root
    env_cfg.events.reset_base.params = {"centered_start": centered_start}
    env_cfg.events.reset_robot_joints.params = {
        "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)
    }
    # Isolate the stair task while retaining the trained camera/observation noise.
    env_cfg.events.physics_material = None
    for name, event in vars(env_cfg.events).items():
        if getattr(event, "mode", None) == "interval":
            setattr(env_cfg.events, name, None)
    # Frozen-actor inference does not use AMASS motions or AMP discriminator inputs.
    # Remove these before scene/observation managers are created so no dataset is read.
    env_cfg.scene.motion_reference = None
    env_cfg.observations.amp_policy = None
    env_cfg.observations.amp_reference = None
    env_cfg.terminations.dataset_exhausted = None
    # Training completion uses the training grid; evaluation has held-out lanes.
    env_cfg.terminations.stair_goal = None
    # Curb training state assumes a curriculum grid. Frozen evaluation uses its
    # own lanes and passive touchdown diagnostics, so remove training-only terms.
    env_cfg.terminations.curb_goal = None
    if hasattr(env_cfg.events, "curb_crossing_reset"):
        env_cfg.events.curb_crossing_reset = None
    for name in ("curb_bridge", "curb_gap_ground"):
        if hasattr(env_cfg.rewards.rewards, name):
            setattr(env_cfg.rewards.rewards, name, None)
    env_cfg.terminations.terrain_out_bound = None
    env_cfg.terminations.root_height = TerminationTermCfg(func=stair_root_height)
    env_cfg.terminations.off_course = TerminationTermCfg(func=off_course)
    env_cfg.terminations.nonfinite_state = TerminationTermCfg(func=nonfinite_state)
    env_cfg.terminations.stair_success = TerminationTermCfg(func=StairSuccess, params={"hold_s": hold_s})
    env_cfg.scene.leg_volume_points.debug_vis = False
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (4.0, 4.0, 3.0)
    env_cfg.viewer.lookat = (1.0, 0.0, 0.0)
    return env_cfg
