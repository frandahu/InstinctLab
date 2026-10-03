"""Fresh stair locomotion: forward routes, bounded gait rewards and progress curriculum."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import trimesh

from isaaclab.managers import CurriculumTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

import instinctlab.tasks.parkour.mdp as mdp
from .g1_parkour_target_amp_cfg import G1ParkourEnvCfg
from .stair_eval_cfg import StaircaseGenerator, StaircaseGeneratorCfg, StairGoalCommand, nonfinite_state
from .stair_training_cases import training_cases


class TrainingStaircaseGenerator:
    """Reuse the tested collision slabs, with independent rows of difficulty."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.grid_cases = training_cases(cfg.seed, cfg.num_rows, cfg.num_cols)
        meshes, origins, targets = [], [], []
        for row, cases in enumerate(self.grid_cases):
            row_cfg = copy.copy(cfg)
            row_cfg.cases = cases
            generated = StaircaseGenerator(row_cfg, device)
            offset = np.asarray([row * cfg.size[0], 0.0, 0.0], dtype=np.float32)
            generated.terrain_mesh.apply_translation(offset)
            meshes.append(generated.terrain_mesh)
            origins.append(generated.terrain_origins[0] + offset)
            targets.append(generated.flat_patches["target"][0] + torch.as_tensor(offset, device=device))
        self.terrain_mesh = trimesh.util.concatenate(meshes)
        self.terrain_origins = np.stack(origins)
        self.flat_patches = {"target": torch.stack(targets)}


@configclass
class TrainingStaircaseGeneratorCfg(StaircaseGeneratorCfg):
    class_type: type = TrainingStaircaseGenerator


class TrainingStairCommand(StairGoalCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.metrics["route_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["terrain_level"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()
        data = route_data(self._env)
        x = self.robot.data.root_pos_w[:, 0] - self._env.scene.env_origins[:, 0]
        self.metrics["route_progress"][:] = (x / data["goal_x_m"]).clamp(0.0, 1.0)
        self.metrics["terrain_level"][:] = self.terrain.terrain_levels.float()

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.max_command_b[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.ranges.lin_vel_x
        )


def route_data(env):
    """Cache all route geometry; select current row after every curriculum reset."""
    terrain = env.scene.terrain
    if not hasattr(env, "_training_stair_grid"):
        cases = [case for row in terrain.terrain_generator.grid_cases for case in row]
        data = {key: [case[key] for case in cases] for key in (
            "goal_x_m", "stair_end_x_m", "lane_min_x_m", "lane_max_x_m", "width_m"
        )}
        data["edges"] = []
        data["heights"] = []
        for case in cases:
            edges = [case["stair_start_x_m"]]
            for depth in case["tread_depths_m"][:-1]:
                edges.append(edges[-1] + depth)
            data["edges"].append(edges)
            data["heights"].append([0.0] + case["surface_heights_m"])
        env._training_stair_grid = {
            key: torch.tensor(value, dtype=torch.float32, device=env.device) for key, value in data.items()
        }
    index = terrain.terrain_levels * terrain.cfg.terrain_generator.num_cols + terrain.terrain_types
    return {key: value[index] for key, value in env._training_stair_grid.items()}


def training_stair_root_height(env, minimum_height=0.5):
    data = route_data(env)
    robot = env.scene["robot"].data
    x = robot.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    indices = (x[:, None] >= data["edges"]).sum(dim=-1)
    floor = data["heights"].gather(1, indices[:, None]).squeeze(1)
    return robot.root_pos_w[:, 2] - floor < minimum_height


def training_stair_off_course(env):
    data = route_data(env)
    xy = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return ((xy[:, 0] < data["lane_min_x_m"] + 0.1)
            | (xy[:, 0] > data["lane_max_x_m"] - 0.1)
            | (xy[:, 1].abs() > data["width_m"] / 2 - 0.15))


def training_stair_goal(env):
    data = route_data(env)
    robot = env.scene["robot"]
    xy = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    # Require both feet on the final floor, not just a torso crossing a boundary.
    if not hasattr(env, "_training_stair_feet"):
        env._training_stair_feet, _ = robot.find_bodies(".*_ankle_roll_link")
        env._training_stair_contacts, _ = env.scene["contact_forces"].find_bodies(".*_ankle_roll_link")
        if len(env._training_stair_feet) != 2 or len(env._training_stair_contacts) != 2:
            raise ValueError("Stair task requires both ankle links")
    feet = robot.data.body_pos_w[:, env._training_stair_feet] - env.scene.env_origins[:, None, :]
    forces = env.scene["contact_forces"].data.net_forces_w[:, env._training_stair_contacts].norm(dim=-1)
    return ((xy[:, 0] > data["goal_x_m"] - 0.30)
            & (xy[:, 1].abs() < 0.35)
            & (feet[:, :, 0] > data["stair_end_x_m"][:, None] + 0.1).all(dim=-1)
            & (feet[:, :, 2].abs() < 0.20).all(dim=-1)
            & (forces.max(dim=-1).values > 5.0)
            & (robot.data.projected_gravity_b[:, 2] < -math.cos(0.5)))


def stair_progress_curriculum(env, env_ids):
    """Promote only a completed, upright route; stationary survival cannot promote."""
    terrain = env.scene.terrain
    goal = env.termination_manager.get_term("stair_goal")[env_ids]
    failed = torch.zeros_like(goal)
    for name in ("root_height", "off_course", "bad_orientation", "base_contact", "nonfinite_state"):
        failed |= env.termination_manager.get_term(name)[env_ids]
    promote = goal & ~failed
    # Do not demote on the initialization reset (episode length zero).
    demote = (env.episode_length_buf[env_ids] > 0) & ~promote
    terrain.update_env_origins(env_ids, promote, demote)
    return terrain.terrain_levels.float().mean()


def moving_feet_air_time(env, command_name, sensor_cfg, min_air_time=0.20, max_reward=0.20):
    """Reward completed swings only, with a cap and an actual-motion gate."""
    sensor = env.scene[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    command = env.command_manager.get_command(command_name)
    moving = (command[:, :2].norm(dim=-1) > 0.15) & (env.scene["robot"].data.root_lin_vel_b[:, 0] > 0.10)
    return ((air_time - min_air_time).clamp(0.0, max_reward) * first_contact).sum(dim=-1) * moving


def velocity_deficit(env, command_name):
    """Continuous bounded penalty for stalling while commanded to move forward."""
    vx = env.command_manager.get_command(command_name)[:, 0]
    measured = env.scene["robot"].data.root_lin_vel_b[:, 0]
    return ((vx - measured).clamp(min=0.0) / vx.clamp(min=0.15)).clamp(max=2.0) * (vx > 0.15)


@configclass
class G1StairTrainingEnvCfg(G1ParkourEnvCfg):
    amp_reward_rate: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 25.0
        terrain = self.scene.terrain
        terrain.terrain_type = "hacked_generator"
        terrain.max_init_terrain_level = 0
        terrain.terrain_generator = TrainingStaircaseGeneratorCfg(
            seed=42, size=(12.0, 4.0), num_rows=8, num_cols=16,
            border_width=0.0, curriculum=True, use_cache=False, sub_terrains={},
        )
        command = self.commands.base_velocity
        command.class_type = TrainingStairCommand
        command.velocity_ranges = None
        command.random_velocity_terrain = None
        command.rel_standing_envs = 0.0
        command.resampling_time_range = (26.0, 26.0)
        command.ranges.lin_vel_x = (0.35, 0.60)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.8, 0.8)
        command.target_dis_threshold = 0.20
        self.curriculum.terrain_levels = CurriculumTermCfg(func=stair_progress_curriculum)
        # Existing reset_without_notice recycles expert clips without resetting
        # the robot. Keep it: disabling it would leave exhausted AMP references.
        self.terminations.dataset_exhausted.params["reset_without_notice"] = True
        self.terminations.terrain_out_bound = None
        self.terminations.root_height = TerminationTermCfg(func=training_stair_root_height)
        self.terminations.off_course = TerminationTermCfg(func=training_stair_off_course)
        self.terminations.nonfinite_state = TerminationTermCfg(func=nonfinite_state)
        self.terminations.stair_goal = TerminationTermCfg(func=training_stair_goal)
        r = self.rewards.rewards
        r.track_lin_vel_xy_exp.weight = 4.0
        r.track_lin_vel_xy_exp.params["std"] = 0.35
        r.track_ang_vel_z_exp.weight = 1.0
        r.heading_error = None
        r.is_alive.weight = 0.5
        r.stand_still.params["offset"] = 0.0
        r.dont_wait = RewardTermCfg(func=velocity_deficit, weight=-1.0, params={"command_name": "base_velocity"})
        r.feet_air_time.func = moving_feet_air_time
        r.feet_air_time.params.pop("vel_threshold")
        r.feet_air_time.weight = 1.0
        r.action_rate_l2.weight = -0.02
        r.dof_vel_l2.weight = -0.001
        r.ang_vel_xy_l2.weight = -0.1
        r.freeze_upper_body.weight = -0.08
        r.flat_orientation_l2.weight = -1.0
        r.pelvis_orientation_l2.weight = -1.0
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.05, 0.05)
        }
        self.events.reset_base.params["velocity_range"] = {}
        self.events.reset_robot_joints.params["position_range"] = (-0.05, 0.05)
        self.events.physics_material.params.update(
            static_friction_range=(0.8, 1.2), dynamic_friction_range=(0.8, 1.2), restitution_range=(0.0, 0.05)
        )
