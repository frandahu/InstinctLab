"""GRAIL-style curb stepping with the existing Mixed actor interface."""

import copy
import numpy as np
import os
import torch
import trimesh
from pathlib import Path

from isaaclab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from .curb_crossing_cases import curb_crossing_cases
from .curb_crossing_mdp import (
    curb_bridge_reward,
    curb_crossing_curriculum,
    curb_crossing_goal,
    curb_gap_ground_contact,
    reset_curb_crossing,
)
from .g1_parkour_target_amp_cfg import G1ParkourEnvCfg
from .stair_eval_cfg import StaircaseGenerator, StaircaseGeneratorCfg, StairGoalCommand, nonfinite_state
from .stair_training_cfg import training_stair_off_course, training_stair_root_height


class CurbCrossingGenerator:
    """Build the curriculum with the existing collision-slab implementation."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.grid_cases = curb_crossing_cases(cfg.seed, cfg.num_rows, cfg.num_cols, cfg.gap_range, cfg.max_step_m)
        meshes, origins, targets = [], [], []
        for row, cases in enumerate(self.grid_cases):
            row_cfg = copy.copy(cfg)
            row_cfg.cases = cases
            generated = StaircaseGenerator(row_cfg, device)
            offset = np.asarray((row * cfg.size[0], 0.0, 0.0), dtype=np.float32)
            if max(case["lane_max_x_m"] + 1.2 for case in cases) > cfg.size[0]:
                raise ValueError("Curb route does not fit terrain row spacing")
            generated.terrain_mesh.apply_translation(offset)
            meshes.append(generated.terrain_mesh)
            origins.append(generated.terrain_origins[0] + offset)
            targets.append(generated.flat_patches["target"][0] + torch.as_tensor(offset, device=device))
        self.terrain_mesh = trimesh.util.concatenate(meshes)
        self.terrain_origins = np.stack(origins)
        self.flat_patches = {"target": torch.stack(targets)}


@configclass
class CurbCrossingGeneratorCfg(StaircaseGeneratorCfg):
    class_type: type = CurbCrossingGenerator
    gap_range: tuple[float, float] = (0.18, 0.30)
    max_step_m: float = 0.60


class CurbCrossingCommand(StairGoalCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        for name in ("curb_clean_crossings", "curb_ground_touched_short_gaps", "curb_route_success"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)


@configclass
class G1CurbCrossingEnvCfg(G1ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        terrain = self.scene.terrain
        terrain.terrain_type = "hacked_generator"
        terrain.max_init_terrain_level = 0
        terrain.terrain_generator = CurbCrossingGeneratorCfg(
            seed=42,
            size=(14.0, 4.0),
            num_rows=8,
            num_cols=16,
            border_width=0.0,
            curriculum=True,
            use_cache=False,
            sub_terrains={},
        )
        self.scene.motion_reference = copy.deepcopy(self.scene.motion_reference)
        motion = self.scene.motion_reference.motion_buffers["run_walk"]
        root = Path(__file__).resolve().parents[7] / "data/grail_instinctlab"
        motion.path = os.environ.get("INSTINCTLAB_CURB_MOTION_ROOT", str(root))
        motion.filtered_motion_selection_filepath = os.environ.get("INSTINCTLAB_CURB_MOTION_SELECTION")
        motion.subset_selection = None
        motion.file_path_patterns = ["curb/*_retargeted.npz"]
        self.rewards.rewards.feet_air_time.weight = 0.75
        self.rewards.rewards.curb_bridge = RewardTermCfg(func=curb_bridge_reward, weight=2.0)
        self.rewards.rewards.curb_gap_ground = RewardTermCfg(func=curb_gap_ground_contact, weight=-4.0)
        command = self.commands.base_velocity
        command.class_type = CurbCrossingCommand
        command.velocity_ranges = None
        command.random_velocity_terrain = None
        command.rel_standing_envs = 0.0
        command.resampling_time_range = (31.0, 31.0)
        command.ranges.lin_vel_x = (0.45, 0.60)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.8, 0.8)
        command.target_dis_threshold = 0.20
        self.curriculum.terrain_levels = CurriculumTermCfg(func=curb_crossing_curriculum)
        self.events.curb_crossing_reset = EventTermCfg(func=reset_curb_crossing, mode="reset")
        self.terminations.terrain_out_bound = None
        self.terminations.root_height = TerminationTermCfg(func=training_stair_root_height)
        self.terminations.off_course = TerminationTermCfg(func=training_stair_off_course)
        self.terminations.nonfinite_state = TerminationTermCfg(func=nonfinite_state)
        self.terminations.curb_goal = TerminationTermCfg(func=curb_crossing_goal)
