"""Minimal GRAIL stair-motion shadowing task on procedural stairs.

This configuration is intentionally a smoke-test task. It keeps the existing whole-body shadowing observations,
rewards, and PPO setup while changing only the reference-motion subset and the terrain. The initial terrain levels
are limited to the easiest two rows so motion/terrain mismatch can be diagnosed before increasing difficulty.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import instinctlab.tasks.shadowing.whole_body.shadowing_env_cfg as shadowing_cfg
import instinctlab.terrains as terrain_gen
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg
from instinctlab.motion_reference.utils import motion_interpolate_bilinear

from .plane_shadowing_cfg import G1_CFG, G1PlaneShadowingEnvCfg, motion_reference_cfg

GRAIL_MOTION_ROOT = os.environ.get(
    "INSTINCTLAB_GRAIL_MOTION_ROOT",
    "/workspace/instinctlab/data/grail_instinctlab",
)

GRAIL_STAIR_MOTION_CFG = AmassMotionCfg(
    path=GRAIL_MOTION_ROOT,
    file_path_patterns=[
        "stair_p1/*_retargeted.npz",
        "stair_p2/*_retargeted.npz",
    ],
    retargetting_func=None,
    filtered_motion_selection_filepath=None,
    motion_start_from_middle_range=[0.0, 0.8],
    motion_start_height_offset=0.0,
    ensure_link_below_zero_ground=False,
    env_starting_stub_sampling_strategy="independent",
    buffer_device="output_device",
    motion_interpolate_func=motion_interpolate_bilinear,
    velocity_estimation_method="frontbackward",
    motion_bin_length_s=1.0,
)

GRAIL_STAIR_MOTION_REFERENCE_CFG = motion_reference_cfg.replace(
    motion_buffers={"GrailStairs": GRAIL_STAIR_MOTION_CFG},
)

# Start with clean, low stairs. Terrain noise and a wider height range belong in later curriculum stages, after the
# reference/terrain alignment and reset behavior have been checked in the smoke test.
GRAIL_STAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=5,
    num_cols=1,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.PerlinPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.04, 0.12),
            step_width=0.30,
            platform_width=1.5,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            perlin_cfg=None,
        ),
    },
)


@configclass
class G1GrailStairShadowingEnvCfg(G1PlaneShadowingEnvCfg):
    """G1 whole-body shadowing with the 500 GRAIL stair references."""

    scene: shadowing_cfg.ShadowingSceneCfg = shadowing_cfg.ShadowingSceneCfg(
        num_envs=32,
        env_spacing=4.0,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=GRAIL_STAIR_MOTION_REFERENCE_CFG,
        terrain=TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=GRAIL_STAIR_TERRAINS_CFG,
            max_init_terrain_level=1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=(
                    f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                    "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
                ),
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "G1Shadowing_GrailStairs_ProceduralStairs_Debug"
