"""Full-clip GRAIL stair training on one terrain-aligned motion."""

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import instinctlab.tasks.shadowing.whole_body.shadowing_env_cfg as shadowing_cfg
import instinctlab.terrains as terrain_gen
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg
from instinctlab.motion_reference.utils import motion_interpolate_bilinear

from .grail_stair_shadowing_cfg import GRAIL_MOTION_ROOT
from .plane_shadowing_cfg import G1_CFG, G1PlaneShadowingEnvCfg, motion_reference_cfg

# This clip is a clean, nearly straight ascent: 3.164 m forward and 0.756 m upward over ten seconds. The offset
# places its final pose on the center platform and its initial pose on the positive-Y ground approach.
GRAIL_BOOTSTRAP_MOTION_CFG = AmassMotionCfg(
    path=GRAIL_MOTION_ROOT,
    file_path_patterns=["stair_p2/terrain_stairs__stairs_3630__0000_retargeted.npz"],
    root_position_offset=(0.00918864, 2.47129717, -0.72000000),
    retargetting_func=None,
    filtered_motion_selection_filepath=None,
    # The bin sampler below overrides this field and covers the clip from its first second onward.
    motion_start_from_middle_range=(0.0, 0.0),
    motion_start_height_offset=0.0,
    ensure_link_below_zero_ground=False,
    env_starting_stub_sampling_strategy="independent",
    buffer_device="output_device",
    motion_interpolate_func=motion_interpolate_bilinear,
    velocity_estimation_method="frontbackward",
    # InstinctLab samples across the entire clip through its adaptive motion-bin curriculum.
    motion_bin_length_s=1.0,
)

GRAIL_BOOTSTRAP_MOTION_REFERENCE_CFG = motion_reference_cfg.replace(
    motion_buffers={"GrailStairBootstrap": GRAIL_BOOTSTRAP_MOTION_CFG},
)

GRAIL_BOOTSTRAP_PLAY_MOTION_REFERENCE_CFG = motion_reference_cfg.replace(
    motion_buffers={
        "GrailStairBootstrap": GRAIL_BOOTSTRAP_MOTION_CFG.replace(motion_bin_length_s=None),
    },
)

# These fixed stair dimensions were fitted to the selected clip's root-height-versus-forward-position trace. The
# resulting reference base clearance has about 2.7 cm RMS variation along the flight, versus 12.5 cm for the first
# generic stair geometry.
GRAIL_BOOTSTRAP_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "aligned_stairs": terrain_gen.PerlinStairsUpDownTerrainCfg(
            proportion=1.0,
            per_step_height=0.06,
            per_step_width=2.0,
            per_step_length=0.20,
            num_steps=12,
            platform_length=0.25,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            perlin_cfg=None,
        ),
    },
)


def _make_bootstrap_scene_cfg(motion_reference_cfg) -> shadowing_cfg.ShadowingSceneCfg:
    return shadowing_cfg.ShadowingSceneCfg(
        num_envs=32,
        env_spacing=4.0,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
        terrain=TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=GRAIL_BOOTSTRAP_TERRAIN_CFG,
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


@configclass
class G1GrailStairBootstrapEnvCfg(G1PlaneShadowingEnvCfg):
    """Train from scratch on the full aligned stair clip using InstinctLab's standard shadowing setup."""

    scene: shadowing_cfg.ShadowingSceneCfg = _make_bootstrap_scene_cfg(GRAIL_BOOTSTRAP_MOTION_REFERENCE_CFG)

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "G1Shadowing_GrailStairs_FullClip_Aligned"


@configclass
class G1GrailStairBootstrapEnvCfg_PLAY(G1GrailStairBootstrapEnvCfg):
    """Evaluate a checkpoint from the first frame of the same aligned motion."""

    scene: shadowing_cfg.ShadowingSceneCfg = _make_bootstrap_scene_cfg(GRAIL_BOOTSTRAP_PLAY_MOTION_REFERENCE_CFG)

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.push_robot = None
        self.events.reset_robot.params["randomize_pose_range"] = {}
        self.events.reset_robot.params["randomize_velocity_range"] = {}
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)
