"""Single-motion GRAIL stair bootstrap task with deterministic terrain alignment."""

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
    motion_start_from_middle_range=(0.0, 0.0),
    motion_start_height_offset=0.0,
    ensure_link_below_zero_ground=False,
    env_starting_stub_sampling_strategy="independent",
    buffer_device="output_device",
    motion_interpolate_func=motion_interpolate_bilinear,
    velocity_estimation_method="frontbackward",
    motion_bin_length_s=None,
)

GRAIL_BOOTSTRAP_MOTION_REFERENCE_CFG = motion_reference_cfg.replace(
    motion_buffers={"GrailStairBootstrap": GRAIL_BOOTSTRAP_MOTION_CFG},
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


@configclass
class G1GrailStairBootstrapEnvCfg(G1PlaneShadowingEnvCfg):
    """Deterministic learnability test before scaling to the full GRAIL stair set."""

    scene: shadowing_cfg.ShadowingSceneCfg = shadowing_cfg.ShadowingSceneCfg(
        num_envs=32,
        env_spacing=4.0,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=GRAIL_BOOTSTRAP_MOTION_REFERENCE_CFG,
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

    def __post_init__(self):
        super().__post_init__()

        # Bootstrap without robustness randomization. Add these terms back only after the aligned task is learnable.
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.push_robot = None
        self.events.bin_fail_counter_smoothing = None
        self.events.reset_robot.params["randomize_pose_range"] = {}
        self.events.reset_robot.params["randomize_velocity_range"] = {}
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)

        # No temporal-bin curriculum is needed when every episode begins at frame zero of one motion.
        self.curriculum.beyond_adaptive_sampling = None

        # Avoid the previous local optimum where reducing action changes dominated the imitation objective.
        self.rewards.rewards.action_rate_l2.weight = -0.01

        # Keep fall detection, but allow larger transient tracking errors while the bootstrap policy is learning.
        self.terminations.base_pos_too_far.params["distance_threshold"] = 0.5
        self.terminations.link_pos_too_far.params["distance_threshold"] = 0.5

        self.run_name = "G1Shadowing_GrailStairBootstrap_AlignedSingleMotion"


# Stage two begins two seconds into the ten-second clip, shortly before the first stair contact. The observation and
# action spaces remain identical to the frame-zero task, so its checkpoint can be reused.
GRAIL_MIDSTAIR_MOTION_CFG = GRAIL_BOOTSTRAP_MOTION_CFG.replace(
    motion_start_from_middle_range=(0.20, 0.20),
)

GRAIL_MIDSTAIR_MOTION_REFERENCE_CFG = motion_reference_cfg.replace(
    motion_buffers={"GrailStairMidStairBootstrap": GRAIL_MIDSTAIR_MOTION_CFG},
)


@configclass
class G1GrailStairMidStairBootstrapEnvCfg(G1GrailStairBootstrapEnvCfg):
    """Second bootstrap stage focused on stair contact and ascent."""

    scene: shadowing_cfg.ShadowingSceneCfg = G1GrailStairBootstrapEnvCfg.scene.replace(
        motion_reference=GRAIL_MIDSTAIR_MOTION_REFERENCE_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        # The frame-zero policy plateaued because strict tracking resets occurred before it experienced enough of the
        # staircase. Keep projected-gravity fall detection unchanged, but allow larger transient tracking errors.
        self.terminations.base_pos_too_far.params["distance_threshold"] = 0.75
        self.terminations.link_pos_too_far.params["distance_threshold"] = 0.75

        self.run_name = "G1Shadowing_GrailStairBootstrap_MidStair20pct"
