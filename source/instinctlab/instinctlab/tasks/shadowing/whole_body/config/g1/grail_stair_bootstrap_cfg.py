"""Single-motion GRAIL stair bootstrap task with deterministic terrain alignment."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import instinctlab.tasks.shadowing.whole_body.shadowing_env_cfg as shadowing_cfg
import instinctlab.terrains as terrain_gen
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.monitors import MonitorTerm, MonitorTermCfg

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


def _make_bootstrap_scene_cfg(motion_reference_cfg) -> shadowing_cfg.ShadowingSceneCfg:
    # ShadowingSceneCfg removes robot_reference in __post_init__ when debug visualization is disabled, so an
    # initialized scene cannot be cloned with dataclasses.replace. Construct each stage's scene independently.
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
    """Deterministic learnability test before scaling to the full GRAIL stair set."""

    scene: shadowing_cfg.ShadowingSceneCfg = _make_bootstrap_scene_cfg(GRAIL_BOOTSTRAP_MOTION_REFERENCE_CFG)

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


class GrailStairFailureTimeMonitor(MonitorTerm):
    """Log where mid-stair episodes end on the original motion timeline."""

    def __init__(self, cfg: MonitorTermCfg, env):
        super().__init__(cfg, env)
        self._last_log: dict[str, float] = {}

    def reset_idx(self, env_ids):
        # InstinctRlEnv invokes monitors before the scene and episode counters are reset.
        episode_steps = self._env.episode_length_buf[env_ids]
        completed = episode_steps > 0
        if not completed.any():
            self._last_log = {}
            return

        motion_reference = self._env.scene["motion_reference"]
        start_s = (
            motion_reference.complete_motion_lengths[env_ids]
            - motion_reference.assigned_motion_lengths[env_ids]
        )[completed]
        reference_time_s = start_s + episode_steps[completed] * self._env.step_dt

        self._last_log = {
            "episode_count": float(reference_time_s.numel()),
            "reference_start_s_mean": start_s.float().mean().item(),
            "reference_end_s_p10": torch.quantile(reference_time_s.float(), 0.10).item(),
            "reference_end_s_p50": torch.quantile(reference_time_s.float(), 0.50).item(),
            "reference_end_s_p90": torch.quantile(reference_time_s.float(), 0.90).item(),
        }
        for reason in ("base_pg_too_far", "link_pos_too_far", "dataset_exhausted"):
            triggered = self._env.termination_manager.get_term(reason)[env_ids][completed].bool()
            self._last_log[f"{reason}_fraction"] = triggered.float().mean().item()
            if triggered.any():
                self._last_log[f"{reason}_reference_end_s_p50"] = torch.quantile(
                    reference_time_s[triggered].float(), 0.50
                ).item()

    def get_log(self, is_episode=False) -> dict[str, float]:
        return self._last_log if is_episode else {}


@configclass
class GrailStairMidStairMonitorCfg(shadowing_cfg.MonitorCfg):
    failure_time: MonitorTermCfg = MonitorTermCfg(func=GrailStairFailureTimeMonitor)


@configclass
class G1GrailStairMidStairBootstrapEnvCfg(G1GrailStairBootstrapEnvCfg):
    """Second bootstrap stage focused on stair contact and ascent."""

    scene: shadowing_cfg.ShadowingSceneCfg = _make_bootstrap_scene_cfg(GRAIL_MIDSTAIR_MOTION_REFERENCE_CFG)
    monitors: GrailStairMidStairMonitorCfg = GrailStairMidStairMonitorCfg()

    def __post_init__(self):
        super().__post_init__()

        # The frame-zero policy plateaued because strict tracking resets occurred before it experienced enough of the
        # staircase. Keep projected-gravity fall detection unchanged, but allow larger transient tracking errors.
        self.terminations.base_pos_too_far.params["distance_threshold"] = 0.75
        self.terminations.link_pos_too_far.params["distance_threshold"] = 0.75

        self.run_name = "G1Shadowing_GrailStairBootstrap_MidStair20pct"
