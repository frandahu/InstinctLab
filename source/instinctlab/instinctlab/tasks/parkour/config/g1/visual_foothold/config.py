"""Visual foothold training with author + GRAIL experts, and an optional baseline."""

import copy

from isaaclab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from ..mixed_parkour_cfg import G1MixedParkourEnvCfg
from ..agents.mixed_parkour_rl_cfg import G1MixedParkourPPORunnerCfg
from . import mdp
from .motions import grail_motion_reference
from .perception import VisualFootholdSettings
from .reference import ReferenceExperimentSettings


@configclass
class VisualBaselineEnvCfg(G1MixedParkourEnvCfg):
    """Optional baseline; the training entry point combines author + GRAIL experts."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.motion_reference = grail_motion_reference(self.scene.motion_reference)
        self.scene.terrain.terrain_generator = copy.deepcopy(self.scene.terrain.terrain_generator)
        for group in (self.observations.policy, self.observations.critic):
            group.depth_image.params["delayed_frame_ranges"] = (1, 1)


@configclass
class VisualFootholdEnvCfg(VisualBaselineEnvCfg):
    """B: same joint control and AMP, with depth-derived foothold commands."""

    visual_foothold: VisualFootholdSettings = VisualFootholdSettings()
    visual_reference: ReferenceExperimentSettings = ReferenceExperimentSettings()

    def __post_init__(self):
        super().__post_init__()
        for group in (self.observations.policy, self.observations.critic):
            group.visual_foothold = ObservationTermCfg(func=mdp.visual_observation, noise=None)
        self.terminations.visual_capture = TerminationTermCfg(func=mdp.capture_before_reset)
        rewards = self.rewards.rewards
        # Event terms divide by step_dt in the controller; manager integration
        # cancels that dt. Swing is a continuous reward rate. These are initial
        # experiment weights, not empirically tuned or proven-stable values.
        rewards.visual_touchdown = RewardTermCfg(func=mdp.visual_reward, weight=0.5, params={"term": "touchdown"})
        rewards.visual_swing = RewardTermCfg(func=mdp.visual_reward, weight=2.0, params={"term": "swing"})
        rewards.visual_edge = RewardTermCfg(func=mdp.visual_reward, weight=-0.5, params={"term": "edge"})
        rewards.visual_miss = RewardTermCfg(func=mdp.visual_reward, weight=-0.25, params={"term": "miss"})


@configclass
class VisualBaselineRunnerCfg(G1MixedParkourPPORunnerCfg):
    experiment_name = "g1_parkour_visual_baseline"
    resume = False


@configclass
class VisualFootholdRunnerCfg(G1MixedParkourPPORunnerCfg):
    experiment_name = "g1_parkour_visual_foothold"
    resume = False
