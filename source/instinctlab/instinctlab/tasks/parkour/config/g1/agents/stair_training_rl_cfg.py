"""Conservative PPO/AMP profile for a fresh depth-camera stair policy."""

from isaaclab.utils import configclass
from .instinct_rl_amp_cfg import AmpAlgoCfg, G1ParkourPPORunnerCfg, MoEPolicyCfg


@configclass
class StairPolicyCfg(MoEPolicyCfg):
    class_name = "instinctlab.utils.bounded_stair_policy:BoundedStairActorCritic"
    init_noise_std = 0.35
    min_noise_std = 0.05
    max_noise_std = 0.50


@configclass
class StairAmpAlgoCfg(AmpAlgoCfg):
    # Instinct-RL adds auxiliary reward per step, whereas task rewards are rates
    # integrated with dt. train.py recalculates this from the actual env step dt.
    discriminator_reward_coef = 0.25 * 0.02
    entropy_coef = 0.001
    learning_rate = 3.0e-4
    schedule = "fixed"
    clip_min_std = 0.05


@configclass
class G1StairPPORunnerCfg(G1ParkourPPORunnerCfg):
    experiment_name = "g1_stairs_v1"
    max_iterations = 10000
    num_steps_per_env = 32
    save_interval = 500
    log_interval = 50
    resume = False
    policy = StairPolicyCfg()
    algorithm = StairAmpAlgoCfg()
