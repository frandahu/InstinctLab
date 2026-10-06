"""Original MoE/WasabiPPO hyperparameters; independent experiment directory."""

from isaaclab.utils import configclass

from .instinct_rl_amp_cfg import G1ParkourPPORunnerCfg


@configclass
class G1MixedParkourPPORunnerCfg(G1ParkourPPORunnerCfg):
    experiment_name = "g1_parkour_mixed"
