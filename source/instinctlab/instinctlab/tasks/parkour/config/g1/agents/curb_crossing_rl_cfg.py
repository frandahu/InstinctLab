"""Keep Mixed MoE/WasabiPPO settings, with a separate fine-tuning log directory."""

from isaaclab.utils import configclass

from .instinct_rl_amp_cfg import G1ParkourPPORunnerCfg


@configclass
class G1CurbCrossingPPORunnerCfg(G1ParkourPPORunnerCfg):
    experiment_name = "g1_curb_crossing"
