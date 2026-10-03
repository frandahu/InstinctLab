"""Same depth MoE architecture with bounded exploration for stair learning."""

import builtins
import math

import torch
from instinct_rl.modules import EncoderMoEActorCritic


class BoundedStairActorCritic(EncoderMoEActorCritic):
    def __init__(self, *args, min_noise_std=0.05, max_noise_std=0.50, **kwargs):
        if not (math.isfinite(min_noise_std) and math.isfinite(max_noise_std)
                and 0 < min_noise_std <= max_noise_std):
            raise ValueError("Require finite 0 < min_noise_std <= max_noise_std")
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        super().__init__(*args, **kwargs)
        self.clip_std()

    @torch.no_grad()
    def clip_std(self, min=None, max=None):
        low = self.min_noise_std if min is None else builtins.max(min, self.min_noise_std)
        high = self.max_noise_std if max is None else builtins.min(max, self.max_noise_std)
        if low > high:
            raise ValueError("PPO std limits conflict with stair policy bounds")
        if not torch.isfinite(self.std).all():
            raise FloatingPointError("Non-finite stair policy exploration std")
        self.std.clamp_(min=low, max=high)

    def update_distribution(self, observations):
        # Also bound it before minibatch evaluations, including restored policies.
        self.clip_std()
        return super().update_distribution(observations)
