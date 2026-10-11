"""Task-specific reset handling; leaves InstinctRlEnv and existing tasks intact."""

from instinctlab.envs import InstinctRlEnv


class VisualFootholdEnv(InstinctRlEnv):
    def _reset_idx(self, env_ids):
        adapter = getattr(self, "visual_foothold_state", None)
        metrics = adapter.core.episode_metrics(env_ids) if adapter is not None else {}
        result = super()._reset_idx(env_ids)
        if adapter is not None:
            adapter.reset(env_ids)
        self.extras.setdefault("log", {}).update({"VisualFoothold/" + k: v for k, v in metrics.items()})
        return result
