"""Task-local wall policy and reproducible terrain metadata; no simulator imports."""

from collections.abc import Mapping
from copy import deepcopy


def _get(cfg, name, default=None):
    return cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)


def _wall_configs(generator):
    """Yield named sub-terrain configs and their optional Perlin noise configs."""
    for name, cfg in _get(generator, "sub_terrains", {}).items():
        yield name, cfg
        noise = _get(cfg, "perlin_cfg")
        if noise is not None:
            yield name + "/perlin_cfg", noise


def configure_terrain_walls(generator, mode="saved"):
    """Copy a generator, preserving its walls unless explicitly disabled.

    Call AFTER saved-config restoration for playback. Changing only the fresh
    task defaults cannot override wall probabilities in an old env.yaml. The
    shared upstream generator and its terrain geometry must stay untouched.
    """
    if mode not in ("off", "saved"):
        raise ValueError("Terrain wall mode must be 'off' or 'saved'")
    if generator is None:
        raise ValueError("Visual foothold requires a terrain generator")
    result = deepcopy(generator)
    if mode == "off":
        for _, cfg in _wall_configs(result):
            if _get(cfg, "wall_prob") is not None:
                if isinstance(cfg, dict):
                    cfg["wall_prob"] = [0.0] * 4
                else:
                    cfg.wall_prob = [0.0] * 4
    return result


def terrain_wall_summary(generator):
    """Describe actual settings, accepting live config objects or saved mappings."""
    if generator is None:
        return None
    walls = {}
    enabled = []
    for name, cfg in _wall_configs(generator):
        probabilities = _get(cfg, "wall_prob")
        if probabilities is None:
            continue
        walls[name] = {
            "probabilities": list(probabilities),
            "height_m": _get(cfg, "wall_height"),
            "thickness_m": _get(cfg, "wall_thickness"),
        }
        if any(p > 0 for p in probabilities):
            enabled.append(name)
    size = _get(generator, "size")
    return {
        "size_m": list(size) if size is not None else None,
        "num_rows": _get(generator, "num_rows"),
        "num_cols": _get(generator, "num_cols"),
        "border_width_m": _get(generator, "border_width"),
        "border_height_m": _get(generator, "border_height"),
        "random_walls_enabled": bool(enabled),
        "wall_enabled_configs": enabled,
        "wall_configs": walls,
    }
