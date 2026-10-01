"""Read Isaac Lab training YAML, including serialized scene-entity slices."""

from pathlib import Path
from collections.abc import Mapping
from copy import deepcopy


def _plain_config_value(value):
    if value is None or type(value) in (bool, int, float, str, bytes, slice):
        return True
    return isinstance(value, (list, tuple)) and all(_plain_config_value(item) for item in value)


def _initialize_optional_values(target, saved, initialized, path=""):
    """Prime existing None leaves; keep config objects and unknown-key checks."""
    if isinstance(saved, Mapping):
        for key, value in saved.items():
            if isinstance(target, dict):
                if key not in target:
                    continue  # from_dict must still reject unknown schema keys
                current = target[key]
            else:
                if not hasattr(target, key):
                    continue
                current = getattr(target, key)
            field_path = f"{path}/{key}"
            if current is None and value is not None and _plain_config_value(value):
                if isinstance(target, dict):
                    target[key] = deepcopy(value)
                else:
                    setattr(target, key, deepcopy(value))
                initialized.append(field_path)
            elif current is not None:
                _initialize_optional_values(current, value, initialized, field_path)
    elif isinstance(saved, (list, tuple)) and isinstance(target, (list, tuple)) and len(saved) == len(target):
        for index, (current, value) in enumerate(zip(target, saved)):
            _initialize_optional_values(current, value, initialized, f"{path}/{index}")


def restore_training_env_config(env_cfg, saved_env, skip_paths=()):
    """Restore optional values before strict update, excluding explicit overrides.

    Training initialization fills None defaults (seeds, terrain scales, etc.).
    Isaac Lab's older updater rejects these serialized scalar values against a
    fresh None default. Initialize such leaves recursively, then delegate the
    remaining restoration and schema/type checks to the original from_dict().
    """
    restored = deepcopy(saved_env) if skip_paths else saved_env
    skipped = []
    for path in skip_paths:
        parts = path.strip("/").split("/")
        parent = restored
        for key in parts[:-1]:
            if not isinstance(parent, Mapping) or key not in parent:
                parent = None
                break
            parent = parent[key]
        if isinstance(parent, dict) and parts[-1] in parent:
            del parent[parts[-1]]
            skipped.append("/" + "/".join(parts))
    if "seed" in saved_env:
        seed = saved_env["seed"]
        if seed is not None and type(seed) is not int:
            raise ValueError(f"Saved environment seed must be an integer or None, got {type(seed).__name__}")
    initialized = []
    _initialize_optional_values(env_cfg, restored, initialized)
    env_cfg.from_dict(restored)
    return {"initialized_optional_fields": initialized, "skipped_saved_fields": skipped}


def load_training_yaml(path):
    """Restore saved config values without changing PyYAML's global loaders."""
    import yaml

    class TrainingConfigLoader(yaml.FullLoader):
        # Older PyYAML FullLoader versions accept arbitrary object construction.
        # Keep name references and add only the observed builtins.slice tag.
        yaml_multi_constructors = {
            tag: constructor for tag, constructor in yaml.FullLoader.yaml_multi_constructors.items()
            if tag == "tag:yaml.org,2002:python/name:"
        }

    def construct_slice(loader, node):
        args = loader.construct_sequence(node, deep=True)
        if not 1 <= len(args) <= 3 or any(value is not None and type(value) is not int for value in args):
            raise yaml.constructor.ConstructorError(
                None, None, "builtins.slice requires one to three integer/null arguments", node.start_mark
            )
        if len(args) == 3 and args[2] == 0:
            raise yaml.constructor.ConstructorError(None, None, "slice step cannot be zero", node.start_mark)
        return slice(*args)

    TrainingConfigLoader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:builtins.slice", construct_slice
    )
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=TrainingConfigLoader)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a saved training configuration mapping in {path}, got {type(config).__name__}")
    return config
