"""Read Isaac Lab training YAML, including serialized scene-entity slices."""

from pathlib import Path


def restore_training_env_config(env_cfg, saved_env):
    """Restore the training-initialized seed before Isaac Lab's strict update."""
    # train.py sets env_cfg.seed = agent_cfg.seed before saving env.yaml. The
    # registry's fresh config still has seed=None, which older from_dict()
    # rejects when merging the saved integer seed. Match training initialization
    # first, then let Isaac Lab validate and restore every configuration field.
    if "seed" in saved_env:
        seed = saved_env["seed"]
        if seed is not None and type(seed) is not int:
            raise ValueError(f"Saved environment seed must be an integer or None, got {type(seed).__name__}")
        env_cfg.seed = seed
    env_cfg.from_dict(saved_env)


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
