"""Read Isaac Lab training YAML, including serialized scene-entity slices."""

from pathlib import Path


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
