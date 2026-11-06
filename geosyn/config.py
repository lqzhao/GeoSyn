"""YAML configuration with paths relative to the configuration file."""
from pathlib import Path
import yaml


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if config.get("robot") not in ("panda", "kuka", "baxter"):
        raise ValueError("Configuration must specify panda, kuka or baxter")
    for key in ("data_root", "observations_root", "checkpoint", "dino_weight", "initialize_from"):
        if config.get(key):
            value = Path(config[key]).expanduser()
            config[key] = str(
                value.resolve() if value.is_absolute() else (path.parent / value).resolve()
            )
    config["config_path"] = str(path)
    return config


def add_data_arguments(parser):
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, help="Override the DREAM robot data directory")
    parser.add_argument(
        "--observations-root", type=Path, help="Override frozen RGB observation caches"
    )


def apply_overrides(config, args):
    for key in ("data_root", "observations_root"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = str(value.expanduser().resolve())
    return config
