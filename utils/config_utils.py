from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import sys

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent

def load_and_merge_config(config_name: str = "config") -> DictConfig:
    project_root = get_project_root()
    base_config_path = project_root / f"{config_name}.yaml"
    local_config_path = project_root / f"{config_name}.local.yaml"

    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config file not found at {base_config_path}")

    cfg = OmegaConf.load(base_config_path)

    if local_config_path.exists():
        local_cfg = OmegaConf.load(local_config_path)
        cfg = OmegaConf.merge(cfg, local_cfg)

    return cfg

def config_wrapper(config_name: str = "config"):
    def decorator(func):
        def wrapper():
            cfg = load_and_merge_config(config_name)

            # Handle command-line arguments
            cli_config = OmegaConf.from_dotlist(sys.argv[1:])
            cfg = OmegaConf.merge(cfg, cli_config)

            return func(cfg)
        return wrapper
    return decorator