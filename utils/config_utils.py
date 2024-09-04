# utils/config_utils.py

import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

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
        @hydra.main(version_base=None, config_path=None)
        def wrapper(_: DictConfig) -> None:
            # Disable Hydra's output directory creation
            os.environ['HYDRA_FULL_ERROR'] = '1'
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            hydra.initialize(config_path=None, job_name="app")
            
            cfg = load_and_merge_config(config_name)
            return func(cfg)
        return wrapper
    return decorator

# Register your config schema if you have one
cs = ConfigStore.instance()
cs.store(name="base_config", node=DictConfig({}))