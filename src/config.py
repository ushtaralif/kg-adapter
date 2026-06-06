
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_path(cfg: dict, key: str) -> Path:

    val = cfg["paths"][key]
    if val == "<FILL_IN>":
        raise ValueError(
            f"\n[config] paths.{key} is not set.\n"
            f"Edit configs/config.yaml and replace <FILL_IN> with the actual path."
        )
    return Path(val)

def get_active_model_cfg(cfg: dict) -> dict:
    active = cfg["models"]["active"]
    m = cfg["models"][active]
    if m["local_path"] == "<FILL_IN>":
        raise ValueError(
            f"\n[config] models.{active}.local_path is not set.\n"
            f"Edit configs/config.yaml on server-gpu."
        )
    return m