"""
Configuration management for the Thermal Pattern Analysis project.

Loads YAML configs, provides attribute-style access, and handles
device selection + reproducibility seeding.
"""

import os
import yaml
import torch
import random
import numpy as np
from pathlib import Path


class Config:
    """Hierarchical configuration with attribute-style access."""

    def __init__(self, config_dict: dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, list):
                setattr(self, key, [
                    Config(v) if isinstance(v, dict) else v for v in value
                ])
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict:
        """Convert back to a plain dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    v.to_dict() if isinstance(v, Config) else v for v in value
                ]
            else:
                result[key] = value
        return result

    def __repr__(self):
        return f"Config({self.to_dict()})"

    def get(self, key, default=None):
        """Safe attribute access with a default value."""
        return getattr(self, key, default)


def load_config(config_path: str = "configs/config.yaml") -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config object with attribute-style access.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def setup_device(config: Config) -> torch.device:
    """
    Determine the compute device based on config and availability.

    Auto mode picks CUDA if available, otherwise CPU.
    """
    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    return device


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dirs(config: Config):
    """Create all output directories specified in config.paths."""
    paths = config.get("paths", None)
    if paths is None:
        return
    for attr in ["checkpoints", "logs", "results", "visualizations"]:
        dir_path = paths.get(attr, None)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
