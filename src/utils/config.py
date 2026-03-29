"""
src/utils/config.py

Config loader for the ProteinStructure ESP pipeline.

Reads config.yaml from the project root and provides a single
get_config() function used by all modules. The config is loaded
once and cached for the lifetime of the process.

Usage:
    from src.utils.config import get_config
    cfg = get_config()

    data_root  = Path(cfg["paths"]["data_root"])
    pdb2pqr    = cfg["executables"]["pdb2pqr"]
    ph_value   = cfg["electrostatics"]["ph_value"]

The config file is resolved relative to this file's location:
    src/utils/config.py  →  ../../config.yaml  (project root)

Raises:
    FileNotFoundError: if config.yaml does not exist
    KeyError:          if a required section or key is missing
"""

import logging
from functools import lru_cache
from pathlib import Path

import yaml

# Project root is two levels up from src/utils/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH  = _PROJECT_ROOT / "config.yaml"

# Required top-level sections
_REQUIRED_SECTIONS = ["paths", "executables", "electrostatics", "surface", "esp_mapping", "logging"]


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Load and return the project config as a nested dict.
    Result is cached — the file is only read once per process.

    Returns:
        dict with keys: paths, executables, electrostatics,
                        surface, esp_mapping, logging

    Raises:
        FileNotFoundError: if config.yaml is missing
        ValueError:        if required sections are absent
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {_CONFIG_PATH}\n"
            f"Copy config.template.yaml to config.yaml and fill in your paths:\n"
            f"  cp {_PROJECT_ROOT}/config.template.yaml {_CONFIG_PATH}"
        )

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Validate required sections exist
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(
            f"config.yaml is missing required sections: {missing}\n"
            f"Check config.template.yaml for the expected structure."
        )

    # Apply logging level from config
    level = cfg["logging"].get("level", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO))

    return cfg


def get_data_root() -> Path:
    """Convenience: return the external data root as a Path."""
    return Path(get_config()["paths"]["data_root"])


def get_log_file() -> Path:
    """Convenience: return the log file path as a Path."""
    return Path(get_config()["paths"]["log_file"])