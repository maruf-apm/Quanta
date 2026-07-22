"""
YAML configuration loader with hierarchical merging.
"""
import os
import yaml
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    """Load a single YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two config dicts (override takes precedence)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def build_config(test_config_path: str,
                 model_config_path: str = None,
                 training_config_path: str = None) -> Dict[str, Any]:
    """
    Build final config from test-specific + model-default + training-default files.
    """
    test_cfg = load_config(test_config_path)
    cfg = test_cfg.copy()

    if model_config_path and os.path.exists(model_config_path):
        model_cfg = load_config(model_config_path)
        cfg = merge_configs(cfg, model_cfg)

    if training_config_path and os.path.exists(training_config_path):
        train_cfg = load_config(training_config_path)
        cfg = merge_configs(cfg, train_cfg)

    return cfg
