"""Shared YAML config loading with dotted-key --override support."""
from __future__ import annotations

import ast

import yaml


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, value = item.partition("=")
        _set_dotted(cfg, key.strip(), _parse_value(value.strip()))
    return cfg


def _parse_value(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _set_dotted(cfg: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
