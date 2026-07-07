"""Shared YAML config loading with dotted-key --override support."""
from __future__ import annotations

import ast
import platform

import torch
import yaml


def get_default_device() -> str:
    """Pick a sensible default device for the current OS.

    macOS -> mps, Windows -> cuda, otherwise -> cpu.
    Falls back to cpu if the preferred backend isn't actually available.
    """
    system = platform.system()
    if system == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if system == "Windows" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
