"""MiMoS: Microscopic Mobility Simulator, an agent-based model of urban mobility.

Quick start::

    from mimos import load_config, run
    cfg = load_config("examples/town/config.toml", n_agents=1000)   # from: uv run mimos example
    model, report = run(cfg)
"""

from mimos._core import Config, DriveStats, LogLevel, Model, PublicStats
from mimos.config import config_from_dict, load_config
from mimos.model import run

__all__ = [
    "Config", "DriveStats", "LogLevel", "Model", "PublicStats",
    "config_from_dict", "load_config", "run",
]
