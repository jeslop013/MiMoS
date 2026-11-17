"""Configuration loading: defaults, strictness, path handling."""

from pathlib import Path

import pytest

from mimos import Config, LogLevel, config_from_dict, load_config

DATA = Path(__file__).resolve().parent / "fixtures"


def test_defaults_toml_equals_cpp_defaults():
    """defaults.toml (the parameter values of the thesis) must equal the C++ defaults exactly."""
    defaults = Config()
    cfg = load_config(DATA / "defaults.toml")
    skip = {"data_dir", "run_dir", "log_level"}
    fields = [k for k in dir(defaults) if not k.startswith("_") and k not in skip]
    assert fields, "no Config fields exposed"
    different = {k: (getattr(defaults, k), getattr(cfg, k)) for k in fields if getattr(defaults, k) != getattr(cfg, k)}
    assert different == {}


def test_unknown_key_is_an_error():
    with pytest.raises(KeyError, match="beta_trvl_drive"):
        config_from_dict({"utility": {"beta_trvl_drive": -6.0}})


def test_paths_and_log_level(tmp_path):
    cfg = config_from_dict({"run": {"data_dir": "data", "run_dir": "out", "log_level": "debug"}}, base_dir=tmp_path)
    assert cfg.data_dir == str((tmp_path / "data").resolve()) + "/"
    assert cfg.run_dir.endswith("out/")
    assert cfg.log_level == LogLevel.DEBUG


def test_overrides_are_relative_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(DATA / "defaults.toml", n_agents=7, run_dir="here")
    assert cfg.n_agents == 7
    assert cfg.run_dir == str((tmp_path / "here").resolve()) + "/"
