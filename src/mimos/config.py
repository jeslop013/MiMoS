"""Run configuration: TOML file -> :class:`mimos._core.Config`.

The TOML file groups parameters into tables (``[run]``, ``[utility]``, ...) for readability.
The grouping is cosmetic: every key must be the name of a field of the C++ ``Config``
struct (``cpp/include/mimos/config.hpp``), whose defaults are the parameter values of the thesis.
Unknown keys raise an error, so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from mimos._core import Config, LogLevel

_PATH_KEYS = {"data_dir", "run_dir"}
# Old parameter names still accepted (saved run config.toml files, older configs).
_ALIASES = {"results_dir": "run_dir"}
# [view] table: where maps and videos look (not a model parameter; see load_view).
VIEW_KEYS = {"center_lat", "center_lon", "radius_km"}


def _as_dir(value: str, base: Path) -> str:
    """Directory paths are relative to the config file. The C++ core concatenates file names
    directly, so the path must end with a separator."""
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path) + ("" if str(path).endswith(("/", "\\")) else "/")


def config_from_dict(values: dict[str, Any], base_dir: Path | str = ".") -> Config:
    """Builds a Config from a (possibly nested) dict of parameter values."""
    cfg = Config()
    base = Path(base_dir)
    flat: dict[str, Any] = {}
    for key, value in values.items():
        if key == "view":  # map/video view, read by load_view, not a model parameter
            continue
        if isinstance(value, dict):  # TOML table: only for grouping
            flat.update(value)
        else:
            flat[key] = value

    for key, value in flat.items():
        key = _ALIASES.get(key, key)
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown MiMoS parameter '{key}' (see cpp/include/mimos/config.hpp)")
        if key in _PATH_KEYS:
            value = _as_dir(value, base)
        elif key == "log_level":
            value = LogLevel.__members__[str(value).upper()]
        setattr(cfg, key, value)
    return cfg


def load_config(path: Path | str, **overrides: Any) -> Config:
    """Reads a TOML configuration. Keyword overrides (e.g. ``n_agents=1000``) win over the file.

    Paths inside the file are relative to the file; paths given as overrides (e.g. from the
    command line) are relative to the current working directory.
    """
    path = Path(path)
    with path.open("rb") as f:
        values = tomllib.load(f)
    cfg = config_from_dict(values, base_dir=path.parent)
    given = {_ALIASES.get(k, k): v for k, v in overrides.items() if v is not None}
    parsed = config_from_dict(given, base_dir=Path.cwd())  # validates and converts the overrides
    for key in given:
        setattr(cfg, key, getattr(parsed, key))
    return cfg


def load_view(path: Path | str | None) -> dict | None:
    """The [view] table of a TOML file: centre (lat, lon) and radius (km) of maps and videos.

    Example: center_lat = 0.0, center_lon = 0.0, radius_km = 3.7 (the example town). None if the file has no [view] (maps then centre on the traffic).
    """
    if path is None or not Path(path).exists():
        return None
    with Path(path).open("rb") as f:
        view = tomllib.load(f).get("view")
    if not view:
        return None
    unknown = set(view) - VIEW_KEYS
    if unknown:
        raise KeyError(f"Unknown [view] keys {sorted(unknown)}; allowed: {sorted(VIEW_KEYS)}")
    return {k: float(v) for k, v in view.items()}


def save_config(cfg: Config, path: Path | str, view: dict | None = None) -> None:
    """Writes every Config field to a TOML file that ``load_config`` reads back (paths are
    absolute), so a results folder records exactly how it was produced. ``view`` adds the
    [view] table used by the report and the videos."""
    lines = ["# Resolved MiMoS configuration of this run (written by mimos.config.save_config)", "[config]"]
    for key in sorted(k for k in dir(cfg) if not k.startswith("_")):
        value = getattr(cfg, key)
        if callable(value):
            continue
        if isinstance(value, LogLevel):
            value = value.name
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value!r}")
        else:
            lines.append(f"{key} = {str(value)!r}".replace("\\\\", "/"))
    if view:
        lines += ["", "[view]"] + [f"{k} = {v!r}" for k, v in view.items()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
