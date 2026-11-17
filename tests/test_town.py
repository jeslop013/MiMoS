"""The synthetic example town (mimos.scenarios.town): generated from a seed, it runs through the
whole model and every agent finishes. No external data needed."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from mimos import load_config, run
from mimos.scenarios import town


def _digest(folder: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(folder.rglob("*.csv")):
        h.update(f.relative_to(folder).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def test_town_is_reproducible(tmp_path):
    town(tmp_path / "a", n_agents=200, seed=3)
    town(tmp_path / "b", n_agents=200, seed=3)
    town(tmp_path / "c", n_agents=200, seed=4)
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")
    assert _digest(tmp_path / "a") != _digest(tmp_path / "c")


def test_town_runs(tmp_path):
    config = town(tmp_path / "town", n_agents=800, iterations=2)
    model, report = run(load_config(config), verbose=False, progress=False)
    assert all(rep.invariants_ok for rep in report.iterations)
    for rep in report.iterations:
        assert rep.drive.finished == rep.drive.agents > 0
        assert rep.public.finished == rep.public.agents > 0
    rows = list(csv.DictReader((tmp_path / "town" / "run" / "iterations.csv").open()))
    last = rows[-1]
    assert int(last["unfinished_agents"]) == 0 and int(last["gridlocked"]) == 0
    for mode in ("walk", "car", "pt"):  # every main mode is used
        assert float(last[f"trip_{mode}"]) > 0.02
