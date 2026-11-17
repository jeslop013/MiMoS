"""The MiMoS iterative cycle, driven from Python.

Equivalent to ``Model.run()`` in C++, but each stage is a separate
call. You can inspect or change state between stages, add logging, or stop early, without
recompiling. The stages and their order (thesis "MiMoS outline and iterative cycle"):

    load -> plan_bike -> for each iteration:
        start_iteration -> plan -> choose -> prepare_execution
        -> simulate_public -> simulate_drive -> update_memory -> check_invariants -> write
    -> write_final

Besides the CSV outputs of the core, every run writes: ``config.toml`` (resolved
configuration), ``iterations.csv`` (one row per iteration: planned vs executed scores, mode
switches, shares, traffic; see ``mimos.analysis.iteration_summary``) and ``trips.csv`` (every
trip of the last iteration), ``drive_links.npz`` (every road link traversed in the last
iteration, full precision) and ``run_info.json``. ``mimos report`` turns them into an HTML
report.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from mimos._core import Config, Model
import numpy as np

from mimos.analysis import all_planned_scores, append_row, executed_mode, iteration_summary, write_trips
from mimos.config import save_config
from mimos.progress import Progress


@dataclass
class IterationReport:
    iteration: int
    seconds: dict[str, float] = field(default_factory=dict)
    public: object = None   # mimos._core.PublicStats
    drive: object = None    # mimos._core.DriveStats
    invariants_ok: bool = False


@dataclass
class RunReport:
    load_seconds: float = 0.0
    bike_seconds: float = 0.0
    iterations: list[IterationReport] = field(default_factory=list)


def run(cfg: Config, verbose: bool = True, max_km: float | None = None,
        on_iteration=None, view: dict | None = None, progress: bool = True) -> tuple[Model, RunReport]:
    """Runs the full cycle with configuration ``cfg``. Returns the model (for inspection) and
    a report with timings and simulation statistics. ``max_km`` adds per-trip shares of trips
    up to that distance to iterations.csv (survey comparisons, see mimos.analysis).
    ``on_iteration(row)`` is called with each iterations.csv row (e.g. to print progress).
    ``view`` (mimos.config.load_view) is saved with the run for the report and the videos.
    ``progress`` draws a progress bar on the terminal while the stages run (mimos.progress)."""
    results = Path(cfg.run_dir)
    results.mkdir(parents=True, exist_ok=True)
    save_config(cfg, results / "config.toml", view)
    previous_mode = previous_scores = None
    model = Model(cfg)
    report = RunReport()
    bar = Progress(model, cfg.it_number, enabled=progress)

    def say(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    t0 = time.perf_counter()
    with bar.stage("load"):
        model.load()
    report.load_seconds = time.perf_counter() - t0
    say(f"loaded {model.n_agents} agents in {report.load_seconds:.1f} s")

    t0 = time.perf_counter()
    with bar.stage("bike"):
        model.plan_bike()
    report.bike_seconds = time.perf_counter() - t0
    info = {"n_agents": model.n_agents, "population_size": model.population_size,
            "capacity_scale": model.capacity_scale, "load_seconds": round(report.load_seconds, 1),
            "bike_seconds": round(report.bike_seconds, 1)}
    (results / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    for it in range(cfg.it_number):
        rep = IterationReport(iteration=it)
        bar.start_iteration()
        model.start_iteration(it)

        t0 = time.perf_counter()
        with bar.stage("plan", it):
            model.plan(it)
        rep.seconds["plan"] = time.perf_counter() - t0

        model.choose(it)
        model.prepare_execution(it)

        t0 = time.perf_counter()
        with bar.stage("public", it):
            rep.public = model.simulate_public(it)
        rep.seconds["public"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with bar.stage("drive", it):
            rep.drive = model.simulate_drive(it)
        rep.seconds["drive"] = time.perf_counter() - t0

        model.update_memory(it)
        rep.invariants_ok = model.check_invariants(it)
        if it % cfg.print_it == 0:
            model.write_iteration(it)

        report.iterations.append(rep)
        bar.end_iteration()
        row = iteration_summary(model, it, rep.drive, rep.public, previous_mode, max_km,
                                rep.seconds, rep.invariants_ok, previous_scores)
        previous_scores = all_planned_scores(model)
        append_row(results / "iterations.csv", row, first=(it == 0))
        previous_mode = executed_mode(model)
        if on_iteration is not None:
            on_iteration(row)
        say(
            f"iteration {it}: plan {rep.seconds['plan']:.1f} s | {rep.public} | {rep.drive} | "
            f"invariants {'OK' if rep.invariants_ok else 'VIOLATED (see log.txt)'}"
        )

    model.write_final()
    write_trips(model, results / "trips.csv")
    np.savez_compressed(results / "drive_links.npz", **model.drive_link_traversals())
    return model, report
