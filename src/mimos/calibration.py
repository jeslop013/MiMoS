"""Calibrates the mode constants against a survey's per-trip modal split.

    uv run mimos calibrate <config.toml> <targets.toml> [--n-agents N] [--rounds R] [--run-dir DIR]

From Python: ``mimos.calibration.calibrate(config, targets, ...)``. Writes rounds.csv and
summary.txt to the output folder (default: "calibration" next to the config's run_dir).

Method: the standard alternative-specific-constant
(ASC) iteration of discrete-choice models, adapted to MiMoS, where a mode is chosen per DAY but
its constant is added per TRIP:

    dC_m = eta * [ ln(S_m / s_m) - ln(S_walk / s_walk) ] / n_m

S = survey share, s = simulated share of trips by basic mode (only trips up to the
targets' [scope] max_trip_km), n_m = mean trips per day of the
agents executing the day-mode that carries the constant (C_drive: drive days, C_public: public
days, C_bike: bike days). Walking (C_walk = 0) is the reference. Steps are damped (eta) and
capped (max_step) because public transport competes with walking inside a public day through
deterministic routing, not logit, so its response is steeper than a logit model's.

Each round builds a fresh Model: the constants are folded into the access scores when the data
are loaded (cpp/src/population.cpp), so they cannot change between iterations of one run. Every
round uses the same seeds and agent sample, so differences between rounds come from the
constants only.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

from mimos import Model, load_config
from mimos.analysis import (MODES, duration_histograms, duration_summary, load_targets,
                            trip_mode_shares, trip_table, trips_per_day)

# constant -> (basic mode it controls, day-mode whose trips carry it)
PARAMS = {"C_drive": ("car", "drive"), "C_public": ("pt", "public"), "C_bike": ("bike", "bike")}


def simulate(cfg, iterations: int):
    """Runs one model for `iterations` iterations; returns (model, last DriveStats)."""
    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    m = Model(cfg)
    m.load()
    m.plan_bike()
    drive = None
    for it in range(iterations):
        m.start_iteration(it)
        m.plan(it)
        m.choose(it)
        m.prepare_execution(it)
        m.simulate_public(it)
        drive = m.simulate_drive(it)
        m.update_memory(it)
        m.check_invariants(it)
    return m, drive


def fmt(shares: dict) -> str:
    return "  ".join(f"{k} {100 * v:5.2f}%" for k, v in shares.items())


def calibrate(config, targets_file, *, n_agents: int | None = None, iterations: int = 3, rounds: int = 12,
              tol: float = 0.01, eta: float = 0.7, max_step: float = 2.0, final_iterations: int = 10,
              run_dir=None, start: list[str] | None = None) -> Path:
    """Fits C_drive, C_public, C_bike (and C_rent_bike = C_bike) to the targets' per-trip shares.

    iterations: per round; tol: stop when every share is within tol; eta: damping; max_step:
    largest change of a constant per round; final_iterations: validation run with the best
    constants (0 = skip); start: starting constants as "NAME=VALUE" instead of the config's.
    Returns the path of summary.txt.
    """
    targets = load_targets(targets_file)
    S = targets["shares"]
    out = Path(run_dir) if run_dir else Path(load_config(config).run_dir).parent / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    base = load_config(config, n_agents=n_agents, run_dir=str(out / "run"))
    params = {name: getattr(base, name) for name in PARAMS}
    for item in start or []:
        name, value = item.split("=")
        assert name in PARAMS, f"--start: {name} is not one of {list(PARAMS)}"
        params[name] = float(value)
    max_km = targets["scope"].get("max_trip_km")
    print(f"scope: trips up to {max_km} km")
    print(f"targets: {fmt(S)}")
    print(f"start:   {params}")

    rows = []
    with (out / "rounds.csv").open("w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["round", *PARAMS, *(f"s_{m}" for m in MODES), "max_error", "gridlocked", "stuck_moves", "seconds"])
        for rnd in range(rounds):
            t0 = time.perf_counter()
            cfg = load_config(config, n_agents=n_agents, run_dir=str(out / "run"))
            for name, value in params.items():
                setattr(cfg, name, value)
            cfg.C_rent_bike = params["C_bike"]  # bike sharing priced like a private bike
            m, drive = simulate(cfg, iterations)
            s = trip_mode_shares(m, max_km)
            n = trips_per_day(m)
            err = max(abs(s[k] - S[k]) for k in MODES)
            jam = drive.jammed_in_links + drive.jammed_in_requests
            secs = time.perf_counter() - t0
            log.writerow([rnd, *params.values(), *s.values(), err, jam, drive.stuck_moves, round(secs)])
            f.flush()
            rows.append((dict(params), s, err))
            print(f"round {rnd:2d} | {fmt(s)} | max error {100 * err:5.2f} pp | gridlocked {jam} | {secs:.0f} s", flush=True)
            del m
            if err < tol:
                break
            # ASC update relative to walking (shares floored to avoid log(0))
            ref = math.log(S["walk"] / max(s["walk"], 1e-4))
            for name, (mode, day_mode) in PARAMS.items():
                step = eta * (math.log(S[mode] / max(s[mode], 1e-4)) - ref) / n[day_mode]
                step = max(-max_step, min(max_step, step))
                # never below C_limit: that would switch the mode off (routing skipped)
                params[name] = max(params[name] + step, cfg.C_limit + 1.0)
            print(f"         -> {', '.join(f'{k} = {v:.4f}' for k, v in params.items())}", flush=True)

    best_params, best_s, best_err = min(rows, key=lambda r: r[2])
    summary = [f"targets:    {fmt(S)}", f"calibrated: {fmt(best_s)}  (max error {100 * best_err:.2f} pp)",
               "constants:  " + ", ".join(f"{k} = {v:.4f}" for k, v in best_params.items())
               + f", C_rent_bike = {best_params['C_bike']:.4f}"]

    if final_iterations > 0:
        cfg = load_config(config, n_agents=n_agents, run_dir=str(out / "validation"))
        for name, value in best_params.items():
            setattr(cfg, name, value)
        cfg.C_rent_bike = best_params["C_bike"]
        t0 = time.perf_counter()
        m, drive = simulate(cfg, final_iterations)
        s = trip_mode_shares(m, max_km)
        n_all, n_in = len(trip_table(m)[0]), len(trip_table(m, max_km)[0])
        summary += [f"trips within {max_km} km: {n_in} of {n_all} ({100 * n_in / n_all:.1f}%); "
                    f"all trips: {fmt(trip_mode_shares(m))}",f"validation ({final_iterations} iterations, {time.perf_counter() - t0:.0f} s): {fmt(s)}",
                    f"gridlocked {drive.jammed_in_links + drive.jammed_in_requests}, stuck moves {drive.stuck_moves}",
                    "", duration_summary(m, targets, max_km), "", "duration bins (min): sim % / survey %"]
        edges = targets["durations"]["bin_edges"]
        for mode, (sim, ref) in duration_histograms(m, targets, max_km).items():
            summary.append(f"{mode:5s} " + " ".join(f"{e}:{100 * a:.0f}/{100 * b:.0f}" for e, a, b in zip(edges, sim, ref)))

    text = "\n".join(summary)
    print("\n" + text)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    return out / "summary.txt"
