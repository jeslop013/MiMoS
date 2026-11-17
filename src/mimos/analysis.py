"""Per-trip results of a simulated iteration, comparable with mobility surveys.

MiMoS chooses one mode per agent-day (drive, public or bike), but surveys count trips by
"basic mode". ``Model.trip_modes()`` classifies every executed trip: a public day can contain
bus trips and walk-only trips; a drive day can contain walked trips with no road route.

``max_km`` keeps only trips whose two agenda locations are at most that far apart (straight
line). An agenda can contain far destinations; a city mobility survey does not count those
as daily trips.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

MODES = ("walk", "car", "pt", "bike")  # index = mode code of Model.trip_modes()


def _flat(nested, dtype) -> np.ndarray:
    return np.fromiter((x for agent in nested for x in agent), dtype=dtype)


def trip_table(model, max_km: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Flat arrays (mode code, duration in minutes) over the trips of every agent."""
    modes = _flat(model.trip_modes(), np.int8)
    minutes = _flat(model.trip_durations(), float) * 60.0
    if max_km is not None:
        keep = _flat(model.trip_distances_km(), float) <= max_km
        modes, minutes = modes[keep], minutes[keep]
    return modes, minutes


def trip_mode_shares(model, max_km: float | None = None) -> dict[str, float]:
    """Share of trips by basic mode (walk, car, pt, bike), summing to 1."""
    modes, _ = trip_table(model, max_km)
    counts = np.bincount(modes, minlength=len(MODES))
    return {name: counts[k] / counts.sum() for k, name in enumerate(MODES)}


def trips_per_day(model) -> dict[str, float]:
    """Mean number of trips per day of the agents executing each day-mode (drive, public, bike).

    Mode constants are added per trip but chosen per day, so a constant change of dC shifts the
    day utility of an agent with n trips by n * dC (used by the calibration update).
    """
    trips = np.array([len(agent) for agent in model.trip_modes()])
    out = {}
    for name, flags in (("drive", model.exec_drive), ("public", model.exec_public), ("bike", model.exec_bike)):
        flags = np.asarray(flags, dtype=bool)
        out[name] = float(trips[flags].mean()) if flags.any() else float(trips.mean())
    return out


def load_targets(path: Path | str) -> dict:
    """Reads a targets file (a survey targets file) and adds the renormalised ``shares``."""
    with Path(path).open("rb") as f:
        targets = tomllib.load(f)
    trips = targets["trips"]
    total = sum(trips[m] for m in MODES)
    targets["shares"] = {m: trips[m] / total for m in MODES}
    targets.setdefault("scope", {})
    return targets


def duration_summary(model, targets: dict | None = None, max_km: float | None = None) -> str:
    """Mean trip duration (minutes) by basic mode, next to the survey values when given."""
    modes, minutes = trip_table(model, max_km)
    lines = ["mode  | trips  | mean min (sim) | median min (sim) | mean min (survey)"]
    survey = (targets or {}).get("durations", {}).get("mean_minutes", {})
    for k, name in enumerate(MODES):
        sel = (modes == k) & np.isfinite(minutes)
        mean = minutes[sel].mean() if sel.any() else float("nan")
        median = np.median(minutes[sel]) if sel.any() else float("nan")
        ref = survey.get(name)
        lines.append(f"{name:5s} | {int(sel.sum()):6d} | {mean:14.1f} | {median:16.1f} | "
                     f"{'' if ref is None else f'{ref:.1f}'}")
    return "\n".join(lines)


def duration_histograms(model, targets: dict, max_km: float | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per mode: (simulated, survey) fraction of trips in the survey's duration bins."""
    modes, minutes = trip_table(model, max_km)
    d = targets["durations"]
    edges = np.array(d["bin_edges"] + [np.inf], dtype=float)
    out = {}
    for k, name in enumerate(MODES):
        if name not in d:
            continue
        sel = (modes == k) & np.isfinite(minutes)
        sim, _ = np.histogram(minutes[sel], bins=edges)
        ref = np.array(d[name], dtype=float)
        out[name] = (sim / max(sim.sum(), 1), ref / ref.sum())
    return out


# ---- Per-iteration summary and run outputs -----------------------------------------------

DAY_MODES = ("none", "drive", "public", "bike")  # index = executed_mode() code


def executed_mode(model) -> np.ndarray:
    """Day-mode each agent executed: 0 none, 1 drive, 2 public, 3 bike."""
    mode = np.zeros(model.n_agents, dtype=np.int8)
    mode[np.asarray(model.exec_bike, dtype=bool)] = 3
    mode[np.asarray(model.exec_public, dtype=bool)] = 2
    mode[np.asarray(model.exec_drive, dtype=bool)] = 1
    return mode


def planned_scores(model) -> np.ndarray:
    """Planned (perceived) score of the mode each agent executed; NaN if none."""
    mode = executed_mode(model)
    out = np.full(model.n_agents, np.nan)
    for code, scores in ((1, model.drive_scores), (2, model.public_scores), (3, model.bike_scores)):
        out[mode == code] = np.asarray(scores)[mode == code]
    return out


def _stats(prefix: str, x: np.ndarray) -> dict:
    x = x[np.isfinite(x) & (x > -1e5)]  # -1e6 marks "mode unavailable"
    if x.size == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_p5": np.nan, f"{prefix}_p95": np.nan}
    return {f"{prefix}_mean": float(x.mean()), f"{prefix}_p5": float(np.percentile(x, 5)),
            f"{prefix}_p95": float(np.percentile(x, 95))}


def all_planned_scores(model) -> np.ndarray:
    """Planned drive and public scores of every agent (both modes), for score changes."""
    return np.concatenate([np.asarray(model.drive_scores), np.asarray(model.public_scores)])


def iteration_summary(model, it: int, drive=None, public=None, previous_mode=None,
                      max_km: float | None = None, seconds: dict | None = None,
                      invariants_ok: bool | None = None, previous_scores=None) -> dict:
    """One row of iterations.csv: scores (planned vs executed), mode switches, shares, traffic.

    score_change_mean: mean |planned score(it) - planned score(it-1)| over every agent's drive
    and public scores. It does not depend on the score scale (far trips make scores of -300),
    so it shows relaxation directly: it decays towards 0 as perceptions stop changing.
    """
    mode = executed_mode(model)
    plan, execd = planned_scores(model), np.asarray(model.executed_scores())
    row = {"iteration": it}
    row |= _stats("score_planned", plan)
    row |= _stats("score_executed", execd)
    both = np.isfinite(plan) & np.isfinite(execd) & (plan > -1e5)
    row["score_gap_mean"] = float((plan[both] - execd[both]).mean()) if both.any() else np.nan
    row["unfinished_agents"] = int(((mode > 0) & ~np.isfinite(execd)).sum())
    if previous_scores is not None:
        now = all_planned_scores(model)
        ok = np.isfinite(now) & np.isfinite(previous_scores) & (now > -1e5) & (previous_scores > -1e5)
        row["score_change_mean"] = float(np.abs(now[ok] - previous_scores[ok]).mean()) if ok.any() else np.nan
    else:
        row["score_change_mean"] = np.nan
    row["mode_switches"] = int((mode != previous_mode).sum()) if previous_mode is not None else np.nan
    for code, name in enumerate(DAY_MODES[1:], start=1):
        row[f"day_{name}"] = float((mode == code).mean())
    for name, v in trip_mode_shares(model).items():
        row[f"trip_{name}"] = float(v)
    if max_km is not None:
        for name, v in trip_mode_shares(model, max_km).items():
            row[f"trip_{name}_within"] = float(v)
    modes, minutes = trip_table(model, max_km)
    car = minutes[(modes == 1) & np.isfinite(minutes)]
    row["car_trip_minutes_mean"] = float(car.mean()) if car.size else np.nan
    if drive is not None:
        row |= {"gridlocked": drive.jammed_in_links + drive.jammed_in_requests,
                "stuck_moves": drive.stuck_moves, "drive_events": drive.events}
    if public is not None:
        row["public_events"] = public.events
    if invariants_ok is not None:
        row["invariants_ok"] = bool(invariants_ok)
    for stage, s in (seconds or {}).items():
        row[f"seconds_{stage}"] = round(s, 2)
    return row


def append_row(path: Path | str, row: dict, first: bool) -> None:
    """Writes `row` to a CSV, with the header when `first` (the file is then replaced)."""
    import csv

    with Path(path).open("w" if first else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if first:
            w.writeheader()
        w.writerow(row)


def write_trips(model, path: Path | str) -> None:
    """trips.csv of the last iteration: agent, trip, basic mode, departure hour, minutes, km."""
    import csv

    modes, deps, durs, kms = (model.trip_modes(), model.trip_departures(), model.trip_durations(),
                              model.trip_distances_km())
    rows = model.population_rows
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "population_row", "trip", "mode", "departure_hour", "minutes", "distance_km"])
        for i, (m, d, t, k) in enumerate(zip(modes, deps, durs, kms)):
            for il in range(len(m)):
                w.writerow([i, rows[i], il, MODES[m[il]], f"{d[il]:.4f}", f"{60 * t[il]:.2f}", f"{k[il]:.3f}"])
