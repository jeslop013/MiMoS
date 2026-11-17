"""Hot exhaust emissions of the simulated private vehicles.

    uv run mimos emissions <run_dir>

Implements the method of the thesis, section "Preliminary LEZ results".

1. Only agents that executed the private vehicle mode emit. Each draws a vehicle class at random
   from the fleet distribution (the scenario's <data_dir>/fleet.csv, else the example fleet in
   mimos/data/fleet_example.csv, see data/SOURCES.md), once, with a fixed
   seed, so a given agent keeps its vehicle across iterations and runs.
2. For every road link the vehicle traversed (drive_links.npz):
       t_mov  = time moving along the link (free flow; t_end - t_enter in the simulation)
       t_stop = time waiting at the end of the link (t_exit - t_end), i.e. t_link - t_mov
       v      = the link speed (velocity_drive.csv), v_min = the factor's minimum speed
3. Hot emission factor, thesis eq. (ef_speed) = EMEP/EEA guidebook polynomial:
       EF(v) = (alpha v^2 + beta v + gamma + delta / v) / (epsilon v^2 + zita v + hta) * (1 - RF)
   with v clamped to the factor's [min, max] speed, exactly as the EEA Appendix 4 sheet does.
       E_mov  = EF(v) * v * t_mov
       E_stop = EF(v_min) * v_min * t_stop
4. E[link, hour, pollutant] (grams; EC in MJ, SPN23 in number of particles), the hour being the hour of day
   at which the vehicle entered the link. The values are for the SIMULATED SAMPLE (as in the
   thesis); divide by the capacity scale for the whole population.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mimos.networks import load_drive_links

POLLUTANTS = ("CO", "NOx", "VOC", "EC", "CH4", "NH3", "N2O", "SPN23", "PM Exhaust")
# EEA Appendix 4: "EF [g/km] or ECF [MJ/km] or #/km" -> EC is energy consumption (MJ),
# SPN23 the number of solid particles > 23 nm.
UNITS = {p: "g" for p in POLLUTANTS} | {"EC": "MJ", "SPN23": "number"}
DATA = Path(__file__).resolve().parent / "data"  # package data: EEA factors, example fleet (data/SOURCES.md)
EXAMPLE_FLEET = DATA / "fleet_example.csv"


@dataclass
class Settings:
    """Choices the thesis leaves open. Defaults = the documented choices."""

    seed: int = 0
    # EEA driving mode, used only by factors that depend on it (PM, some L-category rows):
    # links at least this fast are "Highway", at least rural_kmh "Rural", otherwise urban,
    # "Urban Peak" during peak_hours and "Urban Off Peak" otherwise.
    highway_kmh: float = 90.0
    rural_kmh: float = 60.0
    peak_hours: tuple = ((7, 10), (17, 20))
    factors: Path = field(default_factory=lambda: DATA / "hot_ef.csv")
    fleet: Path | None = None  # None: the scenario's <data_dir>/fleet.csv if present, else EXAMPLE_FLEET

    @classmethod
    def of_run(cls, run_dir: Path | str) -> "Settings":
        """The settings `mimos emissions` used for this run (seed and fleet, from
        emissions/summary.json), so reports and videos redraw the same vehicles."""
        f = Path(run_dir) / "emissions" / "summary.json"
        saved = json.loads(f.read_text()).get("settings", {}) if f.exists() else {}
        fleet = Path(saved["fleet"]) if saved.get("fleet") and Path(saved["fleet"]).exists() else None
        return cls(seed=int(saved.get("seed", 0)), fleet=fleet)


def fleet_for(s: Settings, data_dir: Path | str) -> Path:
    """The fleet file of a run: the explicit one, the scenario's fleet.csv, or the example fleet."""
    if s.fleet is not None:
        return Path(s.fleet)
    own = Path(data_dir) / "fleet.csv"
    return own if own.exists() else EXAMPLE_FLEET


def hot_ef(row: dict, v: np.ndarray) -> np.ndarray:
    """EEA hot emission factor (g/km) of one parameter row at speeds v (km/h), v clamped to the
    row's validity range as in the EEA sheet."""
    v = np.clip(v, row["min_speed"], row["max_speed"])
    num = row["alpha"] * v ** 2 + row["beta"] * v + row["gamma"] + row["delta"] / v
    den = row["epsilon"] * v ** 2 + row["zita"] * v + row["hta"]
    return num / den * (1.0 - row["reduction_factor"])


def hot_ef_times_v(row: dict, v: float) -> float:
    """EF(v) * v (g/h) written as (alpha v^3 + beta v^2 + gamma v + delta) / (...) * (1 - RF):
    the same quantity, but finite at v = 0 (some factors have a minimum speed of 0 km/h)."""
    v = min(max(v, row["min_speed"]), row["max_speed"])
    num = row["alpha"] * v ** 3 + row["beta"] * v ** 2 + row["gamma"] * v + row["delta"]
    den = row["epsilon"] * v ** 2 + row["zita"] * v + row["hta"]
    return num / den * (1.0 - row["reduction_factor"])


class Factors:
    """hot_ef.csv indexed by (category, fuel, segment, euro, pollutant, mode)."""

    NUM = ("min_speed", "max_speed", "alpha", "beta", "gamma", "delta", "epsilon", "zita", "hta",
           "reduction_factor")

    def __init__(self, path: Path):
        self.rows = defaultdict(list)
        with Path(path).open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for k in self.NUM:
                    r[k] = float(r[k]) if r[k] not in ("", None) else 0.0
                self.rows[(r["category"], r["fuel"], r["segment"], r["euro"], r["pollutant"], r["mode"])].append(r)

    def lookup(self, vehicle: tuple, pollutant: str, mode: str) -> list[dict]:
        """Rows for this vehicle and pollutant: the driving-mode rows if the factor has them,
        otherwise the mode-independent rows. Several rows = several technologies (averaged)."""
        return self.rows.get((*vehicle, pollutant, mode)) or self.rows.get((*vehicle, pollutant, ""), [])

    def ef(self, vehicle, pollutant, mode, v):
        """Mean over technologies of EF(v) (g/km) and of EF(v_min) * v_min (g/h); (0, 0) if none."""
        rows = self.lookup(vehicle, pollutant, mode)
        if not rows:
            return None
        e_v = np.mean([hot_ef(r, v) for r in rows], axis=0)
        e_stop = float(np.mean([hot_ef_times_v(r, r["min_speed"]) for r in rows]))
        return e_v, e_stop


def load_fleet(path: Path) -> tuple[list[tuple], np.ndarray]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    classes = [(r["category"], r["fuel"], r["segment"], r["euro"]) for r in rows]
    share = np.array([float(r["share"]) for r in rows])
    return classes, share / share.sum()


def draw_vehicles(n_agents: int, share: np.ndarray, seed: int) -> np.ndarray:
    """Vehicle class of every agent (index into the fleet), fixed by the seed."""
    return np.random.default_rng(seed).choice(len(share), size=n_agents, p=share)


def driving_mode(speed_kmh: np.ndarray, hour: np.ndarray, s: Settings) -> np.ndarray:
    peak = np.zeros(hour.shape, dtype=bool)
    for a, b in s.peak_hours:
        peak |= (hour >= a) & (hour < b)
    mode = np.where(peak, "Urban Peak", "Urban Off Peak").astype(object)
    mode[speed_kmh >= s.rural_kmh] = "Rural"
    mode[speed_kmh >= s.highway_kmh] = "Highway"
    return mode


def traversal_grams(traversals: dict, links, n_agents: int, s: Settings | None = None,
                    pollutants=POLLUTANTS) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Emissions of every link traversal (the thesis' E_mov + E_stop), before any aggregation.

    Returns (G, k, missing, vehicle_of_agent): G[traversal, pollutant] (float32; grams, EC in
    MJ, SPN23 in number), k = link index of each traversal, missing = traversals without a factor
    per pollutant. The report and the videos bin G at any time resolution (e.g. 10 minutes); the
    thesis output E[link, hour, pollutant] is its hourly sum (compute).
    """
    s = s or Settings()
    factors = Factors(s.factors)
    classes, share = load_fleet(s.fleet or EXAMPLE_FLEET)
    vehicle_of_agent = draw_vehicles(n_agents, share, s.seed)

    agent = np.asarray(traversals["agent"])
    k = links.index(np.asarray(traversals["u"]), np.asarray(traversals["av"]))
    t_mov = np.asarray(traversals["t_end"]) - np.asarray(traversals["t_enter"])
    t_stop = np.maximum(0.0, np.asarray(traversals["t_exit"]) - np.asarray(traversals["t_end"]))
    hour = (np.floor(np.asarray(traversals["t_enter"])) % 24).astype(int)
    v = links.speed_kmh[k]
    veh = vehicle_of_agent[agent]
    mode = driving_mode(v, hour, s)

    G = np.zeros((agent.size, len(pollutants)), dtype=np.float32)
    missing = defaultdict(int)
    # group the traversals by (vehicle class, driving mode): one vectorised evaluation each
    mode_names = np.array(["Urban Peak", "Urban Off Peak", "Rural", "Highway"], dtype=object)
    mode_id = np.searchsorted(np.sort(mode_names), mode)  # any consistent integer code
    group = veh.astype(np.int64) * 8 + mode_id
    for g in np.unique(group):
        sel = np.flatnonzero(group == g)
        vehicle, m = classes[veh[sel[0]]], mode[sel[0]]
        if vehicle[1] == "Battery electric":
            continue  # no exhaust
        for p, pol in enumerate(pollutants):
            res = factors.ef(vehicle, pol, m, v[sel])
            if res is None:
                missing[pol] += len(sel)  # e.g. NH3, N2O: not in Appendix 4 for cars and vans
                continue
            e_v, e_stop = res
            G[sel, p] = e_v * v[sel] * t_mov[sel] + e_stop * t_stop[sel]
    info = {"classes": classes, "agent": agent, "missing": missing}
    return G, k, info, vehicle_of_agent


def compute(traversals: dict, links, n_agents: int, s: Settings | None = None) -> tuple[np.ndarray, dict]:
    """E[link, hour, pollutant] for the traversals of drive_links.npz. Returns (E, summary)."""
    s = s or Settings()
    G, k, info, vehicle_of_agent = traversal_grams(traversals, links, n_agents, s)
    hour = (np.floor(np.asarray(traversals["t_enter"])) % 24).astype(int)
    E = np.zeros((len(links.u), 24, len(POLLUTANTS)), dtype=np.float64)
    for p in range(len(POLLUTANTS)):
        np.add.at(E[:, :, p], (k, hour), G[:, p].astype(np.float64))
    agent, classes, missing = info["agent"], info["classes"], info["missing"]

    drawn = np.bincount(vehicle_of_agent[np.unique(agent)], minlength=len(classes)) if agent.size else np.zeros(len(classes))
    summary = {
        "pollutants": list(POLLUTANTS), "units": UNITS,
        "totals_sample": {p: float(E[:, :, i].sum()) for i, p in enumerate(POLLUTANTS)},
        "traversals": int(agent.size), "vehicles": int(len(np.unique(agent))),
        "vehicle_km_sample": float(np.nansum(links.length_km[k])),
        "vehicles_by_class": {" / ".join(c): int(n) for c, n in zip(classes, drawn) if n},
        "traversals_without_factor": {p: missing.get(p, 0) for p in POLLUTANTS}, "settings": {
            "seed": s.seed, "highway_kmh": s.highway_kmh, "rural_kmh": s.rural_kmh,
            "peak_hours": [list(b) for b in s.peak_hours], "factors": str(s.factors), "fleet": str(s.fleet)},
    }
    return E, summary


def run_emissions(run_dir: Path | str, s: Settings | None = None) -> Path:
    """Computes the emissions of a results folder and writes <run_dir>/emissions/."""
    import tomllib

    results = Path(run_dir)
    with (results / "config.toml").open("rb") as f:
        cfg = tomllib.load(f)["config"]
    info = json.loads((results / "run_info.json").read_text())
    links = load_drive_links(cfg["data_dir"])
    s = s or Settings()
    s.fleet = fleet_for(s, cfg["data_dir"])
    E, summary = compute(dict(np.load(results / "drive_links.npz")), links, int(info["n_agents"]), s)
    scale = float(info["capacity_scale"])
    summary["capacity_scale"] = scale
    summary["totals_population"] = {p: v / scale for p, v in summary["totals_sample"].items()}

    out = results / "emissions"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "E.npz", E=E.astype(np.float32), pollutants=np.array(POLLUTANTS),
                        link_u=links.u, link_av=links.av)
    used = np.flatnonzero(E.sum(axis=(1, 2)) > 0)
    for p, pol in enumerate(POLLUTANTS):
        name = pol.replace(" ", "_")
        with (out / f"{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["link", "u", "av", "v", *[f"h{h}" for h in range(24)]])
            for k in used:
                w.writerow([k, links.u[k], links.av[k], links.v[k], *[f"{x:.6g}" for x in E[k, :, p]]])
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return out
