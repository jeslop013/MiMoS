"""End-to-end runs on small synthetic corridors (tests/scenario.py).

These check the simulation invariants:
  * every simulated agent finishes its day exactly once (finished_exec == +1),
  * every vehicle is empty and every road link has its full storage and all its gaps back,
  * the drive simulation stays bounded (no endless event chains).
"""

import numpy as np
import pytest

from mimos import config_from_dict, run
from scenario import write_corridor


# With the default utilities, walking 2.5 km (-3.75 per trip) beats driving (C_drive = -13 per
# trip), so nobody would drive on this short corridor. Drive tests make walking expensive.
DRIVERS = {"C_drive": 0.0, "beta_trav_walk": -60.0}


def make_cfg(tmp_path, data, **params):
    values = {"data_dir": str(data), "run_dir": str(tmp_path / "out"), "log_level": "WARNING",
              "it_number": 3, **params}
    return config_from_dict(values)


@pytest.mark.parametrize("n_agents,capacity,lanes", [(5, 20, 1), (40, 20, 1), (40, 3, 1), (60, 5, 2)])
def test_drive_corridor_everyone_finishes(tmp_path, n_agents, capacity, lanes):
    data = write_corridor(tmp_path / "data", n_agents=n_agents, capacity=capacity, lanes=lanes)
    cfg = make_cfg(tmp_path, data, n_agents=n_agents, **DRIVERS)
    model, report = run(cfg, verbose=False)

    assert model.exec_drive.all(), "with default utilities everybody should drive"
    for rep in report.iterations:
        assert rep.invariants_ok
        assert rep.drive.finished == rep.drive.agents == n_agents
        assert rep.drive.jammed_in_links == rep.drive.jammed_in_requests == 0
        # 3 trips worth of events per agent and link, plus a generous margin
        assert rep.drive.events < 200 * n_agents
    # -1 when the agent enters a simulation, +1 when it ends its day: 0 = finished exactly once
    # (< 0: never finished; > 0: duplicated).
    assert all(f == 0 for f in model.finished_exec)


def test_safety_gap_spaces_departures(tmp_path):
    """Cars leaving together enter a 1-lane link one t_s (3 s) apart (thesis: gap dynamics)."""
    n = 6
    data = write_corridor(tmp_path / "data", n_agents=n, capacity=50, lanes=1)
    cfg = make_cfg(tmp_path, data, n_agents=n, it_number=1, **DRIVERS)
    model, _ = run(cfg, verbose=False)

    nodes, times = model.drive_trajectories
    # First record after leaving home (-1) and reaching node 0 is the entry to link 0 -> 1.
    entries = []
    for nd, tm in zip(nodes, times):
        k = [j for j in range(1, len(nd)) if nd[j] == 0][-1]  # last record at node 0 = entry time
        entries.append(tm[k])
    entries = np.sort(np.array(entries))
    gaps = np.diff(entries) * 3600.0
    assert np.allclose(gaps, 3.0, atol=1e-6), gaps


def test_public_corridor(tmp_path):
    """With driving made impossible, everybody rides the bus and all buses end empty."""
    n = 30
    data = write_corridor(tmp_path / "data", n_agents=n)
    # C_drive below C_limit: drive routing is skipped, so the drive score is -1e6.
    cfg = make_cfg(tmp_path, data, n_agents=n, C_drive=-1000.0, C_public=-1.0)
    model, report = run(cfg, verbose=False)

    assert model.exec_public.all()
    for rep in report.iterations:
        assert rep.invariants_ok
        assert rep.public.finished == rep.public.agents == n
    paths = model.public_paths
    assert all(len(p[0]) > 1 for p in paths), "everyone should board the eastbound line in the morning"


@pytest.mark.parametrize("legacy", [True, False])
def test_junction_merge_into_priority_stream(tmp_path, legacy):
    """Side-road cars give way to a dense motorway stream (2 s headway < 3 s safety time),
    then merge. Everybody must finish, and merging cars must have waited at the junction."""
    from scenario import write_network

    nodes = [(41.65, -4.73), (41.65, -4.72), (41.65, -4.71), (41.64, -4.72)]  # W, J, E, S
    links = [(0, 1, 0.5, 1, 50, 1), (1, 2, 0.5, 1, 50, 1), (3, 1, 0.5, 1, 50, 6)]
    main = [(0, 2, 8.0 + k * 2.0 / 3600) for k in range(200)]
    side = [(3, 2, 8.0 + 0.01 + k * 0.005) for k in range(5)]
    write_network(tmp_path / "data", nodes, links, main + side)
    cfg = make_cfg(tmp_path, tmp_path / "data", n_agents=len(main + side), it_number=1,
                   legacy_mode=legacy, **DRIVERS)
    model, report = run(cfg, verbose=False)

    rep = report.iterations[0]
    assert rep.invariants_ok
    assert rep.drive.finished == rep.drive.agents == len(main + side)
    assert rep.drive.jammed_in_links == rep.drive.jammed_in_requests == 0

    # Side cars: trajectory records the junction twice when they had to wait there
    # (arrival at the end of S->J, then the actual entry into J->E).
    nodes_, times_ = model.drive_trajectories
    for i in range(len(main), len(main + side)):
        at_j = [t for n, t in zip(nodes_[i], times_[i]) if n == 1]
        assert len(at_j) >= 2 and at_j[-1] - at_j[0] > 0, "a side car merged without waiting"


def test_random_sampling_and_capacity_scale(tmp_path):
    """Random sample: right size, unique sorted rows, reproducible, automatic capacity scale."""
    data = write_corridor(tmp_path / "data", n_agents=20)

    def sample(seed, out):
        cfg = make_cfg(tmp_path / out, data, n_agents=8, it_number=1, sample_random=True,
                       sample_seed=seed, capacity_scale=0.0)
        model, report = run(cfg, verbose=False)
        assert report.iterations[0].invariants_ok
        return model

    a, b, c = sample(1, "a"), sample(1, "b"), sample(2, "c")
    rows = list(a.population_rows)
    assert len(rows) == 8 and rows == sorted(set(rows)) and all(0 <= r < 20 for r in rows)
    assert rows == list(b.population_rows), "same seed must give the same sample"
    assert rows != list(c.population_rows), "a different seed should give a different sample"
    assert a.population_size == 20 and a.capacity_scale == pytest.approx(8 / 20)


# ---- FIFO link exit and stuck time ----------------------------------

@pytest.mark.parametrize("fifo", [False, True])
def test_fifo_link_exit(tmp_path, fifo):
    """A car that cannot enter a full link blocks the car behind it only with fifo_link_exit.

    A -> B splits into B -> C (room for 1 car, occupied for 30 min by a blocker) and B -> D.
    Car 1 (A -> C) reaches B first and must wait; car 2 (A -> D) arrives 10 s later.
    """
    from scenario import write_network

    nodes = [(41.65, -4.73), (41.65, -4.72), (41.65, -4.71), (41.66, -4.72)]  # A, B, C, D
    links = [(0, 1, 1.0, 1, 10, 6), (1, 2, 30.0, 1, 1, 6), (1, 3, 1.0, 1, 10, 6)]
    agents = [(1, 2, 8.0), (0, 2, 8.0 + 10 / 3600), (0, 3, 8.0 + 20 / 3600)]
    write_network(tmp_path / "data", nodes, links, agents)
    cfg = make_cfg(tmp_path, tmp_path / "data", n_agents=3, it_number=1, fifo_link_exit=fifo, **DRIVERS)
    model, report = run(cfg, verbose=False)

    rep = report.iterations[0]
    assert rep.invariants_ok and rep.drive.finished == 3
    nodes_, times_ = model.drive_trajectories
    at_b = [t for n, t in zip(nodes_[2], times_[2]) if n == 1]  # car 2: end of A -> B, entry to B -> D
    wait_h = at_b[-1] - at_b[0]
    if fifo:
        assert wait_h > 0.4, "car 2 must wait behind car 1 until the blocker leaves B -> C"
    else:
        assert wait_h < 1e-9, "without FIFO car 2 overtakes car 1 at the end of A -> B"


def test_spill_back_keeps_order(tmp_path):
    """A slow bottleneck fills, the queue spills back over the upstream links, cars keep order."""
    from scenario import write_network

    nodes = [(41.65, -4.73), (41.65, -4.72), (41.65, -4.71), (41.65, -4.70)]
    # storage: 3 + 3 + 2 = 8 cars; the bottleneck 2 -> 3 takes 10 minutes
    links = [(0, 1, 0.5, 1, 3, 6), (1, 2, 0.5, 1, 3, 6), (2, 3, 10.0, 1, 2, 6)]
    n = 10
    agents = [(0, 3, 8.0 + 5 * k / 3600) for k in range(n)]
    write_network(tmp_path / "data", nodes, links, agents)
    cfg = make_cfg(tmp_path, tmp_path / "data", n_agents=n, it_number=1, fifo_link_exit=True, **DRIVERS)
    model, report = run(cfg, verbose=False)

    rep = report.iterations[0]
    assert rep.invariants_ok and rep.drive.finished == n
    nodes_, times_ = model.drive_trajectories
    entry0 = [max(t for nd, t in zip(nodes_[i], times_[i]) if nd == 0) for i in range(n)]
    exit3 = [min(t for nd, t in zip(nodes_[i], times_[i]) if nd == 3) for i in range(n)]
    assert exit3 == sorted(exit3), "cars must leave the corridor in the order they entered"
    # Cars 9 and 10 find all 8 places taken: they enter only when the first car leaves 2 -> 3.
    assert all(entry0[i] >= exit3[0] - 1e-9 for i in (8, 9)), "queue did not spill back to the entry"
    assert all(entry0[i] < exit3[0] for i in range(8))


@pytest.mark.parametrize("fifo", [False, True])
def test_stuck_time_resolves_ring_gridlock(tmp_path, fifo):
    """Four cars on a ring of 1-car links, each wanting the link ahead: a permanent gridlock,
    unless stuck_time lets a car move on into the full link."""
    from scenario import write_network

    nodes = [(41.65, -4.73), (41.65, -4.72), (41.66, -4.72), (41.66, -4.73)]
    links = [(k, (k + 1) % 4, 0.5, 1, 1, 6) for k in range(4)]
    agents = [(k, (k + 2) % 4, 8.0) for k in range(4)]
    write_network(tmp_path / "data", nodes, links, agents)

    def simulate(stuck_s, out):
        cfg = make_cfg(tmp_path / out, tmp_path / "data", n_agents=4, it_number=1, fifo_link_exit=fifo,
                       stuck_time=stuck_s / 3600, **DRIVERS)
        model, report = run(cfg, verbose=False)
        return model, report.iterations[0]

    _, jammed = simulate(0, "off")
    assert jammed.drive.jammed_in_links + jammed.drive.jammed_in_requests == 4
    assert jammed.drive.finished == 0

    model, rep = simulate(30, "on")
    assert rep.invariants_ok
    assert rep.drive.finished == 4 and rep.drive.stuck_moves >= 1
    assert rep.drive.jammed_in_links == rep.drive.jammed_in_requests == 0
    assert all(f == 0 for f in model.finished_exec)


def test_link_exit_options_rejected_in_legacy_mode(tmp_path):
    data = write_corridor(tmp_path / "data", n_agents=2)
    cfg = make_cfg(tmp_path, data, n_agents=2, legacy_mode=True, fifo_link_exit=True)
    with pytest.raises(ValueError):
        run(cfg, verbose=False)


# ---- Per-trip modes and durations (mimos.analysis) ---------------------------------------

@pytest.mark.parametrize("params,mode", [(DRIVERS, 1), ({"C_drive": -1000.0, "C_public": -1.0}, 2)])
def test_trip_modes_and_durations(tmp_path, params, mode):
    """Corridor commuters: every trip has the day's mode (car or pt) and a plausible duration."""
    from mimos.analysis import trip_mode_shares

    n = 10
    data = write_corridor(tmp_path / "data", n_agents=n)
    cfg = make_cfg(tmp_path, data, n_agents=n, it_number=1, **params)
    model, _ = run(cfg, verbose=False)

    assert model.trip_modes() == [[mode, mode]] * n
    shares = trip_mode_shares(model)
    assert shares[("walk", "car", "pt", "bike")[mode]] == 1.0
    for trips in model.trip_durations():
        # 2.5 km corridor: a few minutes by car, longer by bus (walk + wait + ride)
        assert all(0.02 < d < 1.0 for d in trips), trips


# ---- Run diagnostics -----------------------------------------------------------------------

def test_run_writes_diagnostics(tmp_path):
    """iterations.csv, config.toml and trips.csv; executed = planned score for uncongested buses."""
    import csv

    from mimos import load_config

    n = 6
    data = write_corridor(tmp_path / "data", n_agents=n)
    cfg = make_cfg(tmp_path, data, n_agents=n, it_number=2, C_drive=-1000.0, C_public=-1.0)
    model, _ = run(cfg, verbose=False)
    out = tmp_path / "out"

    rows = list(csv.DictReader((out / "iterations.csv").open()))
    assert [int(r["iteration"]) for r in rows] == [0, 1]
    assert float(rows[0]["trip_pt"]) == 1.0 and int(rows[1]["mode_switches"]) == 0
    # Planned waiting/riding times are what the buses deliver: the scores agree.
    assert abs(float(rows[1]["score_gap_mean"])) < 0.05
    assert np.allclose(model.executed_scores(), model.public_scores, atol=0.05)

    trips = list(csv.DictReader((out / "trips.csv").open()))
    assert len(trips) == 2 * n and {t["mode"] for t in trips} == {"pt"}

    again = load_config(out / "config.toml")
    assert again.n_agents == n and again.C_public == -1.0 and again.data_dir == cfg.data_dir
