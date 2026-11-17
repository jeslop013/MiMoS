"""Emissions module (thesis "Preliminary LEZ results")."""

import csv

import numpy as np
import pytest

from mimos.emissions import POLLUTANTS, Settings, compute, draw_vehicles, hot_ef
from mimos.networks import DriveLinks

FACTOR_COLUMNS = ["category", "fuel", "segment", "euro", "technology", "pollutant", "mode", "min_speed",
                  "max_speed", "alpha", "beta", "gamma", "delta", "epsilon", "zita", "hta", "reduction_factor"]


def test_hot_ef_matches_hand_calculation():
    # EEA Appendix 4 row: Passenger Cars, Diesel, Mini, Euro 6 a/b/c, DPF, NOx
    row = dict(min_speed=10, max_speed=130, alpha=6.67136e-05, beta=-0.011381467, gamma=0.945951727,
               delta=1.923608149, epsilon=-5.15046e-05, zita=0.004264272, hta=1, reduction_factor=0.176306450219664)
    # (a v^2 + b v + g + d/v) / (e v^2 + z v + h) * (1 - RF) at 50 km/h, computed by hand
    assert hot_ef(row, np.array(50.0)) == pytest.approx(0.4421592, rel=1e-6)
    # below / above the validity range the EEA sheet clamps the speed
    assert hot_ef(row, np.array(3.0)) == pytest.approx(hot_ef(row, np.array(10.0)))
    assert hot_ef(row, np.array(200.0)) == pytest.approx(hot_ef(row, np.array(130.0)))


def _inputs(tmp_path, ef_g_km=1.0, vmin=10.0):
    """One vehicle class with a speed-independent EF for every pollutant (gamma / hta)."""
    factors, fleet = tmp_path / "ef.csv", tmp_path / "fleet.csv"
    with factors.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FACTOR_COLUMNS)
        for p in POLLUTANTS:
            w.writerow(["Passenger Cars", "Petrol", "Medium", "Euro 4", "", p, "", vmin, 130, 0, 0, ef_g_km, 0, 0, 0, 1, 0])
    with fleet.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["category", "fuel", "segment", "euro", "share"],
                                 ["Passenger Cars", "Petrol", "Medium", "Euro 4", 1.0]])
    return Settings(factors=factors, fleet=fleet)


def _links():
    """Two links: 0 -> 1 (1 km at 50 km/h) and 1 -> 2 (2 km at 100 km/h)."""
    return DriveLinks(lat=np.zeros(3), lon=np.zeros(3), u=np.array([0, 1, 2]), av=np.array([0, 0, 0]),
                      v=np.array([1, 2, -1]), length_km=np.array([1.0, 2.0, np.nan]),
                      free_flow_h=np.array([1 / 50, 2 / 100, np.nan]), speed_kmh=np.array([50.0, 100.0, np.nan]),
                      offset=np.array([0, 1, 2]))


def test_free_flow_emissions_are_ef_times_distance(tmp_path):
    s = _inputs(tmp_path, ef_g_km=1.0)
    links = _links()
    # agent 0 drives both links in free flow at 8:00 (no waiting: t_exit = t_end)
    t = {"agent": [0, 0], "u": [0, 1], "av": [0, 0], "t_enter": [8.0, 8.02], "t_end": [8.02, 8.04],
         "t_exit": [8.02, 8.04]}
    E, summary = compute(t, links, n_agents=1, s=s)
    nox = POLLUTANTS.index("NOx")
    assert E[0, 8, nox] == pytest.approx(1.0)   # 1 g/km x 1 km
    assert E[1, 8, nox] == pytest.approx(2.0)   # 1 g/km x 2 km
    assert summary["vehicle_km_sample"] == pytest.approx(3.0)


def test_waiting_adds_ef_at_minimum_speed(tmp_path):
    s = _inputs(tmp_path, ef_g_km=1.0, vmin=10.0)
    links = _links()
    # the car waits 6 minutes (0.1 h) at the end of link 0
    t = {"agent": [0], "u": [0], "av": [0], "t_enter": [8.0], "t_end": [8.02], "t_exit": [8.12]}
    E, _ = compute(t, links, n_agents=1, s=s)
    # E_mov = EF(v) v t_mov = 1 g; E_stop = EF(v_min) v_min t_stop = 1 x 10 x 0.1 = 1 g
    assert E[0, 8, POLLUTANTS.index("CO")] == pytest.approx(2.0)


def test_vehicle_draw_follows_fleet_shares():
    share = np.array([0.7, 0.2, 0.1])
    drawn = draw_vehicles(100000, share, seed=1)
    assert np.allclose(np.bincount(drawn) / drawn.size, share, atol=0.01)
    assert np.array_equal(drawn, draw_vehicles(100000, share, seed=1)), "same seed, same vehicles"
