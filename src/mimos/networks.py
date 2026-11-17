"""Light readers of the network CSV files for analysis (report, emissions).

Same files and units as the C++ loaders (cpp/src/network.cpp): drive link times in minutes,
lengths in metres, speeds in km/h; rows are adjacency lists, -1 for "no link".
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _rows(path: Path, cast=float) -> list[list]:
    with path.open(newline="") as f:
        return [[cast(x) for x in row if x.strip()] for row in csv.reader(f)]


@dataclass
class DriveLinks:
    """Flat link table of the drive network: link k = (u[k], av[k]) -> v[k]."""

    lat: np.ndarray       # node latitudes
    lon: np.ndarray
    u: np.ndarray
    av: np.ndarray
    v: np.ndarray
    length_km: np.ndarray
    free_flow_h: np.ndarray
    speed_kmh: np.ndarray  # velocity_drive.csv (the OSM-based link speed)
    offset: np.ndarray     # link index of (u, 0); link index of (u, av) = offset[u] + av
    priority: np.ndarray | None = None  # priority_drive.csv: 0 motorway ... 6 residential
    lanes: np.ndarray | None = None     # lanes_drive.csv (min 1, as the model uses them)
    storage: np.ndarray | None = None   # capacity_drive.csv: cars the link holds (min 1)
    # Real road shapes for maps (optional link_geometry.npz, see docs/input_format.md):
    # link k = lon/lat[geom_start[k]:geom_start[k + 1]]. None: draw straight lines.
    geom_start: np.ndarray | None = None
    geom_lon: np.ndarray | None = None
    geom_lat: np.ndarray | None = None

    def shape(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(lon, lat) points of link k: its real shape if available, else its two end nodes."""
        if self.geom_start is not None:
            a, b = self.geom_start[k], self.geom_start[k + 1]
            return self.geom_lon[a:b], self.geom_lat[a:b]
        return (np.array([self.lon[self.u[k]], self.lon[self.v[k]]], dtype=np.float32),
                np.array([self.lat[self.u[k]], self.lat[self.v[k]]], dtype=np.float32))

    def index(self, u: np.ndarray, av: np.ndarray) -> np.ndarray:
        return self.offset[u] + av


def load_drive_links(data_dir: Path | str) -> DriveLinks:
    d = Path(data_dir) / "net_files" / "drive_network"
    nodes = np.array(_rows(d / "nodes_drive.csv"))
    adj = _rows(d / "adj_drive.csv", lambda x: int(float(x)))
    lengths = _rows(d / "lengths_drive.csv")
    times = _rows(d / "times_drive.csv")
    speeds = _rows(d / "velocity_drive.csv")
    prios = _rows(d / "priority_drive.csv")
    lanes_rows = _rows(d / "lanes_drive.csv")
    cap_rows = _rows(d / "capacity_drive.csv")
    u, av, v, km, ff, sp, pr, ln, cp = [], [], [], [], [], [], [], [], []
    offset = np.zeros(len(adj), dtype=np.int64)
    for node, row in enumerate(adj):
        offset[node] = len(u)
        for k, nxt in enumerate(row):
            # rows without links are [-1]; keep the slot so offset[u] + av stays valid
            u.append(node)
            av.append(k)
            v.append(nxt)
            km.append(lengths[node][k] / 1000.0 if nxt >= 0 else np.nan)
            ff.append(times[node][k] / 60.0 if nxt >= 0 else np.nan)
            sp.append(speeds[node][k] if nxt >= 0 else np.nan)
            pr.append(prios[node][k] if nxt >= 0 else np.nan)
            # the model raises lanes and storage below 1 to 1 (Model::load, except in legacy_mode)
            ln.append(max(1.0, lanes_rows[node][k]) if nxt >= 0 else np.nan)
            cp.append(max(1.0, cap_rows[node][k]) if nxt >= 0 else np.nan)
    links = DriveLinks(nodes[:, 1], nodes[:, 2], np.array(u), np.array(av), np.array(v), np.array(km),
                       np.array(ff), np.array(sp), offset, np.array(pr), np.array(ln), np.array(cp))
    geom = d / "link_geometry.npz"
    if geom.exists():
        g = np.load(geom)
        if len(g["start"]) == len(links.u) + 1:  # same network
            links.geom_start, links.geom_lon, links.geom_lat = g["start"], g["lon"], g["lat"]
    return links


@dataclass
class PublicNodes:
    mode: np.ndarray   # -1 rental bike, 0 urban bus, 1 regional bus, 2 train, ...
    itin: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    name: list
    next_stop: np.ndarray  # first adjacency entry = next stop of the same itinerary, -1 if none


def load_public_nodes(data_dir: Path | str) -> PublicNodes:
    d = Path(data_dir) / "net_files" / "public_network"
    with (d / "nodes_public.csv").open(newline="") as f:
        rows = list(csv.reader(f))
    adj = _rows(d / "adj_public.csv", lambda x: int(float(x)))
    return PublicNodes(np.array([int(float(r[0])) for r in rows]), np.array([int(float(r[1])) for r in rows]),
                       np.array([float(r[3]) for r in rows]), np.array([float(r[4]) for r in rows]),
                       [r[5] for r in rows], np.array([a[0] if a else -1 for a in adj]))


def read_trajectories(nodes_file: Path, times_file: Path) -> list[tuple[list[int], list[float]]]:
    """exec_<mode>_paths/_passage_times CSV pair -> per agent (nodes, times); [] if not used."""
    out = []
    with nodes_file.open(newline="") as fn, times_file.open(newline="") as ft:
        for rn, rt in zip(csv.reader(fn), csv.reader(ft)):
            if rn == ["-999"]:
                out.append(([], []))
            else:
                out.append(([int(x) for x in rn], [float(x) for x in rt]))
    return out
