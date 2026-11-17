"""Synthetic test scenarios in the exact MiMoS input format (see README "Data").

``write_corridor`` builds a straight corridor of 4 intersections, ~0.8 km apart, with:
  * a two-way road (configurable lanes, storage capacity, link time),
  * the same chain as a bike network,
  * one bus line in each direction with a vehicle every 15 minutes,
  * ``n_agents`` agents going home (A, west end) -> work (B, east end) -> home,
    all leaving home at the same time to create interactions,
  * the precomputed access/egress candidates for every trip.
Small enough that every expected value can be checked by hand.
"""

from __future__ import annotations

import csv
from pathlib import Path

LAT = 41.65
LON0 = -4.73
DLON = 0.01  # ~0.83 km at this latitude
N_NODES = 4
LINK_M = 830.0


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)  # CRLF line endings, like the real pre-processing


def write_corridor(
    root: Path,
    n_agents: int = 20,
    lanes: int = 1,
    capacity: int = 20,
    link_minutes: float = 1.0,
    depart: float = 8.0,
    bus_capacity_mode: int = 0,
) -> Path:
    """Writes a complete dataset under ``root`` and returns ``root`` (with trailing data layout)."""
    net = root / "net_files"
    pop = root / "pop_files"
    xy = [(LAT, LON0 + k * DLON) for k in range(N_NODES)]

    # ---- Drive and bike: two-way chain 0 - 1 - 2 - 3 ------------------------------------
    adj = [[k + d for d in (-1, 1) if 0 <= k + d < N_NODES] for k in range(N_NODES)]
    inv = adj  # symmetric
    attr = lambda value: [[value] * len(row) for row in adj]  # noqa: E731

    d = net / "drive_network"
    _write(d / "nodes_drive.csv", [[1000 + k, lat, lon] for k, (lat, lon) in enumerate(xy)])
    _write(d / "adj_drive.csv", adj)
    _write(d / "inv_adj_drive.csv", inv)
    _write(d / "times_drive.csv", attr(link_minutes))   # minutes
    _write(d / "lengths_drive.csv", attr(LINK_M))       # metres
    _write(d / "lanes_drive.csv", attr(lanes))
    _write(d / "capacity_drive.csv", attr(capacity))
    _write(d / "velocity_drive.csv", attr(50))
    _write(d / "priority_drive.csv", attr(6))

    b = net / "bike_network"
    _write(b / "nodes_bike.csv", [[2000 + k, lat, lon] for k, (lat, lon) in enumerate(xy)])
    _write(b / "adj_bike.csv", adj)
    _write(b / "times_bike.csv", attr(3.0))
    _write(b / "cycleway_bike.csv", attr(0))

    # ---- Public: line 0 eastbound (nodes 0..3), line 1 westbound (nodes 4..7) ------------
    p = net / "public_network"
    nodes, padj, ptimes, plengths, sch, ids = [], [], [], [], [], []
    headers = [5.0 + 0.25 * k for k in range(int((23.0 - 5.0) / 0.25) + 1)]
    bus_min = 2.0  # minutes between stops
    vehicle = 0
    for itin, order in ((0, range(N_NODES)), (1, reversed(range(N_NODES)))):
        first_vehicle = vehicle
        for stop, k in enumerate(order):
            lat, lon = xy[k]
            nodes.append([bus_capacity_mode, itin, stop, lat, lon, f"L{itin}", 0.0, 0])
            node_id = len(nodes) - 1
            last = stop == N_NODES - 1
            padj.append([-1] if last else [node_id + 1])
            ptimes.append([-1] if last else [bus_min])
            plengths.append([-1] if last else [LINK_M])
            sch.append([h + stop * bus_min / 60.0 for h in headers])
            ids.append([first_vehicle + j for j in range(len(headers))])
        vehicle += len(headers)
    _write(p / "nodes_public.csv", nodes)
    _write(p / "adj_public.csv", padj)
    _write(p / "times_public.csv", ptimes)
    _write(p / "lengths_public.csv", plengths)
    _write(p / "schedules_public.csv", sch)
    _write(p / "transport_id_public.csv", ids)

    # ---- Population: home A (west) -> work B (east) -> home A ----------------------------
    a = (LAT + 0.0005, LON0)
    bb = (LAT + 0.0005, LON0 + (N_NODES - 1) * DLON)
    rows = {k: [] for k in ("lat", "lon", "t_start", "t_end", "prio")}
    for _ in range(n_agents):
        rows["lat"].append([a[0], bb[0], a[0]])
        rows["lon"].append([a[1], bb[1], a[1]])
        rows["t_start"].append([4.0, 9.0, 18.0])
        rows["t_end"].append([depart, 17.0, 27.75])
        rows["prio"].append([4, 0, 4])
    for name, r in rows.items():
        _write(pop / f"{name}.csv", r)

    # ---- Precomputed access/egress: trips separated by -1 (time column: -1 too) ---------
    walk_h = 0.06 / 4.0  # ~60 m walk at 4 km/h
    west, east = 0, N_NODES - 1
    # public: closest stop of each itinerary. Eastbound stops are nodes 0..3, westbound 4..7
    # (node 4 is at the east end, node 7 at the west end).
    pub_at = {"west": [0, 7], "east": [3, 4]}
    pre = pop / "distances_pre_computation"

    def pair(trip_nodes):
        n, t = [], []
        for j, tn in enumerate(trip_nodes):
            if j:
                n.append(-1)
                t.append(-1)
            n += tn
            t += [walk_h] * len(tn)
        return n, t

    trips = [("west", "east"), ("east", "west")]
    for mode, at in (("drive", {"west": [west], "east": [east]}),
                     ("bike", {"west": [west], "east": [east]}),
                     ("public", pub_at)):
        src = pair([at[o] for o, _ in trips])
        tgt = pair([at[t] for _, t in trips])
        _write(pre / f"{mode}_nodes_source.csv", [src[0]] * n_agents)
        _write(pre / f"{mode}_times_source.csv", [src[1]] * n_agents)
        _write(pre / f"{mode}_nodes_target.csv", [tgt[0]] * n_agents)
        _write(pre / f"{mode}_times_target.csv", [tgt[1]] * n_agents)
    return root


def write_network(root: Path, nodes, links, agents, far_stop=(41.7, -4.6)) -> Path:
    """Generic drive-only scenario (bike = same graph, public = one unreachable stop).

    nodes:  list of (lat, lon)
    links:  list of (u, v, minutes, lanes, capacity, priority)
    agents: list of (origin_node, destination_node, departure_hour); each makes one trip
            between points 50 m north of the two nodes.
    """
    n = len(nodes)
    out = {u: [] for u in range(n)}
    for link in links:
        out[link[0]].append(link)
    inn = {u: [] for u in range(n)}
    for (u, v, *_rest) in links:
        inn[v].append(u)

    def rows(k):  # attribute k of each outgoing link, -1 for nodes without links
        return [[lk[k] for lk in out[u]] or [-1] for u in range(n)]

    d = root / "net_files" / "drive_network"
    _write(d / "nodes_drive.csv", [[1000 + k, lat, lon] for k, (lat, lon) in enumerate(nodes)])
    _write(d / "adj_drive.csv", rows(1))
    _write(d / "inv_adj_drive.csv", [inn[u] or [-1] for u in range(n)])
    _write(d / "times_drive.csv", rows(2))
    _write(d / "lengths_drive.csv", [[300.0] * len(out[u]) or [-1] for u in range(n)])
    _write(d / "lanes_drive.csv", rows(3))
    _write(d / "capacity_drive.csv", rows(4))
    _write(d / "velocity_drive.csv", [[50] * len(out[u]) or [-1] for u in range(n)])
    _write(d / "priority_drive.csv", rows(5))

    b = root / "net_files" / "bike_network"
    _write(b / "nodes_bike.csv", [[2000 + k, lat, lon] for k, (lat, lon) in enumerate(nodes)])
    _write(b / "adj_bike.csv", rows(1))
    _write(b / "times_bike.csv", rows(2))
    _write(b / "cycleway_bike.csv", [[0] * len(out[u]) or [-1] for u in range(n)])

    p = root / "net_files" / "public_network"
    _write(p / "nodes_public.csv", [[0, 0, 0, far_stop[0], far_stop[1], "L0", 0.0, 0]])
    _write(p / "adj_public.csv", [[-1]])
    _write(p / "times_public.csv", [[-1]])
    _write(p / "lengths_public.csv", [[-1]])
    _write(p / "schedules_public.csv", [[5.0]])
    _write(p / "transport_id_public.csv", [[0]])

    pop = root / "pop_files"
    pt = lambda k: (nodes[k][0] + 0.00045, nodes[k][1])  # noqa: E731  (~50 m north)
    _write(pop / "lat.csv", [[pt(o)[0], pt(t)[0]] for o, t, _ in agents])
    _write(pop / "lon.csv", [[pt(o)[1], pt(t)[1]] for o, t, _ in agents])
    _write(pop / "t_start.csv", [[4.0, dep + 1.0] for _, _, dep in agents])
    _write(pop / "t_end.csv", [[dep, 27.75] for _, _, dep in agents])
    _write(pop / "prio.csv", [[4, 4] for _ in agents])
    pre = pop / "distances_pre_computation"
    walk_h = 0.05 / 4.0
    for mode in ("drive", "bike"):
        _write(pre / f"{mode}_nodes_source.csv", [[o] for o, _, _ in agents])
        _write(pre / f"{mode}_times_source.csv", [[walk_h] for _ in agents])
        _write(pre / f"{mode}_nodes_target.csv", [[t] for _, t, _ in agents])
        _write(pre / f"{mode}_times_target.csv", [[walk_h] for _ in agents])
    # The only public stop is far away (10 h walk): public transport is never better than walking.
    for side in ("source", "target"):
        _write(pre / f"public_nodes_{side}.csv", [[0] for _ in agents])
        _write(pre / f"public_times_{side}.csv", [[10.0] for _ in agents])
    return root
