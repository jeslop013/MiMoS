"""A synthetic example town in the exact MiMoS input format: try the model, and test it, without
real data.

    uv run mimos demo                                   # generate, run, emissions and report in ~1 minute
    uv run mimos example --out examples/town            # or step by step: writes the data and config.toml,
    uv run mimos run examples/town/config.toml          #   then run it like any scenario

Everything is generated from a seed (same seed, same town), in seconds; nothing is downloaded and
nothing needs to be committed. The town is made up and placed at latitude 0, longitude 0, so no
real place is implied (draw its maps with the model's own roads: ``mimos report --no-tiles``).

What it contains, chosen so that every mechanism of the model is used:

* **Roads** (drive network): an organic street network of about 700 junctions in a disc of 3.5
  km radius: an irregular **old core** of short, narrow streets (20 km/h), **grid-pattern
  suburbs** at different orientations, and sparser streets towards the edge. On top of it a
  hierarchy: a **ring road** (70 km/h, 2 lanes), **radial arterials** from the centre to the ring
  and a **middle orbital road** (50 km/h, 2 lanes, cycle lanes); everything else is residential
  (30 km/h, 1 lane). A **river** meanders across the town and has only five **bridges**: two on
  the ring road, two arterial bridges, and one old **single-lane bridge** next to the old core,
  the bottleneck where queues form at the peaks. Two **roundabouts** (circulating traffic has
  priority). A regional road (90 km/h) to a **village** 10 km east of the centre.
* **Bikes**: the same streets in both directions, cycle lanes on the arterials, and a bike-only
  bridge near the old core.
* **Public transport**: five urban bus lines through the centre, a circular line on the middle
  orbital road (all following the roads, both directions, every 8-10 min at the peaks and 15 min
  otherwise, 06:00-23:00), a regional bus to the village every 30 min, eleven rental-bike
  stations, and walking transfers between lines.
* **Population**: workers, students and other residents with daily agendas (home -> activities ->
  home). Jobs, shops and leisure are mostly in the old core, with a business park south of the
  river and a campus in the north-west; homes are spread over the suburbs, on both banks, so the
  bridges reach capacity at the morning peak and queues form upstream. A tenth live in the
  village; 1% of workers work 60 km away (far destinations like those of a real population).
* **Access/egress candidates** (the "precomputed distances"): the 3 nearest drive and bike nodes
  of every agenda location, and the nearest stop of every public-transport itinerary and the
  nearest rental-bike station.

``capacity_scale`` (default 0.1) makes the agents a sample: the default 5,000 agents with scale
0.1 behave like a town of 50,000 (road and vehicle capacities scaled, as MATSim does), which keeps
the example fast (10 iterations in ~10 s) while producing congestion.
"""

from __future__ import annotations

import csv
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

M_PER_DEG = 111_320.0  # metres per degree of latitude (and of longitude at the equator)
CITY_R = 3_500.0       # radius of the built-up area, m
RING_R = 3_000.0       # radius of the ring road, m
MID_R = 1_500.0        # radius of the middle orbital road, m
CORE_R = 650.0         # radius of the old core, m
VILLAGE = (10_000.0, 1_000.0)  # village centre (x, y) in m
FAR_JOB = (60_000.0, 0.0)      # far destination for 1% of workers

# grid-pattern suburbs: (centre x, centre y, width, height, rotation in degrees, street spacing), m
PATCHES = [(-1900.0, 1800.0, 1300.0, 1000.0, 18.0, 170.0),    # north-west
           (1950.0, -1900.0, 1400.0, 900.0, -12.0, 230.0),    # south-east (business park)
           (1700.0, 1700.0, 1100.0, 900.0, 35.0, 190.0)]      # north-east
# bridge targets on the river: (x, kind); "ring" bridges carry the ring road, "old" is the single-lane bottleneck
BRIDGES = [(-2960.0, "ring"), (-1700.0, "art"), (-200.0, "old"), (1100.0, "art"), (2960.0, "ring")]
BIKE_BRIDGE_X = -900.0          # a bike-only bridge
BUS_STOP_SPACING = 450.0        # m along a line between two stops

PRIO_RING, PRIO_ART, PRIO_REG, PRIO_RES, PRIO_ROUNDABOUT = 2, 3, 4, 6, 0  # OSM-like road classes (lower = higher priority)
CAR_LENGTH = 4.5  # m per car in the storage capacity


def river_y(x):
    """y of the river (m) at x: it meanders west to east just south of the old core."""
    return 300.0 * np.sin(np.asarray(x) / 1200.0) - 250.0


def _ll(x: float, y: float) -> tuple[float, float]:
    """(x, y) in metres around (0, 0) -> (lat, lon)."""
    return y / M_PER_DEG, x / M_PER_DEG


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def _rows_or_empty(rows):
    return [r if r else [-1] for r in rows]


# ---- graphs --------------------------------------------------------------------------------

@dataclass
class Graph:
    xy: list = field(default_factory=list)        # node -> (x, y) m
    links: dict = field(default_factory=dict)     # (u, v) -> attributes

    def node(self, x: float, y: float) -> int:
        self.xy.append((x, y))
        return len(self.xy) - 1

    def link(self, u: int, v: int, **attr) -> None:
        self.links[(u, v)] = attr

    def both(self, u: int, v: int, **attr) -> None:
        self.link(u, v, **attr)
        self.link(v, u, **attr)

    def length(self, u: int, v: int) -> float:
        (x0, y0), (x1, y1) = self.xy[u], self.xy[v]
        return math.hypot(x1 - x0, y1 - y0)

    def out(self) -> list[list[int]]:
        o = [[] for _ in self.xy]
        for u, v in self.links:
            o[u].append(v)
        return o


# ---- the street network ----------------------------------------------------------------------

def _spacing(r):
    """Typical distance between neighbouring junctions at distance r from the centre: short in the old core."""
    return 95.0 + 190.0 * (np.asarray(r) / CITY_R) ** 1.3


def _points(rng) -> np.ndarray:
    """Junction positions: lattice points of the suburbs, then random points (dart throwing) whose
    minimum distance grows with the distance from the centre."""
    pts = []
    for cx, cy, w, h, ang, sp in PATCHES:
        a = math.radians(ang)
        gx, gy = np.meshgrid(np.arange(-w / 2, w / 2 + 1e-9, sp), np.arange(-h / 2, h / 2 + 1e-9, sp))
        local = np.c_[gx.ravel(), gy.ravel()]
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        pts.append(local @ rot.T + [cx, cy] + rng.uniform(-4.0, 4.0, local.shape))
    acc = np.vstack(pts)
    r = CITY_R * np.sqrt(rng.random(30_000))
    th = rng.random(30_000) * 2 * math.pi
    cand = np.c_[r * np.cos(th), r * np.sin(th)]
    out = [acc]
    cur = acc
    for c in cand:
        d = np.hypot(cur[:, 0] - c[0], cur[:, 1] - c[1])
        if d.min() > _spacing(math.hypot(c[0], c[1])):
            cur = np.vstack([cur, c])
    return cur


def _gabriel(P: np.ndarray, k: int = 8, tol: float = 1.04) -> list:
    """Planar street-like graph: i-j is an edge if no other point lies (almost) inside the circle with
    diameter i-j (Gabriel graph; tol > 1 also drops the diagonals of lattices)."""
    n = len(P)
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    nn = np.argsort(D, axis=1)[:, 1:k + 1]
    edges = set()
    for i in range(n):
        for j in nn[i]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            if (a, b) in edges:
                continue
            m = (P[a] + P[b]) / 2
            r2 = D[a, b] ** 2 / 4
            near = np.union1d(nn[a], nn[b])
            near = near[(near != a) & (near != b)]
            if np.any((P[near, 0] - m[0]) ** 2 + (P[near, 1] - m[1]) ** 2 < r2 * tol ** 2):
                continue
            if D[a, b] > 2.3 * _spacing(math.hypot(m[0], m[1])):
                continue  # long edges at the border of the point cloud
            edges.add((a, b))
    return sorted(edges)


class _Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        self.p[a] = b
        return True


def _dijkstra(adj, src, dst):
    """Shortest path (by length) between two junctions: list of node ids."""
    import heapq
    best = {src: 0.0}
    prev = {}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            break
        if d > best.get(u, 1e18):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < best.get(v, 1e18):
                best[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


@dataclass
class City:
    drive: Graph
    bike: Graph
    lines: list
    rental: list
    base_xy: np.ndarray      # junctions of the town (not the regional road or the village)
    schools: np.ndarray


def build_city(rng: np.random.Generator) -> City:
    P = _points(rng)
    edges = _gabriel(P)
    north = P[:, 1] > river_y(P[:, 0])
    # the river: keep only the crossing edge nearest to each bridge target
    crossing = [e for e in edges if north[e[0]] != north[e[1]]]
    mid = lambda e: (P[e[0]] + P[e[1]]) / 2  # noqa: E731
    bridge_edge = {}
    for bx, kind in BRIDGES:
        target = np.array([bx, float(river_y(bx))])
        e = min((e for e in crossing if e not in bridge_edge), key=lambda e: np.hypot(*(mid(e) - target)))
        bridge_edge[e] = kind
    crossing_set = set(crossing)
    edges = [e for e in edges if e not in crossing_set or e in bridge_edge]
    # random spanning tree + most of the other edges: connected, with dead ends and irregular blocks
    order = list(bridge_edge) + [edges[i] for i in rng.permutation(len(edges)) if edges[i] not in bridge_edge]
    keep_p = rng.random(len(order))
    uf = _Union(len(P))
    kept = []
    for e, p in zip(order, keep_p):
        if e in bridge_edge or uf.join(e[0], e[1]) or p < 0.88:
            kept.append(e)
    uf2 = _Union(len(P))
    for a, b in kept:
        uf2.join(a, b)
    centre0 = int(np.argmin(np.hypot(P[:, 0], P[:, 1])))
    comp = np.array([uf2.find(i) == uf2.find(centre0) for i in range(len(P))])
    new = -np.ones(len(P), dtype=int)
    new[comp] = np.arange(comp.sum())
    xy = P[comp]
    n = len(xy)
    E = {}
    for a, b in kept:
        if comp[a] and comp[b]:
            E[(int(new[a]), int(new[b]))] = dict(speed=30.0, lanes=1, prio=PRIO_RES, cyc=0)
    bridges = {(int(new[a]), int(new[b])): kind for (a, b), kind in bridge_edge.items() if comp[a] and comp[b]}
    for (a, b), at in E.items():
        r = math.hypot(*((xy[a] + xy[b]) / 2))
        if r < CORE_R:
            at["speed"] = 20.0  # narrow streets of the old core
    length = lambda a, b: float(np.hypot(*(xy[a] - xy[b])))  # noqa: E731
    adj = [[] for _ in range(n)]
    for (a, b) in E:
        adj[a].append((b, length(a, b)))
        adj[b].append((a, length(a, b)))

    def nearest(x, y):
        return int(np.argmin(np.hypot(xy[:, 0] - x, xy[:, 1] - y)))

    def classify(path, speed, lanes, prio, cyc):
        for a, b in zip(path, path[1:]):
            key = (a, b) if (a, b) in E else (b, a)
            E[key].update(speed=speed, lanes=lanes, prio=prio, cyc=cyc)

    c0 = nearest(0.0, 0.0)
    ang = lambda k, m: (2 * math.pi * k / m)  # noqa: E731
    mid_a = [nearest(MID_R * math.cos(ang(k, 8)), MID_R * math.sin(ang(k, 8))) for k in range(8)]
    ring_a = [nearest(RING_R * math.cos(ang(k, 8)), RING_R * math.sin(ang(k, 8))) for k in range(8)]
    orbit = []
    for k in range(8):
        orbit += _dijkstra(adj, mid_a[k], mid_a[(k + 1) % 8])[:-1]
    orbit.append(orbit[0])
    ringp = []
    for k in range(8):
        ringp += _dijkstra(adj, ring_a[k], ring_a[(k + 1) % 8])[:-1]
    ringp.append(ringp[0])
    radial = [_dijkstra(adj, c0, ring_a[k]) for k in range(8)]
    classify(orbit, 50.0, 2, PRIO_ART, 1)
    for p in radial:
        classify(p, 50.0, 2, PRIO_ART, 1)
    classify(ringp, 70.0, 2, PRIO_RING, 0)
    for e, kind in bridges.items():  # bridge classes override the paths: the old bridge stays the bottleneck
        if kind == "old":
            E[e].update(speed=30.0, lanes=1, prio=PRIO_RES, cyc=0)
        elif kind == "art":
            E[e].update(speed=50.0, lanes=2, prio=PRIO_ART, cyc=1)
        else:
            E[e].update(speed=70.0, lanes=2, prio=PRIO_RING, cyc=0)

    # roundabouts at the centre and on the middle orbital road (east)
    round_nodes = {c0, mid_a[0]}
    drive = Graph()
    gid, rings = {}, {}
    for i in range(n):
        if i not in round_nodes:
            gid[i] = drive.node(float(xy[i, 0]), float(xy[i, 1]))
    dirs = {"E": (30, 0), "N": (0, 30), "W": (-30, 0), "S": (0, -30)}
    for c in sorted(round_nodes):
        rings[c] = {d: drive.node(xy[c, 0] + dx, xy[c, 1] + dy) for d, (dx, dy) in dirs.items()}
        for a, b in (("E", "N"), ("N", "W"), ("W", "S"), ("S", "E")):  # counter-clockwise, circulating traffic first
            drive.link(rings[c][a], rings[c][b], speed=30, lanes=1, prio=PRIO_ROUNDABOUT)

    def at(i, j):  # drive node of junction i for the street that goes to junction j
        if i not in rings:
            return gid[i]
        dx, dy = xy[j] - xy[i]
        return rings[i]["ENWS"[int(((math.degrees(math.atan2(dy, dx)) + 45.0) % 360.0) // 90)]]

    for (a, b), at_ in E.items():
        drive.both(at(a, b), at(b, a), speed=at_["speed"], lanes=at_["lanes"], prio=at_["prio"])
    # regional road from the east ring anchor to the village, a node every ~1 km, then 3 x 3 village streets
    reg, vil = _regional(drive, gid[ring_a[0]], PRIO_REG, drive_attrs=True)
    # bikes: the same streets both ways, cycle lanes on arterials, a bike-only bridge, the regional road
    bike = Graph()
    bid = {i: bike.node(float(xy[i, 0]), float(xy[i, 1])) for i in range(n)}
    for (a, b), at_ in E.items():
        bike.both(bid[a], bid[b], cycleway=at_["cyc"])
    south = np.flatnonzero(~(xy[:, 1] > river_y(xy[:, 0])))
    north_i = np.flatnonzero(xy[:, 1] > river_y(xy[:, 0]))
    target = np.array([BIKE_BRIDGE_X, float(river_y(BIKE_BRIDGE_X))])
    ns = min(north_i, key=lambda i: np.hypot(*(xy[i] - target)))
    sn = min(south, key=lambda i: np.hypot(*(xy[i] - xy[ns])))
    bike.both(bid[ns], bid[sn], cycleway=1)
    _regional(bike, bid[ring_a[0]], None, drive_attrs=False)

    # bus lines: shortest paths on the street graph, a stop every ~450 m
    def stops_along(path):
        pts = [xy[i] for i in path]
        out, last = [tuple(map(float, pts[0]))], 0.0
        cum = 0.0
        for p, q in zip(pts, pts[1:]):
            cum += float(np.hypot(*(p - q)))
            if cum - last >= BUS_STOP_SPACING:
                out.append(tuple(map(float, q)))
                last = cum
        if out[-1] != tuple(map(float, pts[-1])):
            out.append(tuple(map(float, pts[-1])))
        return out

    def through(k0, k1):
        return _dijkstra(adj, ring_a[k0], c0)[:-1] + _dijkstra(adj, c0, ring_a[k1])

    # the point of the regional road ~4 km out
    rx = [xy_ for xy_ in reg]
    lines = [
        Line("L1", 0, stops_along(through(3, 7)), 18.0, _urban_headway),     # north-west - south-east
        Line("L2", 0, stops_along(through(1, 5)), 18.0, _urban_headway),     # north-east - south-west
        Line("L3", 0, stops_along(orbit), 18.0, lambda h: 10.0 if 7.0 <= h < 9.5 or 16.5 <= h < 19.5 else 15.0),  # circular
        Line("L4", 0, stops_along(through(2, 6)), 18.0, _urban_headway),     # north - south
        Line("L5", 0, stops_along(through(4, 0)), 18.0, _urban_headway),     # west - east
        Line("R1", 1, [tuple(map(float, xy[c0])), tuple(map(float, xy[ring_a[0]])), rx[len(rx) // 2], rx[-1], VILLAGE], 40.0, lambda h: 30.0),
    ]
    rental = [(0.0, 0.0)] + [(600.0 * math.cos(ang(k, 6)), 600.0 * math.sin(ang(k, 6))) for k in range(6)] \
        + [(1500.0 * math.cos(ang(k, 4) + math.pi / 4), 1500.0 * math.sin(ang(k, 4) + math.pi / 4)) for k in range(4)]
    schools = np.array([xy[nearest(*p)] for p in ((-2200, 1800), (900, 1000), (-1000, -800), (1500, -1200), (300, 2000))])
    return City(drive, bike, lines, rental, xy, schools)


def _regional(g: Graph, start: int, prio, drive_attrs: bool):
    """Regional road from node `start` to the village (a node about every km) and the 3 x 3 village streets.
    Returns the (x, y) of the regional road nodes (for the bus line) and the village nodes."""
    x0, y0 = g.xy[start]
    vx, vy = VILLAGE
    k = int((vx - 600 - x0) // 1000)
    prev, road = start, []
    for t in range(1, k + 1):
        x, y = x0 + (vx - 600 - x0) * t / k, y0 + (vy - y0) * t / k
        n = g.node(float(x), float(y))
        g.both(prev, n, **(dict(speed=90, lanes=1, prio=prio) if drive_attrs else dict(cycleway=0)))
        prev = n
        road.append((float(x), float(y)))
    vil = {(i, j): g.node(vx + (j - 1) * 200, vy + (i - 1) * 200) for i in range(3) for j in range(3)}
    street = dict(speed=30, lanes=1, prio=PRIO_RES) if drive_attrs else dict(cycleway=0)
    for i in range(3):
        for j in range(3):
            if j < 2:
                g.both(vil[(i, j)], vil[(i, j + 1)], **street)
            if i < 2:
                g.both(vil[(i, j)], vil[(i + 1, j)], **street)
    g.both(prev, vil[(1, 0)], **(dict(speed=50, lanes=1, prio=prio) if drive_attrs else dict(cycleway=0)))
    return road, vil


def write_drive(g: Graph, d: Path) -> None:
    out = g.out()
    inn = [[] for _ in g.xy]
    for u, v in g.links:
        inn[v].append(u)
    attr = lambda f: _rows_or_empty([[f(u, v, g.links[(u, v)]) for v in out[u]] for u in range(len(g.xy))])  # noqa: E731
    _write(d / "nodes_drive.csv", [[10_000 + k, *_ll(x, y)] for k, (x, y) in enumerate(g.xy)])
    _write(d / "adj_drive.csv", _rows_or_empty(out))
    _write(d / "inv_adj_drive.csv", _rows_or_empty(inn))
    _write(d / "lengths_drive.csv", attr(lambda u, v, a: round(g.length(u, v), 2)))
    _write(d / "times_drive.csv", attr(lambda u, v, a: round(g.length(u, v) / 1000 / a["speed"] * 60, 5)))
    _write(d / "lanes_drive.csv", attr(lambda u, v, a: a["lanes"]))
    _write(d / "capacity_drive.csv", attr(lambda u, v, a: max(1, int(a["lanes"] * g.length(u, v) / CAR_LENGTH))))
    _write(d / "velocity_drive.csv", attr(lambda u, v, a: a["speed"]))
    _write(d / "priority_drive.csv", attr(lambda u, v, a: a["prio"]))


def write_bike(g: Graph, d: Path) -> None:
    out = g.out()
    attr = lambda f: _rows_or_empty([[f(u, v, g.links[(u, v)]) for v in out[u]] for u in range(len(g.xy))])  # noqa: E731
    _write(d / "nodes_bike.csv", [[20_000 + k, *_ll(x, y)] for k, (x, y) in enumerate(g.xy)])
    _write(d / "adj_bike.csv", _rows_or_empty(out))
    _write(d / "times_bike.csv", attr(lambda u, v, a: round(g.length(u, v) / 1000 / 15 * 60, 5)))  # 15 km/h
    _write(d / "cycleway_bike.csv", attr(lambda u, v, a: a["cycleway"]))


# ---- public transport ------------------------------------------------------------------------

@dataclass
class Line:
    name: str
    mode: int                 # 0 urban bus, 1 regional bus
    stops: list               # (x, y) of the stops, in order (first direction)
    speed: float              # km/h between stops, dwell included
    headway: callable         # hour -> minutes between departures


def _urban_headway(h: float) -> float:
    return 8.0 if 7.0 <= h < 9.5 or 16.5 <= h < 19.5 else 15.0


def write_public(d: Path, lines: list, rental: list) -> dict:
    """Writes the public network; returns the (x, y) and itinerary of every node."""
    nodes, sch, ids = [], [], []
    xy, itin_of = [], []
    # rental-bike stations first (mode -1, itinerary -1)
    for k, (x, y) in enumerate(rental):
        nodes.append([-1, -1, k, *_ll(x, y), "rent_bike", 0.0, 0])
        xy.append((x, y))
        itin_of.append(-1)
        sch.append([-1])  # no timetable: rental bikes are always available
        ids.append([-1])
    itin = 0
    vehicle = 0
    stops_of_itin = {}
    for line in lines:
        for direction in (line.stops, line.stops[::-1]):
            # departures from the first stop
            deps, h = [], 6.0
            while h <= 23.0:
                deps.append(h)
                h += line.headway(h) / 60.0
            offsets = [0.0]
            for (x0, y0), (x1, y1) in zip(direction, direction[1:]):
                offsets.append(offsets[-1] + math.hypot(x1 - x0, y1 - y0) / 1000 / line.speed)
            first = len(nodes)
            for s, (x, y) in enumerate(direction):
                nodes.append([line.mode, itin, s, *_ll(x, y), f"{line.name}{'ab'[direction is not line.stops]}", 0.0, 0])
                xy.append((x, y))
                itin_of.append(itin)
                sch.append([round(t + offsets[s], 6) for t in deps])
                ids.append([vehicle + j for j in range(len(deps))])
            stops_of_itin[itin] = list(range(first, len(nodes)))
            vehicle += len(deps)
            itin += 1
    n = len(nodes)
    P = np.array(xy)
    adj, times, lengths = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]

    def add(u, v, minutes, metres):
        adj[u].append(v)
        times[u].append(round(minutes, 5))
        lengths[u].append(round(metres, 2))

    # next stop first (the first adjacency of a stop is its next stop on the same itinerary)
    for it, stops in stops_of_itin.items():
        for a, b in zip(stops, stops[1:]):
            add(a, b, (sch[b][0] - sch[a][0]) * 60, math.dist(xy[a], xy[b]))
    # walking transfers: nearest stop of every other itinerary within 800 m, rental stations within 400 m
    for u in range(len(rental), n):
        dist = np.hypot(*(P - P[u]).T)
        for it, stops in stops_of_itin.items():
            if it == itin_of[u]:
                continue
            v = min(stops, key=lambda s: dist[s])
            if dist[v] <= 800:
                add(u, v, dist[v] * 1.3 / 1000 / 4 * 60, dist[v] * 1.3)
        for v in range(len(rental)):
            if dist[v] <= 400:
                add(u, v, dist[v] * 1.3 / 1000 / 4 * 60, dist[v] * 1.3)
    # rental stations: ride to any other station (15 km/h, 20% detour)
    for u in range(len(rental)):
        for v in range(len(rental)):
            if u != v:
                m = math.dist(xy[u], xy[v]) * 1.2
                add(u, v, m / 1000 / 15 * 60, m)
    _write(d / "nodes_public.csv", nodes)
    _write(d / "adj_public.csv", _rows_or_empty(adj))
    _write(d / "times_public.csv", _rows_or_empty(times))
    _write(d / "lengths_public.csv", _rows_or_empty(lengths))
    _write(d / "schedules_public.csv", sch)
    _write(d / "transport_id_public.csv", ids)
    return {"xy": P, "itin": np.array(itin_of)}


# ---- population ------------------------------------------------------------------------------

def _home_weights(city: City) -> tuple[np.ndarray, np.ndarray]:
    """Candidate home points (junctions) and weights: denser towards the centre, a little more on the north bank."""
    P = city.base_xy
    r = np.hypot(P[:, 0], P[:, 1]) / 1000
    w = np.exp(-r / 2.0) * np.where(P[:, 1] > river_y(P[:, 0]), 1.4, 1.0)
    return P, w / w.sum()


def _jitter(rng, n, scale=110.0):
    return rng.uniform(-scale, scale, size=(n, 2))


def make_population(n: int, rng: np.random.Generator, city: City) -> dict:
    """Agendas as lists per agent: x, y (m), t_start, t_end (h), prio, tag."""
    home_pts, home_w = _home_weights(city)
    vx, vy = VILLAGE
    in_village = rng.random(n) < 0.10
    homes = home_pts[rng.choice(len(home_pts), size=n, p=home_w)] + _jitter(rng, n)
    homes[in_village] = np.array([vx, vy]) + rng.uniform(-200, 200, size=(in_village.sum(), 2))

    def downtown(k):  # the old core, north bank
        r = CORE_R * np.sqrt(rng.random(k))
        th = rng.random(k) * 2 * math.pi
        return np.c_[r * np.cos(th), r * np.sin(th)]

    industry = lambda k: rng.uniform([1300, -2350], [2600, -1450], size=(k, 2))  # noqa: E731  (business park, south of the river)
    anywhere = lambda k: home_pts[rng.choice(len(home_pts), size=k)] + _jitter(rng, k)  # noqa: E731
    schools = city.schools

    kind = rng.choice(["worker", "student", "other"], size=n, p=[0.55, 0.20, 0.25])
    agents = []
    for i in range(n):
        h = tuple(homes[i])
        locs, starts, ends, prios, tags = [h], [4.0], [], [4], ["home"]
        free = [4.0]  # when the current activity ends

        def travel(a, b):  # rough door-to-door estimate, h
            return math.dist(a, b) / 1000 / 18 + 0.1

        def go(to, start, duration, prio, tag):
            """Leave the current place to start the next activity at `start`. Leaves when the
            current activity ends if that is later (arriving late); goes home in between when
            there would be more than 1.5 h to kill elsewhere."""
            here = locs[-1]
            leave = start - travel(here, to) - rng.uniform(0, 0.25)
            if tag != "home" and here != h and leave > free[0] + 1.5:
                go(h, free[0] + travel(here, h), leave - free[0] - travel(here, h), 4, "home")
                here = h
                leave = start - travel(here, to) - rng.uniform(0, 0.25)
            ends.append(round(max(free[0], starts[-1] + 0.25, leave), 4))
            locs.append(tuple(to))
            starts.append(start)
            prios.append(prio)
            tags.append(tag)
            free[0] = start + duration
            return free[0]

        if kind[i] == "worker":
            u = rng.random()
            if u < 0.01:
                job = FAR_JOB
            elif u < 0.50:
                job = downtown(1)[0]
            elif u < 0.80:
                job = industry(1)[0]
            else:
                job = anywhere(1)[0]
            t = go(job, float(np.clip(rng.normal(8.25, 0.6), 6.5, 10.5)), float(np.clip(rng.normal(8.5, 0.75), 6, 10)), 0, "work")
            if rng.random() < 0.35:
                t = go(downtown(1)[0] if rng.random() < 0.6 else anywhere(1)[0], t + 0.3, rng.uniform(0.5, 1.5), 2, "shop")
        elif kind[i] == "student":
            school = schools[np.argmin(np.hypot(*(schools - h).T))] if rng.random() < 0.7 else schools[rng.integers(len(schools))]
            t = go(school, float(np.clip(rng.normal(8.5, 0.2), 8.0, 9.25)), rng.uniform(5.5, 6.5), 0, "study")
            if rng.random() < 0.3:
                t = go(downtown(1)[0], t + rng.uniform(1.5, 3.0), rng.uniform(1.0, 2.0), 3, "leisure")
        else:
            t = 4.0
            if rng.random() < 0.8:
                t = go(anywhere(1)[0], rng.uniform(9.5, 12.0), rng.uniform(0.5, 1.5), 2, "shop")
            if rng.random() < 0.6 or len(locs) == 1:
                t = go(downtown(1)[0], max(t + 1.0, rng.uniform(16.5, 20.0)), rng.uniform(1.0, 2.0), 3, "leisure")
        go(h, t + travel(locs[-1], h), 0.0, 4, "home")
        ends.append(27.75)
        agents.append({"x": [p[0] for p in locs], "y": [p[1] for p in locs], "t_start": [round(s, 4) for s in starts],
                       "t_end": [round(e, 4) for e in ends], "prio": prios, "tag": tags})
    return agents


def _nearest(points: np.ndarray, targets: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k nearest targets of every point: (indices, distances in m)."""
    idx = np.empty((len(points), k), dtype=int)
    dst = np.empty((len(points), k))
    for a in range(0, len(points), 4096):  # chunks keep memory small for large towns
        d = np.hypot(points[a:a + 4096, None, 0] - targets[None, :, 0], points[a:a + 4096, None, 1] - targets[None, :, 1])
        part = np.argsort(d, axis=1)[:, :k]
        idx[a:a + 4096] = part
        dst[a:a + 4096] = np.take_along_axis(d, part, axis=1)
    return idx, dst


def write_population(agents: list, drive: Graph, bike: Graph, pub: dict, pop: Path) -> None:
    _write(pop / "lat.csv", [[round(_ll(x, y)[0], 7) for x, y in zip(a["x"], a["y"])] for a in agents])
    _write(pop / "lon.csv", [[round(_ll(x, y)[1], 7) for x, y in zip(a["x"], a["y"])] for a in agents])
    for key in ("t_start", "t_end", "prio"):
        _write(pop / f"{key}.csv", [a[key] for a in agents])
    _write(pop / "tags.csv", [a["tag"] for a in agents])  # documentation only (not read by the model)

    locs = np.array([(x, y) for a in agents for x, y in zip(a["x"], a["y"])])
    start = np.cumsum([0] + [len(a["x"]) for a in agents])
    cands = {}
    for mode, g, speed in (("drive", drive, 15.0), ("bike", bike, 4.0)):  # access speed, km/h (as the loader)
        idx, dst = _nearest(locs, np.array(g.xy), 3)
        cands[mode] = [(list(map(int, i)), [d / 1000 / speed for d in ds]) for i, ds in zip(idx, dst)]
    # public: the nearest stop of every itinerary and the nearest rental-bike station (walking 4 km/h)
    P, itin = pub["xy"], pub["itin"]
    groups = [np.flatnonzero(itin == g) for g in sorted(set(itin.tolist()))]
    pub_c = [([], []) for _ in range(len(locs))]
    for members in groups:
        i, d = _nearest(locs, P[members], 1)
        for k in range(len(locs)):
            pub_c[k][0].append(int(members[i[k, 0]]))
            pub_c[k][1].append(d[k, 0] / 1000 / 4.0)
    cands["public"] = pub_c

    pre = pop / "distances_pre_computation"
    for mode, c in cands.items():
        for side, offset in (("source", 0), ("target", 1)):
            nodes_rows, times_rows = [], []
            for a in range(len(agents)):
                n_rows, t_rows = [], []
                for trip in range(len(agents[a]["x"]) - 1):
                    loc = start[a] + trip + offset
                    if trip:
                        n_rows.append(-1)
                        t_rows.append(-1)
                    n_rows += c[loc][0]
                    t_rows += [round(t, 7) for t in c[loc][1]]
                nodes_rows.append(n_rows)
                times_rows.append(t_rows)
            _write(pre / f"{mode}_nodes_{side}.csv", nodes_rows)
            _write(pre / f"{mode}_times_{side}.csv", times_rows)


# ---- entry point -----------------------------------------------------------------------------

def town(out: Path | str = "examples/town", n_agents: int = 5000, seed: int = 0, capacity_scale: float = 0.1,
         iterations: int = 10) -> Path:
    """Writes the example town (data + ``config.toml``) under ``out`` and returns the config path.

    n_agents: simulated agents. capacity_scale: the sample share they represent (0.1: a town of
    10 x n_agents inhabitants on roads and buses scaled to 10%; 1: the agents are everybody).
    """
    out = Path(out)
    rng = np.random.default_rng(seed)
    city = build_city(np.random.default_rng([seed, 1]))   # the streets depend on the seed too
    write_drive(city.drive, out / "net_files" / "drive_network")
    write_bike(city.bike, out / "net_files" / "bike_network")
    pub = write_public(out / "net_files" / "public_network", city.lines, city.rental)
    write_population(make_population(n_agents, rng, city), city.drive, city.bike, pub, out / "pop_files")

    with (Path(__file__).parent / "town.toml").open("rb") as f:
        template = f.read().decode("utf-8")
    header = (f"# Example town generated by mimos.scenarios.town(n_agents={n_agents}, seed={seed}, "
              f"capacity_scale={capacity_scale}).\n")
    run = (f'data_dir = "./"\nrun_dir = "run/"\nn_agents = {n_agents}\n'
           f"capacity_scale = {capacity_scale}\n")
    text = template.replace("[run]\n", "[run]\n" + run, 1).replace("it_number = 10", f"it_number = {iterations}", 1)
    text = header + text + '\n[view]\ncenter_lat = 0.0\ncenter_lon = 0.0\nradius_km = 3.7\n'
    tomllib.loads(text)  # fail here, not in the model, if the template is broken
    cfg = out / "config.toml"
    cfg.write_text(text, encoding="utf-8")
    return cfg
