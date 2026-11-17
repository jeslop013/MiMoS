"""MP4 videos of a MiMoS run, for presentations (too heavy for the HTML report).

    uv run mimos video <run_dir> [--what vehicles traffic emissions] [--start 5 --end 24]

    uv run mimos video <run_dir> --what vehicles --start 08:00 --end 08:05 --speed 10 --tag _slow

Writes <run_dir>/videos/*.mp4 (1920x1080, H.264). --speed is simulated seconds per second of
video (1 = real time); --start/--end are clock times (HH:MM) of the simulated day.
Videos:
  vehicles.mp4   simulated cars moving along the real road shapes (drive_links.npz) and the
                 public transport vehicles running their timetables (public network files),
                 coloured by occupancy (% of capacity, from the executed public trajectories),
                 and the cyclists along their bike-network routes (exec_bike_*.csv)
  traffic.mp4    flow on every link as % of its flow capacity (lanes x 3600 / t_s), moving one-hour
                 window through the day; green = spare capacity, red = at or over capacity
  emissions.mp4  NOx in g per km per hour through the day (needs `mimos emissions` first)

Frames are drawn with matplotlib (Agg) on the model's own road network (main roads, dark theme,
no map tiles) and piped to ffmpeg (imageio-ffmpeg). Rendering is kept cheap: the background is
drawn once and reused (blitting), and each frame only redraws the moving points or recolours a
fixed set of links.

Cars are the SIMULATED SAMPLE: each dot stands for about 1/capacity_scale real cars (shown on
the video). Public transport vehicles are all scheduled vehicles.
"""

from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

import numpy as np

from mimos.networks import load_drive_links, load_public_nodes, read_trajectories

W, H, DPI = 1920, 1080, 100
BG, ROAD, CAR, BUS, TEXT = "#0d1117", "#30363d", "#ff7b54", "#58a6ff", "#e6edf3"
BUS_CMAP = "cool"  # public transport occupancy: cyan (empty) -> magenta (full)
BIKE = "#7ee787"
VIEW = {"center": None, "radius_km": 6.0}  # set by make_videos from [view] (radius = half-height)
TAG = {"tag": ""}  # set by make_videos: suffix of the file names


def _out(results: Path, name: str) -> Path:
    """<results>/videos/<name><tag>.mp4; a tagged clip never touches the untagged video."""
    out = results / "videos" / f"{name}{TAG['tag']}.mp4"
    out.parent.mkdir(exist_ok=True)
    return out
BACKDROP_MAX_PRIORITY = 5.5  # main roads (not residential) as background
BACKDROP_ALL_ROADS = 20_000  # small networks (e.g. the example town): draw every road as background


# ---- geometry helpers --------------------------------------------------------------------------

class LinkPaths:
    """Interpolates positions along the real link shapes, vectorised.

    For every link, the cumulative length along its shape points is stored in one global array
    shifted by link index (G = cumulative + k * BIG), so a single searchsorted finds the segment
    of many (link, fraction) pairs at once.
    """

    BIG = 1e3  # larger than any link length in the scaled-degree metric used here

    def __init__(self, links):
        self.links = links
        if links.geom_start is not None:
            start, lon, lat = links.geom_start, links.geom_lon.astype(float), links.geom_lat.astype(float)
        else:  # straight links: two points each
            n = len(links.u)
            start = np.arange(0, 2 * n + 1, 2)
            lon = np.column_stack([links.lon[links.u], links.lon[np.maximum(links.v, 0)]]).ravel()
            lat = np.column_stack([links.lat[links.u], links.lat[np.maximum(links.v, 0)]]).ravel()
        self.start, self.lon, self.lat = start, lon, lat
        coslat = np.cos(np.radians(np.nanmedian(lat)))
        seg = np.hypot(np.diff(lon) * coslat, np.diff(lat))
        link_of_point = np.repeat(np.arange(len(start) - 1), np.diff(start))
        seg[np.diff(link_of_point) != 0] = 0.0  # no length across link boundaries
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        first = start[:-1]
        cum_in_link = cum - np.repeat(cum[first], np.diff(start))  # starts at 0 for each link
        self.total = cum[start[1:] - 1] - cum[first]
        self.G = cum_in_link + link_of_point * self.BIG

    def position(self, k: np.ndarray, frac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = np.clip(frac, 0, 1) * self.total[k] + k * self.BIG
        j = np.clip(np.searchsorted(self.G, target, side="right") - 1, self.start[k], self.start[k + 1] - 2)
        j = np.maximum(j, self.start[k])
        one_point = self.start[k + 1] - self.start[k] < 2
        j = np.where(one_point, self.start[k], j)
        j2 = np.where(one_point, j, j + 1)
        span = self.G[j2] - self.G[j]
        w = np.where(span > 0, (target - self.G[j]) / np.where(span > 0, span, 1), 0.0)
        return self.lon[j] + w * (self.lon[j2] - self.lon[j]), self.lat[j] + w * (self.lat[j2] - self.lat[j])


def _wmedian(x: np.ndarray, w: np.ndarray | None) -> float:
    if w is None:
        return float(np.median(x))
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])


def _view(links, k: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float, float, float]:
    """(lon0, lon1, lat0, lat1) of a 16:9 view: centred on the configured [view] centre, else on
    the traffic-weighted median; half-height = radius_km."""
    if VIEW["center"] is not None:
        clat, clon = VIEW["center"]
    elif k.size:
        clon, clat = _wmedian(links.lon[links.u[k]], weights), _wmedian(links.lat[links.u[k]], weights)
    else:
        clon, clat = float(np.nanmedian(links.lon)), float(np.nanmedian(links.lat))
    half_lat = VIEW["radius_km"] / 111.2
    half_lon = half_lat * (W / H) / np.cos(np.radians(clat))
    return clon - half_lon, clon + half_lon, clat - half_lat, clat + half_lat


def _segments(paths: LinkPaths, idx: np.ndarray) -> list[np.ndarray]:
    return [np.column_stack([paths.lon[paths.start[k]:paths.start[k + 1]], paths.lat[paths.start[k]:paths.start[k + 1]]])
            for k in idx]


# ---- rendering -------------------------------------------------------------------------------

class Canvas:
    """A 1920x1080 matplotlib figure with the road background drawn once (blitting)."""

    def __init__(self, links, paths: LinkPaths, view, title: str):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        self.plt = plt
        self.fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1], facecolor=BG)
        lon0, lon1, lat0, lat1 = view
        self.ax.set_xlim(lon0, lon1)
        self.ax.set_ylim(lat0, lat1)
        self.ax.set_aspect(1 / np.cos(np.radians((lat0 + lat1) / 2)))
        self.ax.axis("off")
        inside = (links.v >= 0) & (links.lon[links.u] > lon0 - 0.05) & (links.lon[links.u] < lon1 + 0.05) \
            & (links.lat[links.u] > lat0 - 0.05) & (links.lat[links.u] < lat1 + 0.05)
        if links.priority is not None and inside.sum() > BACKDROP_ALL_ROADS:
            inside &= links.priority <= BACKDROP_MAX_PRIORITY
        lc = LineCollection(_segments(paths, np.flatnonzero(inside)), colors=ROAD, linewidths=0.8,
                            capstyle="round", joinstyle="round")
        self.ax.add_collection(lc)
        self.ax.text(0.015, 0.97, title, transform=self.ax.transAxes, color=TEXT, fontsize=22, va="top", weight="bold")
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.LineCollection = LineCollection

    def frame(self, artists) -> bytes:
        c = self.fig.canvas
        c.restore_region(self.background)
        for a in artists:
            self.ax.draw_artist(a)
        return np.asarray(c.buffer_rgba())[:, :, :3].tobytes()

    def close(self):
        self.plt.close(self.fig)


def _writer(path: Path, fps: int):
    import imageio_ffmpeg

    w = imageio_ffmpeg.write_frames(str(path), (W, H), fps=fps, codec="libx264", quality=8,
                                    macro_block_size=8, pix_fmt_out="yuv420p")
    w.send(None)
    return w


def _clock(h: float) -> str:
    h = h % 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


# ---- public transport vehicles from the timetables --------------------------------------------

def _bus_segments(data_dir: Path, pub) -> np.ndarray:
    """Every scheduled vehicle movement between consecutive stops:
    (t0, t1, lon0, lat0, lon1, lat1, stop, departure index)."""
    d = Path(data_dir) / "net_files" / "public_network"
    with (d / "schedules_public.csv").open(newline="") as f:
        sch = [[float(x) for x in r if x.strip()] for r in csv.reader(f)]
    rows = []
    for a, b in enumerate(pub.next_stop):
        if b < 0 or pub.mode[a] < 0 or pub.itin[b] != pub.itin[a]:
            continue  # last stop, rental bikes, or a walking transfer
        n = min(len(sch[a]), len(sch[b]))
        if n == 0:
            continue
        t0, t1 = np.array(sch[a][:n]), np.array(sch[b][:n])
        ok = t1 > t0
        rows.append(np.column_stack([t0[ok], t1[ok], np.full(ok.sum(), pub.lon[a]), np.full(ok.sum(), pub.lat[a]),
                                     np.full(ok.sum(), pub.lon[b]), np.full(ok.sum(), pub.lat[b]),
                                     np.full(ok.sum(), a), np.flatnonzero(ok)]))
    return np.vstack(rows) if rows else np.zeros((0, 8))


def _bike_segments(results: Path, data_dir: Path) -> np.ndarray:
    """Every cyclist movement between consecutive bike-network nodes of the executed (= planned:
    bikes do not interact) routes: (t0, t1, lon0, lat0, lon1, lat1), sorted by t0. Negative
    trajectory entries are the agenda locations (trip start/end markers) and are skipped."""
    files = results / "exec_bike_paths.csv", results / "exec_bike_passage_times.csv"
    if not all(f.exists() for f in files):
        return np.zeros((0, 6))
    with (Path(data_dir) / "net_files" / "bike_network" / "nodes_bike.csv").open(newline="") as f:
        lat_lon = np.array([[float(r[1]), float(r[2])] for r in csv.reader(f)])
    rows = []
    for nodes, times in read_trajectories(*files):
        if len(nodes) < 2:
            continue
        n, t = np.asarray(nodes), np.asarray(times)
        ok = (n[:-1] >= 0) & (n[1:] >= 0) & (t[1:] > t[:-1])
        a, b = n[:-1][ok], n[1:][ok]
        rows.append(np.column_stack([t[:-1][ok], t[1:][ok], lat_lon[a, 1], lat_lon[a, 0], lat_lon[b, 1], lat_lon[b, 0]]))
    out = np.vstack(rows) if rows else np.zeros((0, 6))
    return out[np.argsort(out[:, 0])]


def _bus_loads(results: Path, pub, buses: np.ndarray, cfg: dict, scale: float):
    """Riders and occupancy (% of capacity) of every row of `buses`, or None without trajectories.

    A rider is recorded at every stop of the line at the vehicle's times (public_sim.cpp,
    "Propagate down the itinerary"), so the riders of a departure between stop a and the next
    stop are the trajectory points (a, departure time) followed by next_stop[a]; they match the
    timetable to < 0.2 s. Capacity per vehicle as in Model::public_capacity: city bus, regional
    bus or train capacity, times capacity_scale (rounded, at least 1) for samples.
    """
    files = results / "exec_public_paths.csv", results / "exec_public_passage_times.csv"
    if len(buses) < 2 or not all(f.exists() for f in files):
        return None
    A, T = [], []
    for nodes, times in read_trajectories(*files):
        if len(nodes) < 2:
            continue
        n, t = np.asarray(nodes), np.asarray(times)
        a, b = n[:-1], n[1:]
        ok = (a >= 0) & (b >= 0)
        ok[ok] = (pub.next_stop[a[ok]] == b[ok]) & (pub.mode[a[ok]] >= 0)
        A.append(a[ok])
        T.append(t[:-1][ok])
    if not A:  # nobody rode public transport: every vehicle empty
        return np.zeros(len(buses), dtype=int), np.zeros(len(buses))
    A, T = np.concatenate(A), np.concatenate(T)
    # one sorted key per row, stop * 100 + departure hour, and each rider point to the nearest one
    stop = buses[:, 6].astype(int)
    order = np.lexsort((buses[:, 0], stop))
    key = stop[order] * 100.0 + buses[order, 0]
    r = A * 100.0 + T
    i = np.clip(np.searchsorted(key, r), 1, len(key) - 1)
    i = np.where(np.abs(key[i - 1] - r) < np.abs(key[i] - r), i - 1, i)
    hit = np.abs(key[i] - r) < 1 / 3600  # same stop, departure within 1 s
    riders = np.bincount(order[i[hit]], minlength=len(buses))
    per_mode = {0: cfg["city_bus_cap"], 1: cfg["reg_bus_cap"], 2: cfg["cercanias_cap"],
                3: cfg["city_bus_cap"], 4: cfg["reg_bus_cap"]}
    cap = np.array([per_mode.get(int(m), cfg["city_bus_cap"]) for m in pub.mode[stop]], dtype=float)
    if scale != 1.0:
        cap = np.maximum(1.0, np.floor(cap * scale + 0.5))  # std::lround for positive values
    return riders, 100.0 * riders / cap


# ---- videos ----------------------------------------------------------------------------------

def vehicles_video(results: Path, links, pub, data_dir: Path, scale: float, start: float, end: float,
                   minutes_per_frame: float, fps: int, cfg: dict | None = None) -> Path:
    t = np.load(results / "drive_links.npz")
    k = links.index(t["u"], t["av"])
    t_enter, t_end, t_exit = t["t_enter"], t["t_end"], t["t_exit"]
    paths = LinkPaths(links)
    view = _view(links, k)
    order = np.argsort(t_enter)
    k, t_enter, t_end, t_exit = k[order], t_enter[order], t_end[order], t_exit[order]
    max_dur = float(np.max(t_exit - t_enter)) if k.size else 0.0
    buses = _bus_segments(data_dir, pub)
    loads = _bus_loads(results, pub, buses, cfg, scale) if cfg else None
    bikes = _bike_segments(results, data_dir)
    max_bike = float(np.max(bikes[:, 1] - bikes[:, 0])) if len(bikes) else 0.0

    cv = Canvas(links, paths, view, "MiMoS: simulated traffic")
    cars = cv.ax.scatter([], [], s=14, c=CAR, linewidths=0, alpha=0.9)
    bike = cv.ax.scatter([], [], s=30, c=BIKE, marker="D", linewidths=0, alpha=0.95)
    if loads is None:
        bus = cv.ax.scatter([], [], s=28, c=BUS, marker="s", linewidths=0, alpha=0.95)
        legend = "blue: buses and trains (timetables)"
    else:  # occupancy: cyan empty -> magenta full (no orange in between, so no clash with the cars)
        import matplotlib.colors as mcolors

        norm = mcolors.Normalize(0, 100)
        bus = cv.ax.scatter([], [], s=40, c=np.zeros(0), cmap=BUS_CMAP, norm=norm, marker="s", linewidths=0,
                            alpha=0.95)
        cax = cv.fig.add_axes([0.80, 0.06, 0.17, 0.02])
        cb = cv.fig.colorbar(cv.plt.cm.ScalarMappable(norm=norm, cmap=BUS_CMAP), cax=cax, orientation="horizontal")
        cb.set_label("bus / train occupancy, % of capacity", color=TEXT, fontsize=14)
        cb.ax.tick_params(colors=TEXT, labelsize=12)
        legend = "squares: buses and trains (timetables), coloured by occupancy"
    clock = cv.ax.text(0.015, 0.90, "", transform=cv.ax.transAxes, color=TEXT, fontsize=40, va="top", family="monospace")
    info = cv.ax.text(0.015, 0.83, "", transform=cv.ax.transAxes, color=TEXT, fontsize=16, va="top")
    cv.ax.text(0.015, 0.03, f"orange: cars (simulated sample, 1 dot ~ {1 / scale:.0f} cars)   green diamonds: bikes   {legend}",
               transform=cv.ax.transAxes, color=TEXT, fontsize=14)
    cv.fig.canvas.draw()
    cv.background = cv.fig.canvas.copy_from_bbox(cv.fig.bbox)

    out = _out(results, "vehicles")
    w = _writer(out, fps)
    for h in np.arange(start, end, minutes_per_frame / 60):
        # active traversals: entered before h and not left yet (sorted by entry -> slice + mask)
        lo = np.searchsorted(t_enter, h - max_dur)
        hi = np.searchsorted(t_enter, h, side="right")
        act = np.arange(lo, hi)[t_exit[lo:hi] > h]
        frac = np.where(t_end[act] > t_enter[act], (h - t_enter[act]) / np.maximum(t_end[act] - t_enter[act], 1e-9), 1.0)
        x, y = paths.position(k[act], np.minimum(frac, 1.0))  # after t_end: waiting at the end of the link
        cars.set_offsets(np.column_stack([x, y]) if act.size else np.zeros((0, 2)))
        on = (buses[:, 0] <= h) & (buses[:, 1] > h)
        bs = buses[on]
        f = (h - bs[:, 0]) / (bs[:, 1] - bs[:, 0])
        bus.set_offsets(np.column_stack([bs[:, 2] + f * (bs[:, 4] - bs[:, 2]), bs[:, 3] + f * (bs[:, 5] - bs[:, 3])])
                        if bs.size else np.zeros((0, 2)))
        lo, hi = np.searchsorted(bikes[:, 0], h - max_bike), np.searchsorted(bikes[:, 0], h, side="right")
        bk = bikes[lo:hi][bikes[lo:hi, 1] > h]
        g = (h - bk[:, 0]) / (bk[:, 1] - bk[:, 0])
        bike.set_offsets(np.column_stack([bk[:, 2] + g * (bk[:, 4] - bk[:, 2]), bk[:, 3] + g * (bk[:, 5] - bk[:, 3])])
                         if bk.size else np.zeros((0, 2)))
        riders = ""
        if loads is not None:
            bus.set_array(np.minimum(loads[1][on], 100.0))
            riders = f", {loads[0][on].sum() / scale:,.0f} riders aboard (scaled)"
        clock.set_text(_clock(h))
        info.set_text(f"{act.size / scale:,.0f} cars on the roads, {len(bk) / scale:,.0f} bikes (scaled)   "
                      f"{on.sum()} pt vehicles{riders}")
        w.send(cv.frame([cars, bike, bus, clock, info]))
    w.close()
    cv.close()
    return out


def _hourly_link_video(results: Path, links, values_at, title: str, unit: str, out_name: str, start: float,
                       end: float, minutes_per_frame: float, fps: int, n_links: int = 12000,
                       vmax: float | None = None, cmap_name: str = "inferno") -> Path:
    """A fixed set of links recoloured every frame by values_at(h) (array over those links)."""
    import matplotlib.colors as mcolors
    from matplotlib import colormaps

    top, weights, fn = values_at
    paths = LinkPaths(links)
    view = _view(links, top, weights)
    cv = Canvas(links, paths, view, title)
    lc = cv.LineCollection(_segments(paths, top), linewidths=1.5, capstyle="round", joinstyle="round")
    cmap = colormaps[cmap_name]
    if vmax is None:  # 99th percentile over the day (a fixed scale, e.g. 100 % of capacity, is passed in)
        all_vals = np.stack([fn(h) for h in np.arange(start, end, 1.0)])
        vmax = float(np.percentile(all_vals[all_vals > 0], 99)) if (all_vals > 0).any() else 1.0
    norm = mcolors.Normalize(0, vmax)
    clock = cv.ax.text(0.015, 0.90, "", transform=cv.ax.transAxes, color=TEXT, fontsize=40, va="top", family="monospace")
    # colour bar drawn once into the background
    sm = cv.plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = cv.fig.add_axes([0.80, 0.06, 0.17, 0.02])
    cb = cv.fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(unit, color=TEXT, fontsize=14)
    cb.ax.tick_params(colors=TEXT, labelsize=12)
    cv.fig.canvas.draw()
    cv.background = cv.fig.canvas.copy_from_bbox(cv.fig.bbox)
    cv.ax.add_collection(lc)

    out = _out(results, out_name)
    w = _writer(out, fps)
    for h in np.arange(start, end, minutes_per_frame / 60):
        v = fn(h)
        lc.set_color(cmap(norm(v)))
        lc.set_linewidth(0.6 + 3.4 * np.sqrt(np.clip(v / vmax, 0, 1)))
        clock.set_text(_clock(h))
        w.send(cv.frame([lc, clock]))
    w.close()
    cv.close()
    return out


def traffic_video(results: Path, links, scale: float, t_s_h: float, start, end, minutes_per_frame, fps) -> Path:
    t = np.load(results / "drive_links.npz")
    k = links.index(t["u"], t["av"])
    t_enter = t["t_enter"]
    daily = np.bincount(k, minlength=len(links.u))
    top = np.argsort(daily)[::-1][:12000]
    top = top[daily[top] > 0]
    pos = np.full(len(links.u), -1)
    pos[top] = np.arange(top.size)
    keep = pos[k] >= 0
    kk, te = pos[k[keep]], t_enter[keep]
    order = np.argsort(te)
    kk, te = kk[order], te[order]

    cap_flow = links.lanes[top] / t_s_h  # vehicles per hour the link can take (whole population)

    def fn(h):  # % of flow capacity used by the cars entering in the hour before h (scaled)
        a, b = np.searchsorted(te, h - 1.0), np.searchsorted(te, h)
        return 100 * np.bincount(kk[a:b], minlength=top.size) / scale / cap_flow

    return _hourly_link_video(results, links, (top, daily[top].astype(float), fn),
                              "Traffic: flow as % of link capacity (moving one-hour window)",
                              "% of flow capacity", "traffic", start, end, minutes_per_frame, fps,
                              vmax=100.0, cmap_name="RdYlGn_r")


def emissions_video(results: Path, links, scale: float, n_agents: int, start, end, minutes_per_frame, fps,
                    pollutant: str = "NOx") -> Path:
    """g/km/h of one pollutant on every link over a moving one-hour window, from the emissions of
    each link traversal (same calculation and vehicle draw as `mimos emissions`)."""
    from mimos.emissions import Settings, traversal_grams

    t = dict(np.load(results / "drive_links.npz"))
    G, k, _, _ = traversal_grams(t, links, n_agents, Settings.of_run(results), pollutants=(pollutant,))
    grams = G[:, 0].astype(float)
    length = np.where(links.length_km > 0, links.length_km, np.nan)
    daily = np.bincount(k, weights=grams, minlength=len(links.u))
    top = np.argsort(daily)[::-1][:12000]
    top = top[daily[top] > 0]
    pos = np.full(len(links.u), -1)
    pos[top] = np.arange(top.size)
    keep = pos[k] >= 0
    kk, te, gg = pos[k[keep]], t["t_enter"][keep], grams[keep]
    order = np.argsort(te)
    kk, te, gg = kk[order], te[order], gg[order]
    per_km = 1 / length[top] / scale

    def fn(h):  # grams emitted in the hour centred on h, per km of link (scaled): g/km/h
        a, b = np.searchsorted(te, h - 0.5), np.searchsorted(te, h + 0.5)
        return np.nan_to_num(np.bincount(kk[a:b], weights=gg[a:b], minlength=top.size) * per_km)

    return _hourly_link_video(results, links, (top, daily[top], fn),
                              f"{pollutant} emissions (g per km per hour, moving one-hour window, scaled to the population)",
                              f"{pollutant} g/km/h", f"emissions_{pollutant.replace(' ', '_')}",
                              start, end, minutes_per_frame, fps)


def parse_clock(value) -> float:
    """'08:30' or '8.5' -> 8.5 hours (agendas run past midnight: '25:00' is 1 am next day)."""
    text = str(value)
    if ":" in text:
        h, m = text.split(":", 1)
        return int(h) + int(m) / 60
    return float(text)


def make_videos(run_dir: Path | str, what=("vehicles", "traffic", "emissions"), start: float = 5.0,
                end: float = 24.0, speed: float | None = None, fps: int = 30, view: dict | None = None,
                tag: str = "") -> list[Path]:
    """Renders the chosen videos between `start` and `end` (hours of the day).

    speed: simulated seconds per second of video (1 = real time, 60 = one simulated minute per
    second, 1800 = half an hour per second). Default: 1800 for vehicles and 9000 for the
    traffic/emissions maps (which change slowly). tag: added to the file names, e.g. "_0800_slow".
    view = {center_lat, center_lon, radius_km} overrides the [view] saved with the run.
    """
    results = Path(run_dir)
    with (results / "config.toml").open("rb") as f:
        saved = tomllib.load(f)
    cfg = saved["config"]
    v = view or saved.get("view") or {}
    TAG["tag"] = tag
    VIEW["center"] = (v["center_lat"], v["center_lon"]) if "center_lat" in v and "center_lon" in v else None
    VIEW["radius_km"] = float(v.get("radius_km", 6.0))
    info = json.loads((results / "run_info.json").read_text())
    scale = float(info.get("capacity_scale", 1.0)) or 1.0
    links = load_drive_links(cfg["data_dir"])

    def minutes_per_frame(default_speed):
        return (speed or default_speed) / 60 / fps  # simulated minutes per video frame

    out = []
    if "vehicles" in what:
        out.append(vehicles_video(results, links, load_public_nodes(cfg["data_dir"]), Path(cfg["data_dir"]), scale,
                                  start, end, minutes_per_frame(1800), fps, cfg=cfg))
    if "traffic" in what:
        out.append(traffic_video(results, links, scale, float(cfg["t_s"]), start, end, minutes_per_frame(9000), fps))
    if "emissions" in what and (results / "emissions" / "E.npz").exists():
        out.append(emissions_video(results, links, scale, int(info.get("n_agents", 0)), start, end,
                                   minutes_per_frame(9000), fps))
    return out
