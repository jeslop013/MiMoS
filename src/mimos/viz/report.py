"""Self-contained HTML report of a MiMoS run, for checking results by eye.

    uv run mimos report <run_dir> [--targets <targets.toml>] [--png] [--serve]

Reads what every run writes (config.toml, run_info.json, iterations.csv, trips.csv,
drive_links.npz, exec_public_*.csv; emissions/ when present) and the networks from the run's
data_dir. Writes <run_dir>/report/report.html: one file with plotly.js embedded, so it opens offline
except the map tiles, which need the page served over http (VS Code preview or ``--serve``): opened
as a file, Chrome/Edge leave tile maps blank. ``--no-tiles`` draws the model's own roads instead and
works anywhere (the version to send to others; README, *Viewing and sharing a report*). Sections: run, relaxation,
trips, roads, public transport, emissions; the time-dependent maps (flow, occupancy, waits,
emissions) are also animated hour by hour in the HTML. ``--png`` also writes every
figure, and every layer of the layered (peak-hour) maps, as a 1920x1080 PNG at 2x in <run_dir>/report/figures/
for presentations. Heavier animations (moving vehicles, maps through the day) are MP4 videos:
``mimos video`` (src/mimos/video.py).

Hours: agendas run past midnight (to ~28 h); hourly charts fold them onto 0-23 h.
Population scaling: the simulated sample is scaled by 1/capacity_scale where a chart says
"scaled to the population".
"""

from __future__ import annotations

import html
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from mimos.analysis import MODES, load_targets
from mimos.networks import load_drive_links, load_public_nodes, read_trajectories

COLORS = {"walk": "#4daf4a", "car": "#e41a1c", "pt": "#377eb8", "bike": "#ff7f00",
          "drive": "#e41a1c", "public": "#377eb8"}
BANDS = ((8, 10), (17, 19))  # peak hours of the static (PNG) maps
# Time resolution of the animated maps and the curves: a value every STEP_H, computed over a
# centred moving window (rates per hour). Finer than hourly, but smooth enough for a sample.
STEP_H = 1 / 6          # 10 minutes
MAP_WINDOW_H = 1.0      # maps: moving one-hour window
CURVE_WINDOW_H = 0.5    # curves: moving half-hour window
N_STEPS = int(round(24 / STEP_H))
FRAME_LABELS = [f"{int(i * STEP_H):02d}:{int(round((i * STEP_H % 1) * 60)):02d}" for i in range(N_STEPS)]


def _bin(t: np.ndarray) -> np.ndarray:
    """10-minute bin of the day (0 .. N_STEPS-1) of times in hours (agendas past midnight fold)."""
    return (np.floor(t / STEP_H).astype(np.int64)) % N_STEPS


def _moving_sum(x: np.ndarray, window_h: float) -> np.ndarray:
    """Centred moving sum over the bins of axis 0 (circular: the day wraps around midnight)."""
    w = max(1, int(round(window_h / STEP_H)))
    out = np.zeros_like(x, dtype=float)
    for off in range(-(w // 2), w - w // 2):
        out += np.roll(x, -off, axis=0)
    return out


EXPORT: list[tuple[str, go.Figure]] = []  # figures of the report being built, for --png


def _fig_html(fig: go.Figure, height: int = 420, name: str | None = None, embed: bool = True) -> str:
    """HTML of a figure, registered for --png. embed=False: PNG export only (not in the page),
    used for the peak-hour maps, whose information the animated maps already carry."""
    top = fig.layout.margin.t if fig.layout.margin.t is not None else 50
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=top, b=40), template="plotly_white")
    EXPORT.append((name or f"figure_{len(EXPORT) + 1}", fig))
    if not embed:
        return ""
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def _table(rows: list[tuple[str, object]]) -> str:
    cells = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    return f"<table class='kv'>{cells}</table>"


def _note(text: str) -> str:
    return f"<p class='note'>{text}</p>"


# ---- 1. Run ---------------------------------------------------------------------------------

KEY_PARAMS = ["n_agents", "it_number", "sample_random", "capacity_scale", "legacy_mode", "fifo_link_exit",
              "stuck_time", "car_access_speed", "C_drive", "C_public", "C_bike", "C_walk", "beta_trav_drive",
              "alpha_l_i", "alpha_l_f", "alpha_f_i", "alpha_f_f", "k_min", "k_max", "seed", "data_dir"]


def _short_path(value: str) -> str:
    """A path relative to the current folder when it is inside it (no user folders in shared reports)."""
    try:
        return Path(value).resolve().relative_to(Path.cwd().resolve()).as_posix() + "/"
    except ValueError:
        return Path(value).name + "/"


def section_run(cfg: dict, info: dict, it: pd.DataFrame) -> str:
    rows = [(k, _short_path(cfg[k]) if k == "data_dir" else cfg.get(k)) for k in KEY_PARAMS if k in cfg]
    rows += [("agents simulated / population", f"{info.get('n_agents')} / {info.get('population_size')}"),
             ("capacity scale applied", f"{info.get('capacity_scale', float('nan')):.4f}"),
             ("load + bike planning (s)", f"{info.get('load_seconds')} + {info.get('bike_seconds')}")]
    secs = [c for c in it.columns if c.startswith("seconds_")]
    if secs:
        rows.append(("mean seconds per iteration", f"{it[secs].sum(axis=1).mean():.0f}"))
    if "invariants_ok" in it:
        bad = (~it["invariants_ok"].astype(bool)).sum()
        rows.append(("invariant checks", "all passed" if bad == 0 else f"FAILED in {bad} iterations (see log.txt)"))
    return _table(rows)


# ---- 2. Relaxation --------------------------------------------------------------------------

def section_relaxation(it: pd.DataFrame, targets: dict | None) -> str:
    x = it["iteration"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Mean |change of planned scores| (log)",
                                                        "Score of the executed mode (mean, p5-p95)",
                                                        "Planned - executed (mean)"))
    fig.add_trace(go.Scatter(x=x, y=it["score_change_mean"], mode="lines+markers", line=dict(color="black"),
                             showlegend=False), 1, 1)
    fig.update_yaxes(type="log", row=1, col=1)
    for name, color in (("planned", "#377eb8"), ("executed", "#e41a1c")):
        lo, hi = it[f"score_{name}_p5"], it[f"score_{name}_p95"]
        fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]), y=list(hi) + list(lo[::-1]), fill="toself",
                                 fillcolor=color, opacity=0.15, line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"), 1, 2)
        fig.add_trace(go.Scatter(x=x, y=it[f"score_{name}_mean"], name=name, line=dict(color=color)), 1, 2)
    fig.add_trace(go.Scatter(x=x, y=it["score_gap_mean"], name="gap", line=dict(color="black"),
                             showlegend=False), 1, 3)
    fig.update_xaxes(title_text="iteration")
    out = [_fig_html(fig, name="relaxation_scores")]

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Agents switching mode", "Day-mode shares",
                                                        "Trip shares by basic mode"))
    fig.add_trace(go.Bar(x=x, y=it["mode_switches"], marker_color="grey", showlegend=False), 1, 1)
    for m in ("drive", "public", "bike"):
        fig.add_trace(go.Scatter(x=x, y=100 * it[f"day_{m}"], name=f"day {m}", line=dict(color=COLORS[m])), 1, 2)
    sfx = "_within" if "trip_walk_within" in it else ""
    for m in MODES:
        fig.add_trace(go.Scatter(x=x, y=100 * it[f"trip_{m}{sfx}"], name=f"trips {m}",
                                 line=dict(color=COLORS[m], dash="dot")), 1, 3)
        if targets:
            fig.add_hline(y=100 * targets["shares"][m], line=dict(color=COLORS[m], dash="dash", width=1),
                          row=1, col=3)
    fig.update_xaxes(title_text="iteration")
    fig.update_yaxes(title_text="%", row=1, col=2)
    out.append(_fig_html(fig, name="relaxation_shares"))
    out.append(_note(
        "Left: how much the agents' planned scores (drive and public, every agent) change from one iteration "
        "to the next; it decays towards 0 as perceptions settle. MiMoS plans with <i>perceived</i> times "
        "(free flow + remembered delays); the executed score replaces them with the times met in the simulation. "
        "At a relaxed state the scores and mode shares stop drifting, the gap stays small and few agents switch "
        "mode. Dashed lines: survey targets"
        + (" (trip shares of trips within the targets' max_trip_km)" if sfx else "") + "."))
    return "".join(out)


# ---- 3. Trips --------------------------------------------------------------------------------

def section_trips(trips: pd.DataFrame, targets: dict | None) -> str:
    out = []
    max_km = (targets or {}).get("scope", {}).get("max_trip_km")
    inside = trips[trips["distance_km"] <= max_km] if max_km else trips
    share = inside["mode"].value_counts(normalize=True).reindex(list(MODES)).fillna(0)
    fig = go.Figure(go.Bar(x=list(MODES), y=100 * share.values, name="simulated",
                           marker_color=[COLORS[m] for m in MODES]))
    if targets:
        fig.add_trace(go.Bar(x=list(MODES), y=[100 * targets["shares"][m] for m in MODES], name="survey",
                             marker_color="lightgrey"))
    title = "Trip modal split" + (f" (trips up to {max_km:g} km: {100 * len(inside) / len(trips):.1f}% of all)"
                                  if max_km else "")
    fig.update_layout(title=title, yaxis_title="%", barmode="group")
    out.append(_fig_html(fig, 380, "trips_modal_split"))

    d = (targets or {}).get("durations")
    if not d:
        # All modes in one histogram: same bins, same axes, semi-transparent bars.
        cap = 90.0
        fig = go.Figure()
        beyond = []
        for m in MODES:
            x = inside.loc[inside["mode"] == m, "minutes"].dropna()
            if not len(x):
                continue
            beyond.append(f"{m} {100 * (x > cap).mean():.1f}%")
            fig.add_trace(go.Histogram(x=x.clip(upper=cap), xbins=dict(start=0, end=cap + 2, size=2),
                                       histnorm="percent", name=m, marker_color=COLORS[m], opacity=0.55))
        fig.update_layout(barmode="overlay", title="Trip duration by mode (door to door)",
                          xaxis_title="minutes", yaxis_title="% of the trips of each mode",
                          legend=dict(x=0.99, xanchor="right", y=0.99))
        fig.update_xaxes(range=[0, cap + 2])
        out.append(_fig_html(fig, 420, "trips_durations"))
        out.append(_note(f"Bins of 2 minutes; trips longer than {cap:g} min are counted in the last bin ("
                         + ", ".join(beyond) + ")."))
    modes_d = [m for m in ("walk", "car", "pt") if d and m in d]
    fig = make_subplots(rows=1, cols=max(len(modes_d), 1), subplot_titles=[f"{m}: trip duration" for m in modes_d])
    for c, m in enumerate(modes_d, start=1):
        x = inside.loc[inside["mode"] == m, "minutes"].dropna()
        if d:
            edges = np.array(d["bin_edges"] + [max(240.0, x.max() if len(x) else 0) + 1])
            sim, _ = np.histogram(x, bins=edges)
            labels = [f"{a:g}-" for a in d["bin_edges"]]
            fig.add_trace(go.Bar(x=labels, y=100 * sim / max(sim.sum(), 1), name="simulated",
                                 marker_color=COLORS[m], showlegend=c == 1), 1, c)
            ref = np.array(d[m], dtype=float)
            fig.add_trace(go.Bar(x=labels, y=100 * ref / ref.sum(), name="survey", marker_color="lightgrey",
                                 showlegend=c == 1), 1, c)
        else:
            fig.add_trace(go.Histogram(x=x, histnorm="percent", marker_color=COLORS[m], showlegend=False), 1, c)
    if d:
        fig.update_xaxes(title_text="minutes (bin start)")
        fig.update_yaxes(title_text="% of trips", col=1)
        out.append(_fig_html(fig, 380, "trips_durations"))

    fig = go.Figure()
    for m in MODES:
        dep = (trips.loc[trips["mode"] == m, "departure_hour"].dropna() % 24).astype(int)
        fig.add_trace(go.Bar(x=np.arange(24), y=np.bincount(dep, minlength=24)[:24], name=m,
                             marker_color=COLORS[m]))
    fig.update_layout(barmode="stack", title="Trips by hour of departure and mode (simulated sample)",
                      xaxis_title="hour of day", yaxis_title="trips", legend=dict(x=0.99, xanchor="right", y=0.99))
    out.append(_fig_html(fig, 420, "trips_departures"))

    fig = go.Figure()
    km = trips["distance_km"].clip(lower=0.01)
    bins = np.logspace(-2, np.log10(max(km.max(), 10)) + 0.01, 50)
    for m in MODES:
        h, _ = np.histogram(km[trips["mode"] == m], bins=bins)
        fig.add_trace(go.Scatter(x=bins[:-1], y=h, name=m, line=dict(color=COLORS[m], shape="hv")))
    if max_km:
        fig.add_vline(x=max_km, line=dict(dash="dash", color="black"))
    fig.update_layout(title="Trip distance by mode (straight line, all trips)", yaxis_title="trips")
    fig.update_xaxes(type="log", title_text="km")
    out.append(_fig_html(fig, 380, "trips_distance"))
    far = trips["distance_km"] > (max_km or 50)
    out.append(_note(f"{far.mean() * 100:.1f}% of trips are longer than {max_km or 50:g} km (straight line); "
                     f"{(trips['distance_km'] > 500).mean() * 100:.2f}% longer than 500 km. "
                     "Durations are door to door, from the executed trajectories."))
    return "".join(out)


# ---- 4. Roads -------------------------------------------------------------------------------

def _segments(links, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of links idx as float32 arrays with NaN between links: one trace draws many
    links, and plotly embeds numpy arrays as compact binary (not long decimal text). Links follow
    their real road shape when link_geometry.npz exists (straight lines otherwise)."""
    if idx.size == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    gap = np.array([np.nan], dtype=np.float32)
    lons, lats = [], []
    for k in idx:
        lo, la = links.shape(int(k))
        lons += [lo, gap]
        lats += [la, gap]
    return np.concatenate(lats), np.concatenate(lons)


# Maps. Default: online map tiles (MapLibre, "carto-positron" style; no data cost in the file). They
# need the page served over http: on file:// pages MapLibre's worker stops and the style never loads
# (blank maps in Chrome/Edge; README, *Viewing and sharing a report*). Option tiles=False: the model's own main
# roads as a light grey background instead, fully offline. Rendering cost is kept low either way:
# WebGL traces, one trace per colour class, coordinates as float32 binary arrays, and one figure per
# section with buttons to switch layers (few WebGL contexts on the page).
TILES = {"enabled": True, "style": "carto-positron"}
# Map view: centre and radius from the run's [view] (config.toml) or --center / --radius-km;
# without them, the traffic-weighted median of the drawn links (robust to far-away traffic).
VIEW = {"center": None, "radius_km": 6.0}
MAP_HEIGHT_PX = 560             # plot area height used to turn radius_km into a map zoom
BACKDROP_DEG = (0.15, 0.1)      # background drawn within this half-width
BACKDROP_MAX_PRIORITY = 5.5     # priority_drive.csv: 0 motorway ... 6 residential
BACKDROP_ALL_ROADS = 20_000  # small networks (e.g. the example town): draw every road as background


def _wmedian(x: np.ndarray, w: np.ndarray | None) -> float:
    if w is None:
        return float(np.median(x))
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])


def _center(links, idx: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """(lon, lat) of the map centre: the configured view, else the weighted median of the links."""
    if VIEW["center"] is not None:
        lat, lon = VIEW["center"]
        return float(lon), float(lat)
    if idx.size == 0:
        return float(np.nanmedian(links.lon)), float(np.nanmedian(links.lat))
    return _wmedian(links.lon[links.u[idx]], weights), _wmedian(links.lat[links.u[idx]], weights)


def _zoom(lat: float, radius_km: float) -> float:
    """Web-map zoom at which the plot height shows 2 x radius_km (156543 m per pixel at zoom 0)."""
    m_per_px = 2 * radius_km * 1000 / MAP_HEIGHT_PX
    return float(np.log2(156543.03 * np.cos(np.radians(lat)) / m_per_px))


def _lines(lat, lon, **kw):
    """A line/marker trace on the map: Scattermap (tiles) or Scattergl in lon/lat axes."""
    if TILES["enabled"]:
        return go.Scattermap(lat=lat, lon=lon, **kw)
    return go.Scattergl(x=lon, y=lat, **kw)


def _map_figure(links, center: tuple[float, float], title: str) -> go.Figure:
    clon, clat = center
    if TILES["enabled"]:
        fig = go.Figure()
        fig.update_layout(title=title, legend=dict(orientation="h", y=-0.02),
                          map=dict(style=TILES["style"], center=dict(lat=clat, lon=clon),
                                   zoom=_zoom(clat, VIEW["radius_km"])))
        return fig
    near = (links.v >= 0) & (np.abs(links.lon[links.u] - clon) < BACKDROP_DEG[0]) \
        & (np.abs(links.lat[links.u] - clat) < BACKDROP_DEG[1])
    if links.priority is not None and near.sum() > BACKDROP_ALL_ROADS:
        near &= links.priority <= BACKDROP_MAX_PRIORITY
    lat, lon = _segments(links, np.flatnonzero(near))
    fig = go.Figure(go.Scattergl(x=lon, y=lat, mode="lines", line=dict(color="#d9d9d9", width=1),
                                 hoverinfo="skip", showlegend=False, name="roads"))
    fig.update_layout(title=title, legend=dict(orientation="h", y=-0.08), plot_bgcolor="white")
    half_lat = VIEW["radius_km"] / 111.2
    half_lon = 2 * half_lat / np.cos(np.radians(clat))  # the plot is about twice as wide as tall
    fig.update_xaxes(range=[clon - half_lon, clon + half_lon], showgrid=False, zeroline=False,
                     title_text="longitude")
    # equal distances on both axes: one degree of latitude spans 1/cos(lat) degrees of longitude
    fig.update_yaxes(range=[clat - half_lat, clat + half_lat], scaleanchor="x",
                     scaleratio=1 / np.cos(np.radians(clat)), showgrid=False, zeroline=False, title_text="latitude")
    return fig


# Colour schemes: (lower, upper, colour, line width, legend label). Fixed for a layer, so colours
# mean the same thing on every map and at every hour of an animation.
CAPACITY_COLORS = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]
SEQ_COLORS = ["#fecc5c", "#fd8d3c", "#f03b20", "#bd0026", "#67000d"]
WIDTHS = [1.2, 1.8, 2.5, 3.2, 4.2]


def scheme_percent(what: str) -> list[tuple]:
    """Share of capacity: < 25, 25-50, 50-75, 75-100, > 100 %."""
    edges = [0, 25, 50, 75, 100, np.inf]
    return [(lo, hi, c, w, f"{lo:g}-{hi:g} % {what}" if np.isfinite(hi) else f"> {lo:g} % {what}")
            for lo, hi, c, w in zip(edges[:-1], edges[1:], CAPACITY_COLORS, WIDTHS)]


def scheme_fixed(edges: list[float], unit: str) -> list[tuple]:
    edges = list(edges) + [np.inf]
    return [(lo, hi, c, w, f"{lo:g}-{hi:g} {unit}" if np.isfinite(hi) else f"> {lo:g} {unit}")
            for lo, hi, c, w in zip(edges[:-1], edges[1:], SEQ_COLORS, WIDTHS)]


def scheme_quantiles(values: np.ndarray, unit: str) -> list[tuple]:
    """Classes at the 50/75/90/97.5 % quantiles of the positive values (whole day)."""
    v = values[np.isfinite(values) & (values > 0)]
    q = np.quantile(v, [0.5, 0.75, 0.9, 0.975]) if v.size else np.array([1, 2, 3, 4.0])
    return scheme_fixed([0, *[float(f"{x:.3g}") for x in q]], unit)


WAIT_SCHEME = scheme_fixed([0, 5, 15, 30, 60], "s wait")


def _classify(values: np.ndarray, scheme: list[tuple]) -> np.ndarray:
    """Class 1..len(scheme) of each value (0 = not drawn: zero, negative or NaN)."""
    cls = np.zeros(values.shape, dtype=np.uint8)
    ok = np.isfinite(values) & (values > 0)
    for c, (lo, hi, *_rest) in enumerate(scheme, start=1):
        cls[ok & (values > lo) & (values <= hi)] = c
    return cls


def _class_traces(links, values: np.ndarray, scheme: list[tuple], max_links: int) -> list:
    """Links with a value > 0, one WebGL line trace per class of `scheme`."""
    idx = np.flatnonzero(np.isfinite(values) & (values > 0))
    idx = idx[np.argsort(values[idx])][-max_links:]
    if idx.size == 0:
        return []
    cls = _classify(values[idx], scheme)
    traces = []
    for c, (lo, hi, color, width, label) in enumerate(scheme, start=1):
        sel = idx[cls == c]
        lat, lon = _segments(links, sel)
        traces.append(_lines(lat, lon, mode="lines", line=dict(color=color, width=width), hoverinfo="skip",
                             name=label))
        # Round joints: a dot as wide as the line at every link end, so links meeting at an angle
        # do not show the square ends of WebGL lines. Two points per link (ends shared by
        # links of the same class are drawn once).
        ends = np.unique(np.concatenate([links.u[sel], links.v[sel]]))
        traces.append(_lines(links.lat[ends].astype(np.float32), links.lon[ends].astype(np.float32),
                             mode="markers", marker=dict(color=color, size=width), hoverinfo="skip",
                             showlegend=False))
    return traces


def _layered_map(links, layers: list[tuple[str, np.ndarray, list]], title: str, max_links: int = 15000) -> go.Figure:
    """One map, several layers (label, value per link, colour scheme) switched with buttons."""
    first = layers[0][1]
    idx0 = np.flatnonzero(np.isfinite(first) & (first > 0))
    fig = _map_figure(links, _center(links, idx0, first[idx0] if idx0.size else None), title)
    groups = []
    for label, values, scheme in layers:
        traces = _class_traces(links, values, scheme, max_links)
        start = len(fig.data)
        for tr in traces:
            tr.visible = not groups  # only the first layer is shown at start
            fig.add_trace(tr)
        groups.append((label, range(start, start + len(traces))))
    if len(groups) > 1:
        buttons = []
        for label, rng in groups:
            base = 0 if TILES["enabled"] else 1  # without tiles, trace 0 is the road background
            visible = [True] * base + [i in rng for i in range(base, len(fig.data))]
            buttons.append(dict(label=label, method="update", args=[{"visible": visible}, {"title": f"{title}: {label}"}]))
        fig.update_layout(title=f"{title}: {groups[0][0]}", updatemenus=[dict(
            type="buttons", direction="right", x=0, y=1.08, xanchor="left", yanchor="bottom", buttons=buttons,
            showactive=True)], margin=dict(t=110))
    return fig


import base64  # noqa: E402  (used by the animated maps only)


def _b64(arr: np.ndarray, compress: bool = False) -> str:
    """Base64 of an array's bytes; compress=True deflates them first (the colour classes are mostly
    repeated bytes: ~10x smaller), and the page inflates them with the browser's DecompressionStream."""
    import zlib

    raw = np.ascontiguousarray(arr).tobytes()
    return base64.b64encode(zlib.compress(raw, 9) if compress else raw).decode()


ANIM_JS = """
async function mimosLinkAnim(cfg) {
  // Animated link map. The road shapes are stored once (float32) and each layer holds one byte
  // per link and time step (its colour class, deflated); traces are rebuilt here when the time or
  // the layer changes.
  const bytes = b => { const s = atob(b), u = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; };
  const dec = (b, T) => new T(bytes(b).buffer);
  const inflate = async b => new Uint8Array(await new Response(
    new Blob([bytes(b)]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer());
  const start = dec(cfg.start, Int32Array), lon = dec(cfg.lon, Float32Array), lat = dec(cfg.lat, Float32Array);
  const N = start.length - 1;
  const layers = await Promise.all(cfg.layers.map(async L => Object.assign({}, L, {cls: await inflate(L.classes)})));
  const gd = document.getElementById(cfg.div), sel = document.getElementById(cfg.div + '-layer');
  const slider = document.getElementById(cfg.div + '-hour'), label = document.getElementById(cfg.div + '-label');
  const play = document.getElementById(cfg.div + '-play');
  function traces(li, f) {
    const L = layers[li], out = [];
    L.scheme.forEach((sc, c0) => {
      const c = c0 + 1, x = [], y = [], ex = [], ey = [];
      for (let i = 0; i < N; i++) {
        if (L.cls[f * N + i] !== c) continue;
        const a = start[i], b = start[i + 1];
        for (let p = a; p < b; p++) { x.push(lon[p]); y.push(lat[p]); }
        x.push(null); y.push(null); ex.push(lon[a], lon[b - 1]); ey.push(lat[a], lat[b - 1]);
      }
      const xy = (X, Y) => cfg.tiles ? {type: 'scattermap', lon: X, lat: Y} : {type: 'scattergl', x: X, y: Y};
      out.push(Object.assign(xy(x, y), {mode: 'lines', line: {color: sc[0], width: sc[1]}, name: sc[2], hoverinfo: 'skip'}));
      out.push(Object.assign(xy(ex, ey), {mode: 'markers', marker: {color: sc[0], size: sc[1]}, showlegend: false, hoverinfo: 'skip'}));
    });
    return out;
  }
  function draw() {
    const li = +sel.value, f = +slider.value, L = layers[li];
    label.textContent = cfg.frames[f];
    const layout = Object.assign({}, gd.layout, {title: {text: L.title + ' - ' + cfg.frames[f]}, uirevision: 'keep',
                                                 datarevision: li * 1000 + f});
    Plotly.react(gd, gd.data.slice(0, cfg.nBase).concat(traces(li, f)), layout);
  }
  let timer = null;
  sel.onchange = draw; slider.oninput = draw;
  play.onclick = () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = 'play'; return; }
    play.textContent = 'pause';
    timer = setInterval(() => { slider.value = (+slider.value + 1) % cfg.frames.length; draw(); }, 250);
  };
  draw();
}
"""


def _animated_link_map(links, cand: np.ndarray, layers: list[tuple[str, str, np.ndarray, list]], title: str,
                       div_id: str, frame0: int | None = None, n_links: int = 8000, height: int = 660) -> str:
    """HTML of an animated link map (every 10 minutes) with a layer menu, time slider and play.

    cand: candidate link indices; layers: (label, title, values[N_STEPS, len(cand)], scheme). The
    n_links candidates with the highest classes over the day are kept; their shapes are stored
    once and each layer adds one deflated byte per link and time step.
    """
    frame0 = int(8 / STEP_H) if frame0 is None else frame0
    classes = [_classify(v, sch) for _, _, v, sch in layers]
    score = sum(c.max(axis=0).astype(float) for c in classes) + 1e-6 * np.nansum(layers[0][2], axis=0)
    order = np.flatnonzero(score > 0.5)
    order = order[np.argsort(score[order])[::-1][:n_links]]
    if order.size == 0:
        return _note("Nothing to animate.")
    keep = cand[order]
    shapes = [links.shape(int(k)) for k in keep]
    start = np.concatenate([[0], np.cumsum([len(sh[0]) for sh in shapes])]).astype(np.int32)
    lon = np.concatenate([sh[0] for sh in shapes]).astype(np.float32)
    lat = np.concatenate([sh[1] for sh in shapes]).astype(np.float32)

    w = np.nansum(layers[0][2][:, order], axis=0)
    fig = _map_figure(links, _center(links, keep, w if w.sum() > 0 else None), title)
    n_base = len(fig.data)  # the link traces are drawn by mimosLinkAnim at page load
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=60, b=40), template="plotly_white",
                      title=f"{layers[0][1]} - {FRAME_LABELS[frame0]}")
    plot = fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id, config={"responsive": True})
    cfg = {"div": div_id, "nBase": n_base, "tiles": TILES["enabled"], "start": _b64(start), "lon": _b64(lon),
           "lat": _b64(lat), "frames": FRAME_LABELS,
           "layers": [{"label": lab, "title": tit, "classes": _b64(c[:, order], compress=True),
                       "scheme": [[col, wid, txt] for _, _, col, wid, txt in sch]}
                      for (lab, tit, _, sch), c in zip(layers, classes)]}
    options = "".join(f"<option value='{i}'>{html.escape(lab)}</option>" for i, (lab, *_r) in enumerate(layers))
    controls = (f"<div class='anim'><select id='{div_id}-layer'>{options}</select> "
                f"<button id='{div_id}-play'>play</button> "
                f"<input type='range' id='{div_id}-hour' min='0' max='{N_STEPS - 1}' value='{frame0}' style='width:55%'> "
                f"<span id='{div_id}-label'></span></div>")
    script = f"<script>window.addEventListener('load', () => mimosLinkAnim({json.dumps(cfg)}));</script>"
    return controls + plot + script


def link_series(links, t, k: np.ndarray, scale: float, t_s_h: float, cand: np.ndarray):
    """For the candidate links, every 10 minutes over a moving one-hour window (arrays
    [N_STEPS, len(cand)]): flow as % of flow capacity, occupancy (mean cars on the link) as % of
    storage, and mean wait in seconds (links with 3+ simulated cars in the window, else 0)."""
    pos = np.full(len(links.u), -1)
    pos[cand] = np.arange(cand.size)
    j = pos[k]
    sel = j >= 0
    j = j[sel]
    t_enter, t_end, t_exit = t["t_enter"][sel], t["t_end"][sel], t["t_exit"][sel]
    b = _bin(t_enter)
    shape = (N_STEPS, cand.size)
    entries = np.zeros(shape)
    np.add.at(entries, (b, j), 1.0)
    wsum = np.zeros(shape)
    np.add.at(wsum, (b, j), 3600 * (t_exit - t_end))
    # vehicle-hours on the link in each 10-minute bin (a stay can cover several bins)
    vh = np.zeros(shape)
    b0 = np.floor(t_enter / STEP_H).astype(np.int64)
    b1 = np.floor(t_exit / STEP_H).astype(np.int64)
    one = b0 == b1
    np.add.at(vh, (b0[one] % N_STEPS, j[one]), (t_exit - t_enter)[one])
    many = np.flatnonzero(~one)
    for off in range(0, int((b1[many] - b0[many]).max()) + 1 if many.size else 0):
        m = many[b1[many] - b0[many] >= off]
        lo = np.maximum(t_enter[m], (b0[m] + off) * STEP_H)
        hi = np.minimum(t_exit[m], (b0[m] + off + 1) * STEP_H)
        np.add.at(vh, ((b0[m] + off) % N_STEPS, j[m]), np.clip(hi - lo, 0, None))

    cap_flow = links.lanes[cand] / t_s_h                # vehicles per hour (whole population)
    storage = np.maximum(links.storage[cand], 1.0)
    win_entries = _moving_sum(entries, MAP_WINDOW_H)
    flow = 100 * win_entries / MAP_WINDOW_H / scale / cap_flow
    occ = 100 * _moving_sum(vh, MAP_WINDOW_H) / MAP_WINDOW_H / scale / storage
    win_wait = _moving_sum(wsum, MAP_WINDOW_H)
    wait = np.where(win_entries >= 3, win_wait / np.maximum(win_entries, 1), 0.0)
    return flow, occ, wait


def _band(a: float, b: float) -> slice:
    return slice(int(round(a / STEP_H)), int(round(b / STEP_H)))


def section_roads(results: Path, links, scale: float, t_s_h: float) -> str:
    f = results / "drive_links.npz"
    if not f.exists():
        return _note("No drive_links.npz in this run (written by mimos.run).")
    t = np.load(f)
    k = links.index(t["u"], t["av"])
    delay_s = 3600 * (t["t_exit"] - t["t_end"])
    b = _bin(t["t_enter"])
    out = []
    x = np.arange(N_STEPS) * STEP_H
    ent = _moving_sum(np.bincount(b, minlength=N_STEPS).astype(float), CURVE_WINDOW_H)
    ws = _moving_sum(np.bincount(b, weights=delay_s, minlength=N_STEPS), CURVE_WINDOW_H)
    fig = go.Figure(go.Scatter(x=x, y=ent / CURVE_WINDOW_H / scale, line=dict(color="#e41a1c")))
    fig.update_layout(title="Cars entering road links per hour (scaled to the population)",
                      xaxis_title="hour of day", yaxis_title="cars per hour")
    fig.update_xaxes(dtick=2)
    out.append(_fig_html(fig, 360, "roads_hourly_flow"))
    fig = go.Figure(go.Scatter(x=x, y=np.where(ent > 0, ws / np.maximum(ent, 1), np.nan), line=dict(color="black")))
    fig.update_layout(title="Mean wait at the end of a road link", xaxis_title="hour of day", yaxis_title="seconds")
    fig.update_xaxes(dtick=2)
    out.append(_fig_html(fig, 360, "roads_hourly_wait"))

    # candidate links: the busiest ones of the day (relative to their capacity)
    daily = np.bincount(k, minlength=len(links.u)) / np.maximum(links.lanes, 1)
    cand = np.argsort(daily)[::-1][:20000]
    cand = cand[daily[cand] > 0]
    flow, occ, wait = link_series(links, t, k, scale, t_s_h, cand)
    flow_s, occ_s = scheme_percent("of flow capacity"), scheme_percent("of storage")
    full = lambda v: np.bincount(cand, weights=v, minlength=len(links.u))  # noqa: E731  (values on all links)
    layers = []
    for a, bb in BANDS:
        layers.append((f"flow {a}-{bb} h", full(flow[_band(a, bb)].mean(axis=0)), flow_s))
    for a, bb in BANDS:
        layers.append((f"occupancy {a}-{bb} h", full(occ[_band(a, bb)].mean(axis=0)), occ_s))
    for a, bb in BANDS:
        w = wait[_band(a, bb)]
        layers.append((f"wait {a}-{bb} h", full(w.sum(axis=0) / np.maximum((w > 0).sum(axis=0), 1)), WAIT_SCHEME))
    out.append(_fig_html(_layered_map(links, layers, "Roads (scaled to the population)"), 640, "roads_map", embed=False))
    out.append(_animated_link_map(links, cand, [
        ("flow, % of capacity", "Flow: vehicles per hour as % of the link's flow capacity", flow, flow_s),
        ("occupancy, % of storage", "Occupancy: mean cars on the link as % of its storage capacity", occ, occ_s),
        ("mean wait (s)", "Mean wait at the end of the link (links with 3+ simulated cars)", wait, WAIT_SCHEME)],
        "Roads through the day", "roads_anim"))
    out.append(_note(
        "Every 10 minutes over a moving one-hour window (curves: half-hour window). <b>Flow</b>: vehicles per "
        "hour entering the link, scaled to the population, as % of its flow capacity "
        f"(lanes x 3600 / t_s = {1 / t_s_h:.0f} vehicles per hour per lane). <b>Occupancy</b>: mean number of cars "
        "on the link (moving or queuing), scaled, as % of its storage capacity (cars it holds); near 100 % means "
        "queues fill the link. <b>Wait</b>: mean time a car waits at the end of the link before entering the next "
        "one (gap, storage, priority, FIFO); a per-car quantity, not scaled. --png exports the peak hours as "
        "images. Grey: main roads."))
    return "".join(out)


# ---- 5. Public transport --------------------------------------------------------------------

def section_public(results: Path, pub, links, scale: float) -> str:
    fn, ft = results / "exec_public_paths.csv", results / "exec_public_passage_times.csv"
    if not fn.exists():
        return _note("No public trajectories in this run.")
    board_node, board_hour = [], []
    for nodes, times in read_trajectories(fn, ft):
        riding = False
        for j in range(len(nodes) - 1):
            a, b = nodes[j], nodes[j + 1]
            ride = a >= 0 and b >= 0 and pub.next_stop[a] == b and pub.mode[a] >= 0
            if ride and not riding:
                board_node.append(a)
                board_hour.append(int(times[j]) % 24)
            riding = ride
    out = []
    if not board_node:
        return _note("Nobody boarded a vehicle.")
    board_node, board_hour = np.array(board_node), np.array(board_hour)
    fig = go.Figure(go.Bar(x=np.arange(24), y=np.bincount(board_hour, minlength=24)[:24] / scale,
                           marker_color=COLORS["pt"]))
    fig.update_layout(title="Boardings per hour (scaled to the population)", xaxis_title="hour of day")
    out.append(_fig_html(fig, 340, "public_boardings_hourly"))
    # stops: group nodes at the same place (one node per itinerary and stop)
    key = np.round(pub.lat[board_node], 4) * 1e5 + np.round(pub.lon[board_node], 4)
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    first = np.array([board_node[inv == g][0] for g in range(len(uniq))])
    size = 6 + 30 * np.sqrt(counts / counts.max())
    center = (float(np.average(pub.lon[first], weights=counts)), float(np.average(pub.lat[first], weights=counts)))
    fig = _map_figure(links, center, "Boardings per stop (scaled to the population; hover for values)")
    fig.add_trace(_lines(pub.lat[first], pub.lon[first], mode="markers", showlegend=False,
                         marker=dict(size=size, color=COLORS["pt"], opacity=0.6),
                               text=[f"{pub.name[n]}: {c / scale:.0f} boardings/day" for n, c in zip(first, counts)],
                               hoverinfo="text"))
    out.append(_fig_html(fig, 560, "public_boardings_stops"))
    return "".join(out)


# ---- 6. Emissions ---------------------------------------------------------------------------

def section_emissions(results: Path, links, scale: float, n_agents: int) -> str:
    f = results / "emissions" / "E.npz"
    if not f.exists():
        return _note("No emissions computed for this run. Run <code>uv run mimos emissions &lt;run_dir&gt;</code>.")
    from mimos.emissions import Settings, traversal_grams

    e = np.load(f, allow_pickle=False)
    E, pollutants = e["E"], [str(p) for p in e["pollutants"]]  # E[link, hour, pollutant] (thesis output)
    summary_file = results / "emissions" / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    out = [_table([(name, f"{E[:, :, p].sum() / 1000:.3g} kg sample, {E[:, :, p].sum() / scale / 1000:.3g} kg scaled to "
                    "the population" if name not in ("EC", "SPN23") else
                    f"{E[:, :, p].sum() / scale:.3g} {'MJ' if name == 'EC' else 'particles'} scaled to the population")
                   for p, name in enumerate(pollutants)])]

    # 10-minute resolution from the emissions of every link traversal (same calculation and
    # vehicle draw as E; E is their hourly sum)
    shown = [p for p in ("NOx", "PM Exhaust", "CO") if p in pollutants]
    t = np.load(results / "drive_links.npz")
    G, k, _, _ = traversal_grams(dict(t), links, n_agents, Settings.of_run(results), pollutants=tuple(shown))
    b = _bin(t["t_enter"])
    x = np.arange(N_STEPS) * STEP_H
    fig = go.Figure()
    for i, name in enumerate(shown):
        rate = _moving_sum(np.bincount(b, weights=G[:, i], minlength=N_STEPS), CURVE_WINDOW_H) / CURVE_WINDOW_H / scale
        fig.add_trace(go.Scatter(x=x, y=rate / 1000, name=f"{name} (kg/h)"))
    fig.update_layout(title="Emissions through the day, kg per hour (scaled to the population)",
                      xaxis_title="hour of day", yaxis_title="kg/h", yaxis_type="log")
    fig.update_xaxes(dtick=2)
    out.append(_fig_html(fig, 360, "emissions_hourly"))

    length = np.where(links.length_km > 0, links.length_km, np.nan)
    daily = np.bincount(k, weights=G[:, 0], minlength=len(links.u)) / np.nan_to_num(length, nan=np.inf)
    cand = np.argsort(daily)[::-1][:20000]
    cand = cand[daily[cand] > 0]
    pos = np.full(len(links.u), -1)
    pos[cand] = np.arange(cand.size)
    j = pos[k]
    sel = j >= 0
    layers, anim, scale_notes = [], [], []
    vkm = np.nansum(links.length_km[k])
    for i, name in enumerate(shown):
        per_bin = np.zeros((N_STEPS, cand.size))
        np.add.at(per_bin, (b[sel], j[sel]), G[sel, i])
        g = _moving_sum(per_bin, MAP_WINDOW_H) / MAP_WINDOW_H / length[cand] / scale  # g/km/h, population
        sch = scheme_quantiles(g, f"g/km/h {name}")
        for a, bb in BANDS:
            layers.append((f"{name} {a}-{bb} h", np.bincount(cand, weights=g[_band(a, bb)].mean(axis=0),
                                                             minlength=len(links.u)), sch))
        anim.append((f"{name} (g/km/h)", f"{name} emissions, g per km per hour", np.nan_to_num(g), sch))
        per_car_km = float(G[:, i].sum() / vkm) if vkm > 0 else float("nan")
        scale_notes.append(f"{name}: the average simulated car emits {per_car_km:.3g} g/km, so 1 g/km/h on a link "
                           f"is the emission of about {1 / per_car_km:.0f} average cars per hour")
    out.append(_fig_html(_layered_map(links, layers, "Emissions, g per km per hour (scaled to the population)"), 640,
                         "emissions_map", embed=False))
    out.append(_animated_link_map(links, cand, anim, "Emissions through the day", "emissions_anim"))
    out.append(_note(
        "Every 10 minutes over a moving one-hour window, scaled to the population (curves: half-hour window). "
        "Sense of scale: " + "; ".join(scale_notes) + ". These are <b>emissions</b> (mass released per km of road), "
        "not concentrations: air-quality limits (EU/WHO, e.g. NO2 40 µg/m³ as an annual mean) apply to "
        "concentrations, which also depend on wind, street geometry and background pollution, so they cannot be "
        "read from these maps. Method: thesis 'Preliminary LEZ results'; choices and limits in "
        "the docstring of mimos.emissions."))
    return "".join(out)


# ---- Assembly ---------------------------------------------------------------------------------

CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0 auto;max-width:1200px;padding:0 16px;color:#222}
h1{margin-top:24px} h2{border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:40px}
nav a{margin-right:14px} table.kv{border-collapse:collapse;font-size:14px}
table.kv th{text-align:left;padding:3px 12px 3px 0;color:#555;font-weight:500} table.kv td{padding:3px 0}
.note{color:#555;font-size:14px;max-width:900px}
.anim{margin:8px 0;font-size:14px} .anim select,.anim button{font-size:14px;padding:2px 8px}
"""


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")[:60]


def export_png(folder: Path, width: int = 1920, height: int = 1080, scale: float = 2.0,
               font_size: int = 16) -> list[Path]:
    """Every figure of the last built report as PNG (1920x1080 at 2x by default): one file per
    figure, one per layer for the layered maps (their buttons), and the 8 h frame for the
    hourly animation (the full day is in ``mimos video``). Needs kaleido and a Chrome/Edge."""
    import plotly.io as pio

    folder.mkdir(parents=True, exist_ok=True)
    figs, paths = [], []
    for n, (name, fig) in enumerate(EXPORT, start=1):
        f = go.Figure(fig)
        f.frames = []
        f.update_layout(sliders=[], height=None, width=None)
        menus = fig.layout.updatemenus
        layer_buttons = [b for m in menus for b in m.buttons if b.method == "update"] if menus else []
        f.layout.updatemenus = ()  # buttons do nothing in a PNG
        if layer_buttons:
            for b in layer_buttons:
                g = go.Figure(f)
                for tr, vis in zip(g.data, b.args[0]["visible"]):
                    tr.visible = vis
                g.update_layout(title=b.args[1]["title"].replace(": ", "<br>", 1), margin=dict(t=110))
                g.update_xaxes(title_text=None)  # map PNGs: the legend sits where the axis title was
                g.update_yaxes(title_text=None)
                figs.append(g)
                paths.append(folder / f"{n:02d}_{name}_{_slug(b.label)}.png")
        else:
            figs.append(f)
            paths.append(folder / f"{n:02d}_{name}.png")
    for f in figs:
        f.update_layout(font=dict(size=font_size))
    if hasattr(pio, "write_images"):  # one browser session for all figures (plotly >= 6.1)
        pio.write_images(figs, paths, width=width, height=height, scale=scale)
    else:
        for f, path in zip(figs, paths):
            f.write_image(path, width=width, height=height, scale=scale)
    return paths


def build_report(run_dir: Path | str, targets: Path | str | None = None, out: Path | str | None = None,
                 tiles: bool = True, png: bool = False, view: dict | None = None,
                 png_size: tuple[int, int] = (1920, 1080), png_font: int = 16) -> Path:
    """Writes the report. tiles=False draws maps on the model's road network instead of online
    map tiles (offline, slightly larger file). png=True also exports every figure as PNG to
    <run_dir>/report/figures/. view = {center_lat, center_lon, radius_km} overrides the run's [view]."""
    TILES["enabled"] = tiles
    EXPORT.clear()
    results = Path(run_dir)
    with (results / "config.toml").open("rb") as f:
        saved = tomllib.load(f)
    cfg = saved["config"]
    v = view or saved.get("view") or {}  # explicit view, else the one saved with the run
    VIEW["center"] = (v["center_lat"], v["center_lon"]) if "center_lat" in v and "center_lon" in v else None
    VIEW["radius_km"] = float(v.get("radius_km", 6.0))
    info_file = results / "run_info.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    scale = float(info.get("capacity_scale", 1.0)) or 1.0
    tg = load_targets(targets) if targets else None
    it = pd.read_csv(results / "iterations.csv")
    trips = pd.read_csv(results / "trips.csv")
    links = load_drive_links(cfg["data_dir"])
    pub = load_public_nodes(cfg["data_dir"])

    sections = [("run", "Run", section_run(cfg, info, it)),
                ("relaxation", "Relaxation", section_relaxation(it, tg)),
                ("trips", "Trips", section_trips(trips, tg)),
                ("roads", "Roads", section_roads(results, links, scale, float(cfg["t_s"]))),
                ("public", "Public transport", section_public(results, pub, links, scale)),
                ("emissions", "Emissions", section_emissions(results, links, scale, int(info.get("n_agents", 0))))]
    nav = "".join(f"<a href='#{key}'>{title}</a>" for key, title, _ in sections)
    body = "".join(f"<h2 id='{key}'>{title}</h2>{content}" for key, title, content in sections)
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>MiMoS run report</title>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'><style>{CSS}</style>"
            f"<script>{get_plotlyjs()}</script><script>{ANIM_JS}</script></head><body><h1>MiMoS run report</h1>"
            f"<p class='note'>{html.escape(results.as_posix())}"
            f"{' - targets: ' + html.escape(str(targets)) if targets else ''}</p><nav>{nav}</nav>{body}</body></html>")
    path = Path(out) if out else results / "report" / "report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    if png:
        export_png(path.parent / "figures", width=png_size[0], height=png_size[1], font_size=png_font)
    return path
