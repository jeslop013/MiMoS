# Example scenarios: trying and testing MiMoS without data

The inputs of a real city take many GB and are not part of the repository. To try the model, show it to someone, or test it, MiMoS can **generate a complete example scenario** in the exact input format: a made-up town, built from a seed in a few seconds. Nothing is downloaded and nothing is committed.

```
uv sync                                    # once: builds the C++ core
uv run mimos demo                          # town -> 10 iterations -> emissions -> report (~30 s)
uv run mimos demo --video                  # ... plus a video of the morning peak (~1 min)
```

Open `examples/town/run/report/report.html` in a browser (it draws the town's own roads; there are no map tiles for a made-up place). Step by step instead:

```
uv run mimos example --out examples/town   # writes the data and examples/town/config.toml
uv run mimos run examples/town/config.toml
uv run mimos emissions examples/town/run
uv run mimos report examples/town/run --no-tiles
uv run mimos video examples/town/run --what vehicles --start 07:30 --end 08:30 --speed 60
```

Options of `mimos example` / `mimos demo`: `--agents` (default 5,000), `--scale` (sample share they represent, default 0.1: a town of 50,000 on roads and buses scaled to 10%, as MATSim does; 1 = the agents are everybody), `--iterations` (default 10), `--seed` (a different seed gives a different town of the same design), `--out`. From Python: `mimos.scenarios.town(path, n_agents=5000, seed=0)` returns the config path. `examples/` is git-ignored.

## The example town

Code: `src/mimos/scenarios/town.py` (parameters in `town.toml` next to it: round, made-up behavioural values that make the example plausible; nothing is calibrated). Placed at latitude 0, longitude 0 so that no real place is implied. Designed so that every mechanism of the model is used:

| Part | What it contains | What it exercises |
|---|---|---|
| Roads | an organic street network of about 700 junctions in a disc of 3.5 km radius: an irregular old core of narrow streets (20 km/h), grid-pattern suburbs at three orientations, sparser streets towards the edge; on top, a ring road (70 km/h, 2 lanes), eight radial arterials and a middle orbital road (50 km/h, 2 lanes, cycle lanes); everything else residential (30 km/h, 1 lane) | road hierarchy and junction priority, flow and storage capacity, dead ends and irregular blocks |
| River and bridges | a meandering river with five bridges: two on the ring road, two arterial bridges and an old single-lane bridge by the old core | the bottleneck: at the morning peak the bridges run near capacity, queues build upstream (spill-back, FIFO exit, stuck time) |
| Roundabouts | in the centre and on the middle orbital road | priority rules at a different junction type |
| Village | 10 km east on a regional road (90 km/h) | longer trips where car and bus compete |
| Bikes | the same streets both ways, cycle lanes on the arterials, a bike-only bridge | bike routing and the cycleway flag |
| Public transport | five urban bus lines through the centre and a circular line, all following the roads (both directions; every 8-10 min at peaks, 15 min otherwise, 06:00-23:00), a regional bus to the village every 30 min, 11 rental-bike stations, walking transfers within 800 m | timetable routing, transfers, rental bikes, vehicle occupancy |
| Population | workers (55%), students (20%), others (25%); jobs, shops and leisure mostly in the old core, a business park south of the river, a campus in the north-west; homes spread over both banks; 10% living in the village, 1% of workers with a job 60 km away | agendas with several activities, priorities, long trips |
| Access candidates | 3 nearest drive and bike nodes of every location; nearest stop of every itinerary and nearest rental station | the precomputed access/egress files, written by the generator itself |

The streets depend on the seed: another seed gives another town of the same design. Typical result (5,000 agents, 10 iterations): walk ~26%, car ~40%, public transport ~23%, bike ~11% of trips; the mean wait at the end of a link is ~10 s at the morning peak and ~0 at midday; no gridlock. It is a demonstration, not a validated city.

## Use in tests

`tests/test_town.py` generates small towns and runs them: same seed, byte-identical files; every agent finishes, invariants hold, every main mode is used. These tests need no external data, so they run on any machine and in continuous integration. The tiny hand-checkable scenarios of `tests/scenario.py` (a 4-node corridor, small junctions) stay for unit tests with exact expected values.

## Other ways to get example data (not implemented)

| Option | How | For | Against |
|---|---|---|---|
| **Synthetic town** (implemented) | generated from a seed | offline, any size, reproducible, no licence issues, covers every mechanism | not a real place |
| **Small real area downloaded on demand** | a script fetches a few km of a real city from OpenStreetMap (osmnx) and GTFS, plus a synthetic population | real streets and map tiles | needs internet and osmnx; results change when OSM changes; needs the OSM/GTFS importers (planned, see [roadmap.md](roadmap.md)) |
| **Small real extract committed to git** | a cut of a real city's inputs, a few MB | real data, no tooling | data in git; the licence of the population data must allow it |

New model features (traffic signals, low-emission zones) will get their own example and test scenarios when they exist.
