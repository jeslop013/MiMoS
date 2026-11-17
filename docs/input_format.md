# MiMoS input format

A MiMoS scenario is one folder, the **`data_dir`** of the run configuration, with the files below. MiMoS reads them as they are: preparing them from raw sources (OpenStreetMap, timetables, a synthetic population) is done before MiMoS, by the user or by a separate pre-processing project. The generated example town (`mimos example`, `src/mimos/scenarios/town.py`) writes every file of this format and is the reference implementation; `tests/scenario.py` writes minimal versions.

```
<data_dir>/
  net_files/
    drive_network/   nodes, adj, inv_adj, times, lengths, lanes, capacity, velocity, priority   (+ highway, link_geometry.npz)
    bike_network/    nodes, adj, times, cycleway                                                (+ lengths)
    public_network/  nodes, adj, times, lengths, schedules, transport_id
  pop_files/
    lat, lon, t_start, t_end, prio                                                              (+ tags, attributes, global_id, t_duration)
    distances_pre_computation/  {drive,bike,public}_{nodes,times}_{source,target}
  fleet.csv          (optional) vehicle fleet for `mimos emissions`
  validation/        (optional) survey and measurement data, free format; not read by MiMoS
```

## General conventions

- **CSV without header**, comma-separated, one row per node (networks) or per agent (population). Line endings LF or CRLF.
- **Row number = id.** Node `k` of a network is row `k` (0-based) of its `nodes_*.csv`, and row `k` of every other file of that network describes node `k`. Ids used inside rows (adjacency, access candidates) are these row numbers.
- **Adjacency files are row-aligned**: in `adj_*.csv` row `u` lists the end nodes of the links leaving `u`; in the attribute files (`times`, `lengths`, `lanes`, ...) row `u` holds one value per link, **in the same order**. A link is identified by `(u, av)`: start node and position in the row.
- **A node without outgoing links** has the single value `-1` in its adjacency row and in every attribute row.
- **Units**: link times in **minutes**, lengths in **metres**, speeds in **km/h**; times of day and access times in **hours** (agendas run past midnight: 25.5 = 01:30 next day).
- Coordinates in **degrees, WGS84** (lat, lon).

## Road network: `net_files/drive_network/`

| File | Row `u` contains | Notes |
|---|---|---|
| `nodes_drive.csv` | `id, lat, lon` | `id` is an external label (e.g. the OSM id); the model uses the row number |
| `adj_drive.csv` | end node of each outgoing link | directed: a two-way street is two links |
| `inv_adj_drive.csv` | start node of each incoming link | must be consistent with `adj` (every link `u -> v` appears as `u` in row `v`) |
| `times_drive.csv` | free-flow time per link, minutes | length / speed |
| `lengths_drive.csv` | length per link, metres | |
| `lanes_drive.csv` | lanes per link | integer; values < 1 are set to 1 (logged) |
| `capacity_drive.csv` | storage capacity per link, vehicles | integer, `lanes * length / 4.5 m`; < 1 is set to 1 |
| `velocity_drive.csv` | speed limit per link, km/h | |
| `priority_drive.csv` | road class per link (lower = higher priority at junctions) | 0 roundabout, 1 motorway, 2 trunk, 3 primary, 4 secondary, 5 tertiary, 6 residential/unclassified; **+0.5** for link (slip) roads |
| `highway_drive.csv` | *(optional)* OSM `highway` tag per link | descriptive only |
| `link_geometry.npz` | *(optional)* real shape of every link, for maps and videos | arrays `start` (int, length = number of links + 1), `lon`, `lat` (float32): link `k` (links counted row by row in `adj` order) is the polyline `lon/lat[start[k]:start[k+1]]`; ignored if its size does not match; without it links are drawn straight |

## Bike network: `net_files/bike_network/`

| File | Row `u` contains | Notes |
|---|---|---|
| `nodes_bike.csv` | `id, lat, lon` | |
| `adj_bike.csv` | end node of each outgoing link | usually both directions |
| `times_bike.csv` | riding time per link, minutes | |
| `cycleway_bike.csv` | `1` cycleway, `0` shared road, per link | (`legacy_mode` reads it as the thesis version did: 0 also counts as cycleway) |
| `lengths_bike.csv` | *(optional)* length per link, metres | descriptive only |

## Public transport: `net_files/public_network/`

Nodes are **stops of an itinerary** (one direction of one line: the same physical stop served by two itineraries is two nodes) and **rental-bike stations**.

`nodes_public.csv`, 8 columns: `mode, itinerary, stop, lat, lon, name, fee, circular`

| Column | Meaning |
|---|---|
| `mode` | `-1` rental-bike station; `0` urban bus; `1` regional bus; `2` train; `3` and `4` other urban and regional bus services. Vehicle capacity by mode: 0 and 3 `city_bus_cap`, 1 and 4 `reg_bus_cap`, 2 `cercanias_cap` (config) |
| `itinerary` | itinerary id (`-1` for rental-bike stations) |
| `stop` | position of the stop along its itinerary, from 0 |
| `name` | free text (line name) |
| `fee` | fare per boarding (read; fares are currently set with `ticket` / `daily_cost_public` in the config) |
| `circular` | `1` circular line (timetables wrap around), else `0` |

| File | Row `u` contains |
|---|---|
| `adj_public.csv` | **first entry: the next stop of the same itinerary** (absent for the last stop); **then walking transfers** to nodes of other itineraries and to rental-bike stations. A rental-bike station lists the other stations it can ride to |
| `times_public.csv` | per link: riding time to the next stop, or walking / cycling time of a transfer or rental-bike ride, minutes |
| `lengths_public.csv` | per link, metres |
| `schedules_public.csv` | departure times of the vehicles at this stop, hours, ascending; `-1` for rental-bike stations (always available) |
| `transport_id_public.csv` | the vehicle id of each departure in `schedules`, same length; **a vehicle keeps its id along the itinerary**, which is how occupancy is tracked; `-1` for rental-bike stations |

Consistency: for each vehicle, `schedule[next stop] - schedule[stop]` should equal the riding time of the link (the simulation moves riders with the link times). Walking transfers are usually limited to a few hundred metres, one per other itinerary.

## Population: `pop_files/`

One row per agent; the same row in every file is the same agent. Columns are the **agenda locations** in visiting order; an agenda with `L` locations makes `L - 1` trips.

| File | Row = one agent; column `i` = location `i` |
|---|---|
| `lat.csv`, `lon.csv` | coordinates of the location |
| `t_start.csv` | start of the activity there, hours (the first location, home, starts at the beginning of the day, e.g. 4.0) |
| `t_end.csv` | departure from the location, hours (the last value ends the simulated day, e.g. 27.75) |
| `prio.csv` | priority of the activity, integer, lower = more important (e.g. 0 work or study, 2-3 errands and leisure, 4 home); used when a trip does not fit between activities |
| `tags.csv`, `attributes.csv`, `global_id.csv`, `t_duration.csv` | *(optional, descriptive)* activity labels, personal attributes, ids, durations |

`n_agents` in the config selects rows: the first ones, or a random sample (`sample_random`). Road and vehicle capacities are scaled to the sample share (`capacity_scale`).

## Access candidates: `pop_files/distances_pre_computation/`

Where each trip can enter and leave each network. 12 files: `{drive,bike,public}_{nodes,times}_{source,target}.csv`.

- One row per agent (same order as the population). For each **trip** in order, a list of candidate nodes; **trips are separated by `-1`**. The `times` file has the same layout (with `-1` at the separators).
- `source` = candidates near the trip's **origin** (location `i`), `target` = near its **destination** (location `i + 1`).
- `times` = the off-network leg in **hours**: **driving at 15 km/h** for `drive`, **walking at 4 km/h** for `bike` and `public`. (The access score uses `length = speed x time`.)
- Typical candidates: `drive`, `bike`: the few nearest network nodes (for example the 3 nearest, or the nodes within an expanding radius). `public`: **the nearest stop of every itinerary** and the nearest rental-bike station; a candidate with `itinerary >= 0` gets the boarding constant, a rental station the rental-bike constant.
- Every trip needs at least one candidate per mode.

## Fleet: `fleet.csv` (optional)

Vehicle fleet for `mimos emissions`, with a header: `category, fuel, segment, euro, share` (EEA/COPERT classification, `share` normalised by MiMoS). Without it, the example fleet shipped with MiMoS (`src/mimos/data/fleet_example.csv`) is used. Emission factors (`src/mimos/data/hot_ef.csv`, EEA Appendix 4) are generic and ship with MiMoS; sources in `src/mimos/data/SOURCES.md`.

## Checking a scenario

Until a validator exists (planned, see [roadmap.md](roadmap.md)), the quickest check is to run 1-2 iterations with a few hundred agents: the loader reports malformed rows, and `check_invariants` and `iterations.csv` show agents that never finish.
