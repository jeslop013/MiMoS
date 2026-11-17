// Model lifecycle: loading, per-iteration resets, event-queue seeding, invariant checks and
// the complete cycle (run()). See model.hpp for the list of stages.
#include "mimos/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace mimos {

Model::Model(Config c) : cfg(std::move(c)), mt(cfg.seed) {
    log.open(cfg.run_dir + "log.txt", cfg.log_level);
}

void Model::load() {
    MIMOS_LOG(log, INFO, "MiMoS: N = %d, it_number = %d, alpha_l_i = %f, alpha_l_f = %f, k_max = %d",
              cfg.n_agents, cfg.it_number, cfg.alpha_l_i, cfg.alpha_l_f, cfg.k_max);

    // The link-exit options change the dynamics, which legacy_mode must reproduce.
    if (cfg.legacy_mode && (cfg.fifo_link_exit || cfg.stuck_time > 0))
        throw std::invalid_argument("fifo_link_exit and stuck_time require legacy_mode = false");

    net = load_networks(cfg.data_dir, log);
    pop = load_population(cfg, net.pub, log);
    N = cfg.n_agents;

    // ---- Interpretation of raw network attributes -----------------------------------------
    // Cycleway flag: 1 = cycleway (legacy_mode: every value >= 0, as the thesis version).
    net.bike.cycle.clear();
    for (const auto& row : net.bike.cycle_raw) {
        std::vector<bool> flags;
        for (int v : row) flags.push_back(cfg.legacy_mode ? !(v < 0) : v > 0);
        net.bike.cycle.push_back(flags);
    }

    // Links need at least one lane (gap) and room for one car, otherwise nobody can ever enter
    // them and every route through them ends in a phantom gridlock (for example 0 lanes from
    // an unparsed OSM lane string). legacy_mode keeps the values as loaded.
    auto& D = net.drive;
    if (!cfg.legacy_mode) {
        int fixed_lanes = 0, fixed_capacity = 0;
        for (size_t u = 0; u < D.adj.size(); ++u)
            for (size_t av = 0; av < D.adj[u].size(); ++av) {
                if (D.adj[u][av] < 0) continue;
                if (D.lanes[u][av] < 1) { D.lanes[u][av] = 1; ++fixed_lanes; }
                if (D.capacity[u][av] < 1) { D.capacity[u][av] = 1; ++fixed_capacity; }
            }
        if (fixed_lanes + fixed_capacity > 0)
            MIMOS_LOG(log, WARNING, "Drive network: %d links had < 1 lane and %d links had capacity < 1; set to 1", fixed_lanes, fixed_capacity);
    }

    // Sample scaling (Config::capacity_scale; <= 0 means sample share n_agents / population).
    capacity_scale = cfg.capacity_scale > 0 ? cfg.capacity_scale
                                            : static_cast<double>(N) / pop.population_size;
    MIMOS_LOG(log, INFO, "Capacity scale: %f (%d of %d agents)", capacity_scale, N, pop.population_size);
    // Road storage actually simulated (min 1 car per link).
    drive_capacity = D.capacity;
    if (capacity_scale != 1.0)
        for (size_t u = 0; u < D.adj.size(); ++u)
            for (size_t av = 0; av < D.adj[u].size(); ++av)
                if (D.adj[u][av] >= 0)
                    drive_capacity[u][av] = std::max(1, static_cast<int>(std::floor(D.capacity[u][av] * capacity_scale)));

    // Vehicle ownership (thesis "Vehicle ownership constraints"): N draws for cars, then N
    // draws for bikes.
    std::discrete_distribution<> car({1.0 - cfg.p_car, cfg.p_car});
    for (int i = 0; i < N; ++i) pop.owns_car.push_back(car(mt) == 1);
    std::discrete_distribution<> bk({1.0 - cfg.p_bike, cfg.p_bike});
    for (int i = 0; i < N; ++i) pop.owns_bike.push_back(bk(mt) == 1);

    // ---- Planning and execution containers ------------------------------------------------
    for (ModePlan* m : {&bike, &pub, &drive}) {
        m->preferred.assign(N, false);
        m->scores.assign(N, 0.0);
        m->times.assign(N, 0.0);
        m->paths.assign(N, {});
        m->access.assign(N, {});
        m->passage.assign(N, {});
        m->departure.assign(N, {});
    }
    for (int i = 0; i < N; ++i) {
        const int trips = pop.n_trips(i);
        // Bike paths are appended trip by trip in plan_bike(); public/drive are indexed by trip.
        for (ModePlan* m : {&pub, &drive}) {
            m->paths[i].assign(trips, {});
            m->access[i].assign(trips, {});
            m->passage[i].assign(trips, {});
        }
        // Departures start as the agenda end times of every activity but the last.
        for (ModePlan* m : {&bike, &pub, &drive})
            m->departure[i].assign(pop.end[i].begin(), pop.end[i].end() - 1);
    }

    exec_public.assign(N, false);
    exec_drive.assign(N, false);
    exec_bike.assign(N, false);
    traj_bike.reset(N);
    traj_public.reset(N);
    traj_drive.reset(N);

    short_public_mem.assign(N, {});
    short_drive_mem.assign(N, {});
    long_public_mem.assign(N, {});
    long_drive_mem.assign(N, {});
    for (int i = 0; i < N; ++i) {
        long_public_mem[i].assign(pop.n_trips(i), {});
        long_drive_mem[i].assign(pop.n_trips(i), {});
    }
    // Routing workspaces are created per thread on first use (routing.cpp, for_each_agent).
}

// Per-iteration resets.
void Model::start_iteration(int it) {
    MIMOS_LOG(log, INFO, "--------------------------------- ITERATION %d ---------------------------------", it);
    bike.preferred.assign(N, false);
    pub.preferred.assign(N, false);
    drive.preferred.assign(N, false);
    traj_public.reset(N);
    traj_drive.reset(N);
    for (int i = 0; i < N; ++i) {
        short_public_mem[i].assign(pop.n_trips(i), {});
        short_drive_mem[i].assign(pop.n_trips(i), {});
    }
}

// Seeds the two event queues with every agent's first departure.
void Model::prepare_execution(int it) {
    (void)it;
    public_queue_ = PublicQueue();
    drive_queue_ = DriveQueue();
    in_public_simulation_ = 0;
    in_drive_simulation_ = 0;
    finished_exec.assign(N, 0);

    for (int i = 0; i < N; ++i) {
        if (exec_drive[i] == true) {
            drive_queue_.push(DriveEvent(drive.departure[i][0], true, i, 0, -1));
            ++in_drive_simulation_;
            finished_exec[i] = -1;
        } else if (exec_public[i] == true) {
            // boarding = 1e6: walking to a stop / at an agenda location (see types.hpp)
            public_queue_.push(PublicEvent(pub.departure[i][0], 1000000, i, 0, -1));
            ++in_public_simulation_;
            finished_exec[i] = -1;
        }
    }
}

// Vehicle capacity of the itinerary starting at header node u, scaled
// like the roads when simulating a sample (min 1 seat).
int Model::public_capacity(int u) const {
    int cap = 0;
    switch (net.pub.mode[u]) {
        case 0: cap = cfg.city_bus_cap; break;
        case 1: cap = cfg.reg_bus_cap; break;
        case 2: cap = cfg.cercanias_cap; break;
        case 3: cap = cfg.city_bus_cap; break;
        case 4: cap = cfg.reg_bus_cap; break;
    }
    if (capacity_scale != 1.0) cap = std::max(1, static_cast<int>(std::lround(cap * capacity_scale)));
    return cap;
}

// Conservation checks after the simulations.
// Returns true if every vehicle is empty again and every road has its full capacity back.
// Violations are logged and returned, so tests can assert on them.
bool Model::check_invariants(int it) {
    (void)it;
    bool ok = true;
    for (int i = 0; i < N; ++i)
        if (finished_exec[i] < -0.5) {
            MIMOS_LOG(log, WARNING, "Agent %d hasn't finished simulation", i);
            ok = false;
        }

    for (size_t u = 0; u < net.pub.size(); ++u)
        if (net.pub.stop[u] == 0 && net.pub.mode[u] > -0.5)
            for (int v : net.pub.id[u])
                if (space_map[v] != public_capacity(static_cast<int>(u))) {
                    MIMOS_LOG(log, CRITICAL, "Error: At node %zu, space_map[%d] = %d but should be %d", u, v, space_map[v], public_capacity(static_cast<int>(u)));
                    ok = false;
                }

    const auto& D = net.drive;
    for (size_t u = 0; u < drive_mouth_space.size(); ++u)
        for (size_t j = 0; j < drive_mouth_space[u].size(); ++j)
            if (drive_mouth_space[u][j] != D.lanes[u][j]) {
                MIMOS_LOG(log, CRITICAL, "Error: drive_mouth_space[%zu][%zu] = %d but should be %d", u, j, drive_mouth_space[u][j], D.lanes[u][j]);
                ok = false;
            }
    for (size_t u = 0; u < drive_link_space.size(); ++u)
        for (size_t j = 0; j < drive_link_space[u].size(); ++j)
            if (drive_link_space[u][j] != drive_capacity[u][j]) {
                MIMOS_LOG(log, WARNING, "Error: drive_link_space[%zu][%zu] = %d but should be %d", u, j, drive_link_space[u][j], drive_capacity[u][j]);
                ok = false;
            }
    return ok;
}

std::vector<std::vector<int>> Model::trip_modes() const {
    std::vector<std::vector<int>> modes(N);
    for (int i = 0; i < N; ++i) {
        const int trips = pop.n_trips(i);
        modes[i].assign(trips, TRIP_WALK);
        for (int il = 0; il < trips; ++il) {
            if (exec_drive[i]) {
                // a drive route has at least two nodes; otherwise the agent walked
                if (drive.paths[i][il].size() > 1) modes[i][il] = TRIP_CAR;
            } else if (exec_public[i]) {
                bool bus = false, rental = false;
                for (int node : pub.paths[i][il]) {
                    if (net.pub.mode[node] > -0.5) bus = true;
                    else rental = true;  // mode -1: rental bike station
                }
                if (bus) modes[i][il] = TRIP_PT;
                else if (rental) modes[i][il] = TRIP_BIKE;
            } else if (exec_bike[i]) {
                if (bike.paths[i][il].size() > 1) modes[i][il] = TRIP_BIKE;
            }
        }
    }
    return modes;
}

void Model::trip_times(std::vector<std::vector<double>>& departures,
                       std::vector<std::vector<double>>& durations) const {
    departures.assign(N, {});
    durations.assign(N, {});
    for (int i = 0; i < N; ++i) {
        const Trajectories* t = exec_drive[i] ? &traj_drive : exec_public[i] ? &traj_public
                              : exec_bike[i] ? &traj_bike : nullptr;
        const int trips = pop.n_trips(i);
        departures[i].assign(trips, std::nan(""));
        durations[i].assign(trips, std::nan(""));
        if (!t) continue;
        const auto& nodes = t->nodes[i];
        const auto& times = t->times[i];
        // Location il is recorded as -1-il: on leaving it (departure of trip il) and on
        // arriving at it (end of trip il-1, a second record if the agent waits there).
        size_t from = 0;
        for (int il = 0; il < trips; ++il) {
            double dep = std::nan("");
            for (size_t k = from; k < nodes.size(); ++k) {
                if (nodes[k] == -1 - il) dep = times[k];  // the last record = actual departure
                else if (nodes[k] == -2 - il) {
                    departures[i][il] = dep;
                    durations[i][il] = times[k] - dep;
                    from = k;
                    break;
                }
            }
        }
    }
}

std::vector<std::vector<double>> Model::trip_durations() const {
    std::vector<std::vector<double>> dep, dur;
    trip_times(dep, dur);
    return dur;
}

std::vector<std::vector<double>> Model::trip_departures() const {
    std::vector<std::vector<double>> dep, dur;
    trip_times(dep, dur);
    return dep;
}

std::vector<std::vector<double>> Model::trip_distances_km() const {
    std::vector<std::vector<double>> km(N);
    for (int i = 0; i < N; ++i)
        for (int il = 0; il < pop.n_trips(i); ++il)
            km[i].push_back(haversine_d(pop.lat[i][il], pop.lon[i][il], pop.lat[i][il + 1], pop.lon[i][il + 1]));
    return km;
}

std::vector<double> Model::executed_scores() const {
    const auto durations = trip_durations();
    std::vector<double> out(N, std::nan(""));
    for (int i = 0; i < N; ++i) {
        const ModePlan* m = nullptr;
        double beta = 0.0;
        if (exec_drive[i]) { m = &drive; beta = cfg.beta_trav_delay; }
        else if (exec_public[i]) { m = &pub; beta = cfg.beta_trav_walk; }
        else if (exec_bike[i]) { out[i] = bike.scores[i]; continue; }
        else continue;
        double t_exec = 0.0;
        for (double d : durations[i]) t_exec += d;  // NaN if a trip never ended
        out[i] = m->scores[i] + beta * (t_exec - m->times[i]);
    }
    return out;
}

Model::LinkTraversals Model::drive_link_traversals() const {
    LinkTraversals out;
    for (int i = 0; i < N; ++i) {
        if (!exec_drive[i]) continue;
        const auto& nodes = traj_drive.nodes[i];
        const auto& times = traj_drive.times[i];
        size_t k = 0;
        for (int il = 0; il < pop.n_trips(i); ++il) {
            const auto& path = drive.paths[i][il];
            // records of this trip: after the last -1-il record, up to the -2-il record
            size_t start = k, end = nodes.size();
            for (; k < nodes.size(); ++k) {
                if (nodes[k] == -1 - il) start = k + 1;
                else if (nodes[k] == -2 - il) { end = k; break; }
            }
            if (k >= nodes.size() || path.size() < 2) continue;  // unfinished trip, or walked
            // Network records come in one block per path node (no node is visited twice):
            // first record = arrival at the node, last record = entry into the next link.
            std::vector<double> first(path.size(), std::nan("")), last(path.size(), std::nan(""));
            size_t p = 0;
            for (size_t r = start; r < end; ++r) {
                while (p < path.size() && nodes[r] != path[p]) ++p;
                if (p == path.size()) break;
                if (std::isnan(first[p])) first[p] = times[r];
                last[p] = times[r];
            }
            for (size_t j = 0; j + 1 < path.size(); ++j) {
                if (std::isnan(last[j]) || std::isnan(first[j + 1])) continue;
                out.agent.push_back(i);
                out.trip.push_back(il);
                out.u.push_back(path[j]);
                out.av.push_back(drive.access[i][il][j]);
                out.t_enter.push_back(last[j]);
                out.t_end.push_back(first[j + 1]);
                out.t_exit.push_back(last[j + 1]);
            }
        }
    }
    return out;
}

// The complete cycle, with stage timings on stdout.
void Model::run() {
    using clock = std::chrono::steady_clock;
    auto timed = [](const char* what, auto&& f) {
        auto t0 = clock::now();
        f();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
        std::cout << what << ": " << ms << " ms" << std::endl;
    };

    timed("load", [&] { load(); });
    timed("plan bike", [&] { plan_bike(); });
    for (int it = 0; it < cfg.it_number; ++it) {
        start_iteration(it);
        timed("  plan", [&] { plan(it); });
        choose(it);
        prepare_execution(it);
        timed("  simulate public", [&] { simulate_public(it); });
        timed("  simulate drive", [&] { simulate_drive(it); });
        update_memory(it);
        check_invariants(it);
        if (it % cfg.print_it == 0) write_iteration(it);
    }
    write_final();
}

}  // namespace mimos
