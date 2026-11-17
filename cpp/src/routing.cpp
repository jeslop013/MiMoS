// Routing (thesis section "Routing routine", Algorithm "Utility-based routing algorithm").
//
// For every agent, trip and mode, a Dijkstra-like search MAXIMISES the linear route utility
// (thesis eq. U_route). "dist" is the best utility found for a node, "pred" its predecessor.
// Utilities only decrease along a path, so once a destination candidate is reached, every
// fringe node with a lower utility can be pruned ("cutoff").
//
// The three relaxation rules are kept separate (they really differ); the shared pieces are
// helpers below.
//
// Performance:
//  * Lazy deletion with version stamps instead of rebuilding the whole heap on every
//    relaxation. See Fringe below for why the pop sequence equals that of a heap that keeps
//    one entry per node.
//  * Per-thread workspaces with touched-list resets instead of re-allocating three arrays of
//    the size of the largest network for every trip.
//  * O(1) lookup of destination candidates (target_slot) instead of scanning the (possibly
//    hundreds of) candidates of a public trip on every relaxation.
//  * Agents are planned in parallel (OpenMP). Each agent's plan only reads shared data, and
//    writes its own entries, so the results do not depend on the number of threads.
#include <algorithm>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mimos/model.hpp"

namespace mimos {

// ------------------------------------------------------------------------------------------
// Workspace
// ------------------------------------------------------------------------------------------
void Model::RoutingWorkspace::init(size_t n) {
    dist.assign(n, -1e6);  // -1e6 is the "unreached" utility; results depend on it, keep it
    time.assign(n, -1e6);
    pred.assign(n, -1);
    version.assign(n, 0);
    target_slot.assign(n, -1);
    touched.clear();
}

void Model::RoutingWorkspace::set(int v, double d, double t, int p) {
    dist[v] = d;
    time[v] = t;
    pred[v] = p;
    touched.push_back(v);
}

void Model::RoutingWorkspace::reset() {
    for (int v : touched) {
        dist[v] = -1e6;
        time[v] = -1e6;
        pred[v] = -1;
        version[v] = 0;
    }
    touched.clear();
}

namespace {

using Workspace = Model::RoutingWorkspace;

// Max-heap of (utility, node) with lazy deletion.
// The fringe keeps at most one valid entry per improved node: each push of v increments
// version[v] and older entries are skipped when popped. Seeding pushes (access candidates)
// keep version 0, so duplicates inserted while seeding stay valid. Valid entries are
// ordered by (utility, node) like std::pair, so the sequence of processed nodes is the same.
struct FringeEntry {
    double score;
    int node;
    unsigned version;
};
struct FringeLess {
    bool operator()(const FringeEntry& a, const FringeEntry& b) const {
        return a.score < b.score || (a.score == b.score && a.node < b.node);
    }
};
using Fringe = std::priority_queue<FringeEntry, std::vector<FringeEntry>, FringeLess>;

// Initial access candidate: pushed without invalidating other entries.
inline void seed(Fringe& f, Workspace& w, int v, double score) { f.push({score, v, w.version[v]}); }

// Improved node: invalidates older entries of v.
inline void push_replacing(Fringe& f, Workspace& w, int v, double score) {
    ++w.version[v];
    f.push({score, v, w.version[v]});
}

// Pops the next valid node, or -1 when the fringe is exhausted.
inline int pop_valid(Fringe& f, const Workspace& w) {
    while (!f.empty()) {
        FringeEntry e = f.top();
        f.pop();
        if (e.version == w.version[e.node]) return e.node;
    }
    return -1;
}

// Marks the destination candidates of a trip (first occurrence wins).
void mark_targets(Workspace& w, const std::vector<Access>& tgt) {
    for (size_t s = 0; s < tgt.size(); ++s) {
        int v = std::get<2>(tgt[s]);
        if (w.target_slot[v] < 0) w.target_slot[v] = static_cast<int>(s);
    }
}
void unmark_targets(Workspace& w, const std::vector<Access>& tgt) {
    for (const auto& p : tgt) w.target_slot[std::get<2>(p)] = -1;
}

// Walks the predecessors back from the last network node of the best route. Returns an empty
// path if no route was found (pred_final < 0), or if a node repeats (corrupted predecessors;
// logs an error and falls back to walking).
void reconstruct_path(int pred_final, double current_t, const Workspace& w, std::vector<int>& path,
                      std::vector<double>& passage, Logger& log, int agent) {
    path.clear();
    passage.clear();
    if (pred_final > -0.5) {
        int node = pred_final;
        while (node > -0.5) {
            path.push_back(node);
            passage.push_back(current_t + w.time[node]);
            node = w.pred[node];
            if (std::count(path.begin(), path.end(), node) >= 2) {
                MIMOS_LOG(log, ERROR, "Detected double occurrence of node %d in agent's %d best_path. Aborting path reconstruction.", node, agent);
                path.clear();
                passage.clear();
                break;
            }
        }
        std::reverse(path.begin(), path.end());
        std::reverse(passage.begin(), passage.end());
    }
}

// av (position of the next node in adj[u]) for every link of the path.
std::vector<int> access_vector(const std::vector<int>& path, const std::vector<std::vector<int>>& adj) {
    std::vector<int> access;
    for (size_t b = 0; b + 1 < path.size(); ++b) {
        int iv = -1;
        for (int v : adj[path[b]]) {
            ++iv;
            if (v == path[b + 1]) {
                access.push_back(iv);
                break;
            }
        }
    }
    return access;
}

bool same_node_in_both(const std::vector<Access>& src, const std::vector<Access>& tgt) {
    for (const auto& t : src)
        for (const auto& p : tgt)
            if (std::get<2>(t) == std::get<2>(p)) return true;
    return false;
}

// Agent clock after trip il: arrive, or wait until the next planned departure.
double next_trip_time(double current_t, double time_final, const std::vector<double>& departure, int il) {
    if (il + 1.5 < departure.size()) {
        if (current_t + time_final > departure[il + 1]) return current_t + time_final;
        return departure[il + 1];
    }
    return -1;
}

// Runs `body(i, workspace)` for every agent, in parallel when OpenMP is available.
template <typename Body>
void for_each_agent(int n, int threads, std::vector<Workspace>& ws, size_t net_size, Body body) {
#ifdef _OPENMP
    int nt = threads > 0 ? threads : omp_get_max_threads();
#else
    int nt = 1;
    (void)threads;
#endif
    if (static_cast<int>(ws.size()) < nt) ws.resize(nt);
    for (int t = 0; t < nt; ++t)
        if (ws[t].dist.size() != net_size) ws[t].init(net_size);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8) num_threads(nt)
#endif
    for (int i = 0; i < n; ++i) {
#ifdef _OPENMP
        Workspace& w = ws[omp_get_thread_num()];
#else
        Workspace& w = ws[0];
#endif
        body(i, w);
    }
}

}  // namespace

// Agenda-priority departure correction (thesis "Model input": the activity hierarchy decides
// whether an agent leaves early or late when a trip does not fit between activities).
// Updates departure[il] and returns the time shift applied (also added to the agent's clock).
// With cfg.departure_correction = 0 (default) nothing changes, unless `correction` is
// not finite; then 0 * correction = NaN.
double Model::correct_departure(std::vector<double>& departure, int i, int il, double time_final) {
    const auto& start = pop.start[i];
    const auto& end = pop.end[i];
    const auto& prio = pop.prio[i];
    double correction = 0.0;
    if (start[il + 1] - departure[il] > time_final) {  // trip fits between activities
        if (prio[il + 1] <= prio[il])                  // destination at least as important: leave asap
            correction = end[il] - departure[il];
        else                                           // origin more important: leave as late as possible
            correction = start[il + 1] - time_final - departure[il];
    } else {                                           // trip does not fit
        if (prio[il + 1] < prio[il]) {
            correction = start[il + 1] - time_final - departure[il];
        } else if (prio[il + 1] > prio[il]) {
            correction = end[il] - departure[il];
        } else {                                       // same priority: share the overlap
            double alpha = (end[il + 1] - start[il + 1]) / (end[il] - start[il]);
            if (alpha < 1.0)
                correction = (1 / (1 - alpha)) * (end[il + 1] - start[il] - end[il] * alpha) - departure[il];
            else
                correction = end[il] - 0.5 * (end[il] + time_final - start[il + 1]) - departure[il];
        }
    }
    departure[il] = departure[il] + cfg.departure_correction * correction;
    return cfg.departure_correction * correction;
}

// ------------------------------------------------------------------------------------------
// Bike: no interaction, so it is planned once before the iterations.
// The executed bike trajectory is the planned one and is recorded here.
// ------------------------------------------------------------------------------------------
void Model::plan_bike() {
    MIMOS_LOG(log, INFO, "---------------------------------- BIKE PLANNING -------------------------------");
    progress_agents_.store(0, std::memory_order_relaxed);
    for_each_agent(N, cfg.threads, ws_, net.largest(), [&](int i, RoutingWorkspace& w) {
        progress_agents_.fetch_add(1, std::memory_order_relaxed);
        plan_bike_agent(i, w);
    });
}

void Model::plan_bike_agent(int i, RoutingWorkspace& w) {
    const auto& B = net.bike;
    double current_bike_t = bike.departure[i][0];

    for (int il = 0; il < pop.n_trips(i); ++il) {
        Fringe fringe;
        const auto& src = pop.bike_src[i][il];
        for (const auto& t : src) {
            int k = std::get<2>(t);
            w.set(k, std::get<0>(t), std::get<1>(t), -1);
            seed(fringe, w, k, w.dist[k]);
        }
        const auto& tgt = pop.bike_tgt[i][il];

        double cutoff = -1000000;
        double dist_final = -1000000;
        int pred_final = -1;
        double time_final = 0.0;

        // Skip routing when the mode constant makes it hopeless, or when origin and
        // destination share a network node (then the trip is walked).
        bool dijkstra = true;
        if (cfg.C_bike < cfg.C_limit) {
            dijkstra = false;
        } else if (same_node_in_both(src, tgt)) {
            double d = geo_km(pop.lat[i][il], pop.lon[i][il], pop.lat[i][il + 1], pop.lon[i][il + 1]);
            dist_final = cfg.C_walk + (cfg.beta_trav_walk / 4.0 + cfg.beta_d_walk) * d;
            time_final = geo_km(pop.lat[i][il], pop.lon[i][il], pop.lat[i][il + 1], pop.lon[i][il + 1]) / 4.0;
            dijkstra = false;
        }

        if (dijkstra) {
            mark_targets(w, tgt);
            for (int u = pop_valid(fringe, w); u >= 0; u = pop_valid(fringe, w)) {
                if (w.dist[u] < cutoff) continue;  // cannot beat the best route found

                const auto& time_row = B.times[u];
                int iv = -1;
                for (int v : B.adj[u]) {
                    ++iv;
                    if (v < 0) break;
                    // Bike distance is approximated as 15 km/h * time.
                    double alt = w.dist[u] + cfg.beta_trav_bike * time_row[iv] +
                                 (cfg.beta_d_bike + cfg.beta_m * cfg.gamma_d_bike) * time_row[iv] * 15.0;
                    if (B.cycle[u][iv] == false) alt += cfg.beta_trav_no_cycleway * (time_row[iv]);

                    if (alt > w.dist[v]) {
                        w.set(v, alt, w.time[u] + time_row[iv], u);
                        int s = w.target_slot[v];
                        if (s >= 0) {  // destination candidate: not expanded further
                            const auto& p = tgt[s];
                            if (w.dist[v] + std::get<0>(p) > dist_final) {
                                dist_final = w.dist[v] + std::get<0>(p);
                                pred_final = v;
                                time_final = w.time[v] + std::get<1>(p);
                                cutoff = dist_final;
                            }
                        } else {
                            push_replacing(fringe, w, v, alt);
                        }
                    }
                }
            }
            unmark_targets(w, tgt);
        }

        std::vector<int> best_path;
        std::vector<double> passage;
        reconstruct_path(pred_final, current_bike_t, w, best_path, passage, log, i);
        std::vector<int> access = access_vector(best_path, B.adj);
        w.reset();

        // Bike execution == plan: record the trajectory with agenda markers -il-1.
        traj_bike.add(i, -1 - il, current_bike_t);
        for (size_t b = 0; b < best_path.size(); ++b) traj_bike.add(i, best_path[b], passage[b]);
        traj_bike.add(i, -1 - il - 1, current_bike_t + time_final);

        current_bike_t = current_bike_t + correct_departure(bike.departure[i], i, il, time_final);
        current_bike_t = next_trip_time(current_bike_t, time_final, bike.departure[i], il);

        bike.paths[i].push_back(best_path);
        bike.access[i].push_back(access);
        bike.passage[i].push_back(passage);
        bike.scores[i] = bike.scores[i] + dist_final;
        bike.times[i] = bike.times[i] + time_final;
    }

    // A mode where every trip is walked is not a real option (thesis "outline").
    bool walks_all = true;
    for (const auto& trip : bike.paths[i])
        if (trip.size() > 0.5) { walks_all = false; break; }
    if (walks_all) {
        MIMOS_LOG(log, WARNING, "Agent %d walks when trying to plan bike trips so score bike as -inf as it's not an option", i);
        bike.scores[i] = -1e6;
    }
}

// ------------------------------------------------------------------------------------------
// Public and drive planning, per iteration. A mode is re-planned only in
// iteration 0 or if the agent simulated it last iteration (new information in memory).
// ------------------------------------------------------------------------------------------
void Model::plan(int it) {
    MIMOS_LOG(log, INFO, "----------------------------------- PLANNING -----------------------------------");
    progress_agents_.store(0, std::memory_order_relaxed);
    for_each_agent(N, cfg.threads, ws_, net.largest(), [&](int i, RoutingWorkspace& w) {
        progress_agents_.fetch_add(1, std::memory_order_relaxed);
        double current_public_t = pub.departure[i][0];
        double current_drive_t = drive.departure[i][0];
        const bool replan_public = it == 0 || exec_public[i] == true;
        const bool replan_drive = it == 0 || exec_drive[i] == true;
        for (int il = 0; il < pop.n_trips(i); ++il) {
            if (replan_public) plan_public_trip(i, il, current_public_t, w);
            if (replan_drive) plan_drive_trip(i, il, current_drive_t, w);
        }
        // Fixed daily costs (MATSim dailyMonetaryConstant), only for plans that use the mode
        // (any trip with a network route). Added once, right after the scores were rebuilt.
        auto uses_network = [](const std::vector<std::vector<int>>& trips) {
            for (const auto& t : trips) if (!t.empty()) return true;
            return false;
        };
        if (replan_public && cfg.daily_cost_public != 0.0 && uses_network(pub.paths[i]))
            pub.scores[i] += cfg.beta_m * cfg.daily_cost_public;
        if (replan_drive && cfg.daily_cost_drive != 0.0 && uses_network(drive.paths[i]))
            drive.scores[i] += cfg.beta_m * cfg.daily_cost_drive;
        bool walks_all = true;
        for (const auto& trip : drive.paths[i])
            if (trip.size() > 0.5) { walks_all = false; break; }
        if (walks_all) {
            MIMOS_LOG(log, WARNING, "Agent %d walks when trying to plan drive trips so score drive as -inf as it's not an option", i);
            drive.scores[i] = -1e6;
        }
    });
}

// Public transport routing for trip il. Differences from bike/drive:
//  * time-dependent: boarding a new itinerary waits for its next scheduled vehicle;
//  * the initial cutoff is walking straight to the destination;
//  * destination candidates are still expanded (no "not_target"), because a later stop of
//    the same itinerary may be better.
// Next-departure lookups stay linear: 21 schedule rows (circular lines) are not sorted.
void Model::plan_public_trip(int i, int il, double& current_public_t, RoutingWorkspace& w) {
    const auto& P = net.pub;

    if (il == 0) {
        pub.scores[i] = 0.0;
        pub.times[i] = 0.0;
    }

    Fringe fringe;
    // Access candidates enter the fringe through a "dirty fringe" that adds the waiting time
    // for the next vehicle at the stop (+ remembered delays) to their walking utility.
    std::priority_queue<Access> dirty_fringe;
    for (const auto& t : pop.public_src[i][il]) dirty_fringe.push(t);
    const auto& tgt = pop.public_tgt[i][il];
    const auto& mem = long_public_mem[i][il];

    while (!dirty_fringe.empty()) {
        int u = std::get<2>(dirty_fringe.top());
        double sc = std::get<0>(dirty_fringe.top());
        double tim = std::get<1>(dirty_fringe.top());
        dirty_fringe.pop();

        double delay_mem = 0.0;
        auto remembered = mem.find(u);
        if (remembered != mem.end()) delay_mem += remembered->second;

        double wait_time = 100000000.0;  // "no vehicle left today"
        if (P.itin[u] < -0.5) {
            wait_time = 0;  // rental bike
        } else {
            for (double sch : P.sch[u]) {
                if (sch >= current_public_t + tim + delay_mem) {
                    wait_time = sch - current_public_t - tim - delay_mem;
                    break;
                }
            }
        }
        sc += cfg.beta_trav_walk * (wait_time + delay_mem);
        tim += wait_time + delay_mem;

        seed(fringe, w, u, sc);
        w.set(u, sc, tim, w.pred[u]);
    }

    // Walking straight to the destination is the alternative to beat.
    const double walk_km = geo_km(pop.lat[i][il], pop.lon[i][il], pop.lat[i][il + 1], pop.lon[i][il + 1]);
    double cutoff = cfg.C_walk + (cfg.beta_trav_walk / 4.0 + cfg.beta_d_walk) * walk_km;
    double dist_final = cfg.C_walk + (cfg.beta_trav_walk / 4.0 + cfg.beta_d_walk) * walk_km;
    int pred_final = -1;
    double time_final = walk_km / 4.0;

    mark_targets(w, tgt);
    for (int u = pop_valid(fringe, w); u >= 0; u = pop_valid(fringe, w)) {
        if (w.dist[u] < cutoff) continue;

        const auto& time_row = P.times[u];
        int iv = -1;
        for (int v : P.adj[u]) {
            ++iv;
            if (v < 0) break;

            double temp_time = time_row[iv];
            double time_alt = w.time[u] + temp_time;
            double time_insim = time_alt + current_public_t;
            double beta_trav_temp, beta_m_temp, beta_transf_temp, beta_d_temp;

            if (P.itin[u] == P.itin[v]) {  // same itinerary: in-vehicle (or on rental bike)
                beta_trav_temp = cfg.beta_trav_public * temp_time;
                beta_m_temp = 0.0;
                beta_transf_temp = 0.0;
                if (P.itin[u] < -0.5)
                    beta_d_temp = (cfg.beta_d_rent_bike + cfg.beta_m * cfg.gamma_d_rent_bike) * P.lengths[u][iv];
                else
                    beta_d_temp = (cfg.beta_d_public + cfg.beta_m * cfg.gamma_d_public) * P.lengths[u][iv];
                // The last vehicle of a circular line stops when it completes its round.
                if (P.stop[u] == 0 && P.circ[u] == true && current_public_t + w.time[u] > P.sch[u].back())
                    beta_trav_temp = -100000000.0;
            } else {  // transfer: walk + wait for the next vehicle + fare + transfer constant
                double wait_time = 100000000.0;
                double delay_mem = 0.0;
                auto remembered = mem.find(v);
                if (remembered != mem.end()) delay_mem = remembered->second;
                time_insim = time_insim + delay_mem;

                if (P.mode[v] > -0.5) {
                    for (double sch : P.sch[v]) {
                        if (sch >= time_insim) {
                            wait_time = sch - time_insim;
                            break;
                        }
                    }
                    beta_trav_temp = cfg.beta_trav_walk * (temp_time + wait_time + delay_mem);
                    beta_m_temp = cfg.beta_m * cfg.ticket;
                    beta_transf_temp = cfg.C_transfer;  // utility of a line switch
                    time_alt = time_alt + wait_time + delay_mem;
                } else {  // rental bike: no waiting
                    beta_trav_temp = cfg.beta_trav_rent_bike * (temp_time);
                    beta_m_temp = cfg.beta_m * cfg.bike_ticket;
                    beta_transf_temp = cfg.C_rent_bike;  // utility of picking up a rental bike
                }
                beta_d_temp = cfg.beta_d_walk * P.lengths[u][iv];
            }

            double alt = w.dist[u] + beta_trav_temp + beta_m_temp + beta_transf_temp + beta_d_temp;
            if (alt > w.dist[v]) {
                w.set(v, alt, time_alt, u);
                int s = w.target_slot[v];
                if (s >= 0) {
                    const auto& p = tgt[s];
                    if (w.dist[v] + std::get<0>(p) > dist_final) {
                        dist_final = w.dist[v] + std::get<0>(p);
                        pred_final = v;
                        time_final = w.time[v] + std::get<1>(p);
                        cutoff = dist_final;
                    }
                }
                push_replacing(fringe, w, v, alt);
            }
        }
    }
    unmark_targets(w, tgt);

    std::vector<int> best_path;
    std::vector<double> passage;
    reconstruct_path(pred_final, current_public_t, w, best_path, passage, log, i);
    std::vector<int> access = access_vector(best_path, P.adj);
    w.reset();

    current_public_t = current_public_t + correct_departure(pub.departure[i], i, il, time_final);
    current_public_t = next_trip_time(current_public_t, time_final, pub.departure[i], il);

    pub.paths[i][il] = std::move(best_path);
    pub.access[i][il] = std::move(access);
    pub.passage[i][il] = std::move(passage);
    pub.scores[i] = pub.scores[i] + dist_final;
    pub.times[i] = pub.times[i] + time_final;
}

// Drive routing for trip il. Remembered delays on a link (long memory)
// add both disutility and time.
void Model::plan_drive_trip(int i, int il, double& current_drive_t, RoutingWorkspace& w) {
    const auto& D = net.drive;

    if (il == 0) {
        drive.scores[i] = 0.0;
        drive.times[i] = 0.0;
    }

    Fringe fringe;
    const auto& src = pop.drive_src[i][il];
    for (const auto& t : src) {
        int k = std::get<2>(t);
        w.set(k, std::get<0>(t), std::get<1>(t), -1);
        seed(fringe, w, k, w.dist[k]);
    }
    const auto& tgt = pop.drive_tgt[i][il];
    const auto& mem = long_drive_mem[i][il];

    double cutoff = -1000000.0;
    double dist_final = -1000000.0;
    int pred_final = -1;
    double time_final = 0.0;

    bool dijkstra = true;
    if (cfg.C_drive < cfg.C_limit) {
        dijkstra = false;
    } else if (same_node_in_both(src, tgt)) {
        double d = geo_km(pop.lat[i][il], pop.lon[i][il], pop.lat[i][il + 1], pop.lon[i][il + 1]);
        dist_final = cfg.C_walk + (cfg.beta_trav_walk / 4.0 + cfg.beta_d_walk) * d;
        time_final = geo_km(pop.lat[i][il], pop.lon[i][il], pop.lat[i][il + 1], pop.lon[i][il + 1]) / 4.0;
        dijkstra = false;
    }

    if (dijkstra) {
        mark_targets(w, tgt);
        const bool has_memory = !mem.empty();
        for (int u = pop_valid(fringe, w); u >= 0; u = pop_valid(fringe, w)) {
            if (w.dist[u] < cutoff) continue;

            const auto& time_row = D.times[u];
            const auto& length_row = D.lengths[u];
            int iv = -1;
            for (int v : D.adj[u]) {
                ++iv;
                if (v < 0) break;

                double alt = w.dist[u] + cfg.beta_trav_drive * time_row[iv] +
                             (cfg.beta_d_drive + cfg.beta_m * cfg.gamma_d_drive) * length_row[iv];
                float delay = 0.0f;
                bool delayed = false;
                if (has_memory) {
                    auto remembered = mem.find(std::make_pair(u, iv));
                    if (remembered != mem.end()) {
                        delay = remembered->second;
                        alt += cfg.beta_trav_delay * delay;
                        delayed = true;
                    }
                }

                if (alt > w.dist[v]) {
                    double t = w.time[u] + time_row[iv];
                    if (delayed) t += delay;
                    w.set(v, alt, t, u);
                    int s = w.target_slot[v];
                    if (s >= 0) {  // destination candidate: not expanded further
                        const auto& p = tgt[s];
                        if (w.dist[v] + std::get<0>(p) > dist_final) {
                            dist_final = w.dist[v] + std::get<0>(p);
                            pred_final = v;
                            time_final = w.time[v] + std::get<1>(p);
                            cutoff = dist_final;
                        }
                    } else {
                        push_replacing(fringe, w, v, alt);
                    }
                }
            }
        }
        unmark_targets(w, tgt);
    }

    std::vector<int> best_path;
    std::vector<double> passage;
    reconstruct_path(pred_final, current_drive_t, w, best_path, passage, log, i);
    std::vector<int> access = access_vector(best_path, D.adj);
    w.reset();

    current_drive_t = current_drive_t + correct_departure(drive.departure[i], i, il, time_final);
    current_drive_t = next_trip_time(current_drive_t, time_final, drive.departure[i], il);

    drive.paths[i][il] = std::move(best_path);
    drive.access[i][il] = std::move(access);
    drive.passage[i][il] = std::move(passage);
    drive.scores[i] = drive.scores[i] + dist_final;
    drive.times[i] = drive.times[i] + time_final;
}

}  // namespace mimos
