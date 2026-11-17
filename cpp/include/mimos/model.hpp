// MiMoS model: state and the stages of the iterative cycle (thesis "MiMoS outline and
// iterative cycle", Fig. "Overview diagram of MiMoS").
//
// Each stage is a method implemented in its own file:
//
//   load()                  model.cpp        networks, agenda, ownership, access candidates
//   plan_bike()             routing.cpp      bike routes (once: no interaction)
//   start_iteration(it)     model.cpp        per-iteration resets
//   plan(it)                routing.cpp      public + drive routes
//   choose(it)              choice.cpp       MNL (it 0) / MCMC mode choice
//   prepare_execution(it)   model.cpp        seed both event queues
//   simulate_public(it)     public_sim.cpp   public transport simulation
//   simulate_drive(it)      drive_sim.cpp    private vehicle simulation + gridlocks
//   update_memory(it)       memory.cpp       long memory update (learning/forgetting)
//   check_invariants(it)    model.cpp        conservation checks
//   write_iteration(it)     io.cpp           per-iteration outputs
//   write_final()           io.cpp           final outputs
//
// run() executes the whole cycle; the Python package calls the stages one by one.
#pragma once

#include <atomic>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "mimos/config.hpp"
#include "mimos/logger.hpp"
#include "mimos/network.hpp"
#include "mimos/population.hpp"
#include "mimos/types.hpp"

namespace mimos {

// Routes planned for one mode, for every agent and trip.
struct ModePlan {
    std::vector<bool> preferred;      // chosen by the MNL this iteration
    std::vector<double> scores;       // whole-day utility
    std::vector<double> times;        // whole-day travel time (hours)
    std::vector<std::vector<std::vector<int>>> paths;     // [agent][trip] node sequence
    std::vector<std::vector<std::vector<int>>> access;    // [agent][trip] av of each link
    std::vector<std::vector<std::vector<double>>> passage;// [agent][trip] planned node times
    std::vector<std::vector<double>> departure;           // [agent][trip] departure time
};

// Executed (simulated) trajectories of one mode: node ids, with agenda locations encoded as
// -il-1, and the time of each record (thesis "Saving the results").
struct Trajectories {
    std::vector<std::vector<int>> nodes;
    std::vector<std::vector<double>> times;

    void reset(int n) {
        nodes.assign(n, {});
        times.assign(n, {});
    }
    void add(int i, int node, double t) {
        nodes[i].push_back(node);
        times[i].push_back(t);
    }
};

using PublicMemory = std::vector<std::vector<std::map<int, float>>>;                // [agent][trip][stop]
using DriveMemory = std::vector<std::vector<std::map<std::pair<int, int>, float>>>; // [agent][trip][(u,av)]

// Summary of one drive simulation, for tests and diagnostics.
struct DriveStats {
    long long events = 0;
    int agents = 0;
    int finished = 0;
    int jammed_in_links = 0;     // cars left inside links at the end (gridlock)
    int jammed_in_requests = 0;  // cars left waiting to enter a link
    int stuck_moves = 0;         // cars that entered a link after stuck_time (Config::stuck_time)
};

struct PublicStats {
    long long events = 0;
    int agents = 0;
    int finished = 0;
};

class Model {
public:
    explicit Model(Config cfg);

    // Stages (see the table at the top of this file)
    void load();
    void plan_bike();
    void start_iteration(int it);
    void plan(int it);
    void choose(int it);
    void prepare_execution(int it);
    PublicStats simulate_public(int it);
    DriveStats simulate_drive(int it);
    void update_memory(int it);
    bool check_invariants(int it);
    void write_iteration(int it);
    void write_final();

    // Whole cycle: load, bike planning, it_number iterations, final outputs.
    void run();

    // Progress of the running stage, for a progress bar polled from another thread (the long
    // stages release the GIL; mimos.progress). Agents started in plan()/plan_bike(), and the
    // simulated clock (hours) of simulate_public()/simulate_drive(). No effect on results.
    int progress_agents() const { return progress_agents_.load(std::memory_order_relaxed); }
    double progress_clock() const { return progress_clock_.load(std::memory_order_relaxed); }

    // ---- Per-trip results of the last simulated iteration (analysis, calibration) ---------
    // Surveys count trips, but MiMoS chooses one mode per agent-day: a public day can mix bus
    // and walk-only trips, and a drive day can contain walked trips without a road route.
    // Mode codes follow the "basic mode" of mobility surveys: a trip that boards a bus or train is
    // public transport even if it also walks or rides a rental bike.
    enum TripMode { TRIP_WALK = 0, TRIP_CAR = 1, TRIP_PT = 2, TRIP_BIKE = 3 };
    std::vector<std::vector<int>> trip_modes() const;          // [agent][trip] TripMode
    // Door-to-door duration (hours) from the executed trajectories; NaN if the trip never
    // ended (gridlock).
    std::vector<std::vector<double>> trip_durations() const;   // [agent][trip]
    std::vector<std::vector<double>> trip_departures() const;  // [agent][trip] hours, NaN if none
    // Straight-line distance (km) between the two agenda locations of each trip. Used to keep
    // survey comparisons to daily urban mobility: an agenda can contain far destinations
    // that no city survey counts.
    std::vector<std::vector<double>> trip_distances_km() const; // [agent][trip]
    // Executed score of each agent's simulated mode (NaN for agents not simulated or
    // with an unfinished trip). MiMoS plans with PERCEIVED times (free flow + remembered
    // delays); only time depends on the simulation, so
    //     executed = planned + beta * (executed day time - planned day time)
    // with beta = beta_trav_delay (drive) or beta_trav_walk (public: waiting). Bike does not
    // interact, so executed = planned. The gap planned - executed is the relaxation measure
    // MATSim reports (thesis: MiMoS scores perceived plans, MATSim executed ones).
    std::vector<double> executed_scores() const;

    // Every road link traversed in the last drive simulation (report and emissions).
    // A car moves along a link at free-flow speed and waits at its END to enter the next one,
    // so t_end - t_enter = free-flow time and t_exit - t_end = waiting (delay) on that link.
    // Trips that never ended (gridlock) are left out.
    struct LinkTraversals {
        std::vector<int> agent, trip, u, av;
        std::vector<double> t_enter, t_end, t_exit;  // hours
    };
    LinkTraversals drive_link_traversals() const;

    // ---- State (public so the Python bindings and tests can read it) ---------------------
    Config cfg;
    Logger log;
    std::mt19937 mt;
    std::uniform_real_distribution<double> uniform{0.0, 1.0};

    Networks net;
    Population pop;
    int N = 0;

    ModePlan bike, pub, drive;

    // Mode actually simulated (preference can differ because of ownership constraints).
    std::vector<bool> exec_public, exec_drive, exec_bike;
    Trajectories traj_bike, traj_public, traj_drive;
    // Per agent: set to -1 when the agent enters a simulation, +1 when it ends its day.
    // So 0 = finished exactly once (or not simulated, e.g. bike), < 0 = never finished,
    // > 0 = finished more than once (duplicated events).
    std::vector<int> finished_exec;

    PublicMemory short_public_mem, long_public_mem;
    DriveMemory short_drive_mem, long_drive_mem;

    // Drive simulation state, kept after simulate_drive() for the invariant checks.
    std::vector<std::vector<int>> drive_link_space;   // free storage per link
    std::vector<std::vector<int>> drive_mouth_space;  // free gaps at each link entrance
    // Sample scaling actually applied (cfg.capacity_scale, or n_agents / population size when
    // cfg.capacity_scale <= 0). Set by load().
    double capacity_scale = 1.0;
    // Storage capacity actually simulated: network capacity scaled by capacity_scale
    // (min 1 car), and at least 1 when not in legacy_mode (see load()).
    std::vector<std::vector<int>> drive_capacity;
    // Public simulation state.
    std::map<int, int> space_map;                     // vehicle id -> free seats

    // Distance in km between two points: double-precision haversine, or the float-argument
    // version of the thesis in legacy_mode.
    double geo_km(double lat1, double lon1, double lat2, double lon2) const {
        if (cfg.legacy_mode)
            return haversine(static_cast<float>(lat1), static_cast<float>(lon1), static_cast<float>(lat2),
                             static_cast<float>(lon2));
        return haversine_d(lat1, lon1, lat2, lon2);
    }
    // Priority of drive link (u, av) as compared at intersections (truncated in legacy_mode).
    double link_priority(int u, int av) const {
        double p = net.drive.priority[u][av];
        return cfg.legacy_mode ? std::trunc(p) : p;
    }

    // Per-thread scratch arrays of the routing search, at the size of the largest network.
    // They are allocated once per thread, and only the entries a search touched are reset
    // afterwards.
    struct RoutingWorkspace {
        std::vector<double> dist, time;   // best utility / time found for each node
        std::vector<int> pred;            // predecessor on the best route
        std::vector<unsigned> version;    // lazy-deletion stamp of the node's fringe entry
        std::vector<int> target_slot;     // index in the trip's target list, or -1
        std::vector<int> touched;         // nodes to reset after the search

        void init(size_t n);
        void set(int v, double d, double t, int p);  // dist/time/pred of v (records the touch)
        void reset();                                // restores every touched entry
    };

private:
    std::vector<RoutingWorkspace> ws_;    // one per thread
    std::atomic<int> progress_agents_{0};
    std::atomic<double> progress_clock_{0.0};

    // Event queues, seeded by prepare_execution().
    PublicQueue public_queue_;
    DriveQueue drive_queue_;
    int in_public_simulation_ = 0;
    int in_drive_simulation_ = 0;

    // routing.cpp helpers
    double correct_departure(std::vector<double>& departure, int i, int il, double time_final);
    void plan_bike_agent(int i, RoutingWorkspace& w);
    void plan_public_trip(int i, int il, double& current_t, RoutingWorkspace& w);
    void plan_drive_trip(int i, int il, double& current_t, RoutingWorkspace& w);

    int public_capacity(int u) const;
    // Departure time and duration of every trip from the executed trajectories (model.cpp).
    void trip_times(std::vector<std::vector<double>>& departures, std::vector<std::vector<double>>& durations) const;
};

}  // namespace mimos
