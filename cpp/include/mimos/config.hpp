// MiMoS run configuration.
//
// Defaults are the parameter values of the thesis. Python fills this struct from a TOML
// file (src/mimos/config.py). Units: times in hours, distances in km, utilities
// dimensionless.
#pragma once

#include <string>

#include "mimos/logger.hpp"

namespace mimos {

struct Config {
    // ---- Run -------------------------------------------------------------------------
    std::string data_dir = "./";       // must contain net_files/ and pop_files/
    std::string run_dir = "./results/";
    int n_agents = 20000;              // number of agents simulated
    // false: the FIRST n_agents rows of the agenda. These are not necessarily a random sample
    // of the population (the row order of the input decides).
    // true: a uniform random sample of n_agents rows (without replacement, sample_seed).
    bool sample_random = false;
    unsigned int sample_seed = 0;
    int it_number = 2;                 // number of iterations
    int print_it = 1;                  // write per-iteration outputs every print_it iterations
    unsigned int seed = 0;             // std::mt19937 seed
    LogLevel log_level = INFO;
    int threads = 0;                   // planning threads (OpenMP); 0 = all cores. Results do not
                                       // depend on it: agents are planned independently.

    // ---- Thesis version ------------------------------------------------------------------
    // true: reproduce the model version used in the thesis (numerics and rules marked
    //       `if (cfg.legacy_mode)` in the code). false (default): the current version.
    bool legacy_mode = false;

    // ---- Capacity scaling for samples --------------------------------------------------------
    // Simulating a sample of the population on full capacities underestimates congestion.
    // capacity_scale scales road storage (min 1 car), road flow (the safety time becomes
    // t_s / capacity_scale) and public transport vehicle capacity (min 1), as MATSim does with
    // flowCapacityFactor / storageCapacityFactor = sample share.
    //   1.0  = no scaling;  <= 0 = automatic: n_agents / population size.
    double capacity_scale = 1.0;

    // ---- Utility function (thesis eq. "U_route") ---------------------------------------------
    double beta_m = -1.0;              // marginal utility of money
    double beta_transfer = -1.0;       // (not used)
    double C_limit = -100.0;           // modes whose constant is below this skip routing
    double C_public = -200.0;          // per public-transport trip (added when boarding the first vehicle)
    double C_transfer = -200.0;        // per transfer to another line (thesis: same as C_public;
                                       // MATSim calls it utilityOfLineSwitch, default -1)
    double C_rent_bike = -110.0;
    double C_walk = 0.0;
    double C_bike = -220.0;
    double C_drive = -13.0;

    double beta_trav_public = -6.0;
    double beta_d_public = -1e-3;
    double gamma_d_public = 0.0;

    double beta_trav_rent_bike = -6.0;
    double beta_d_rent_bike = 0.0;
    double gamma_d_rent_bike = 0.0;

    double beta_trav_walk = -6.0;
    double beta_d_walk = -1e-3;

    double beta_trav_bike = -6.0;
    double beta_d_bike = 0.0;
    double gamma_d_bike = 0.0;
    double beta_trav_no_cycleway = -6.0;

    double beta_trav_drive = -6.0;
    double beta_trav_delay = -6.0;     // extra disutility per hour of remembered delay
    double beta_d_drive = 0.0;
    double gamma_d_drive = 2e-1;

    double beta_dur = 0.0;             // (not used)
    double beta_late = 0.0;            // (not used)

    double jam_penalty = 24.0;         // hours of delay "shock" for gridlocked agents
    // Fixed money costs per day of using the mode (MATSim dailyMonetaryConstant, entered here
    // as a positive cost in money units; multiplied by beta_m). Thesis: 0. Added once per day
    // to the whole-day score of agents whose plan actually uses that mode.
    double daily_cost_drive = 0.0;
    double daily_cost_public = 0.0;
    double departure_correction = 0.0; // 0 disables agenda-priority departure corrections

    // ---- Public transport capacities -----------------------------------------------------------
    int city_bus_cap = 80;
    int reg_bus_cap = 105;
    int cercanias_cap = 900;
    double ticket = 0.0;
    double bike_ticket = 0.0;

    // ---- Drive interaction ------------------------------------------------------------------
    double t_s = 3.0 / 3600.0;         // safety time gap between cars (hours)
    // Speed (km/h) of the straight-line car leg between an agenda location and the road network
    // in the drive SIMULATION (thesis: 30). Planning uses the precomputed access times; if they
    // assume another speed (15 km/h in docs/input_format.md), set the same value here.
    double car_access_speed = 30.0;
    // Link-exit dynamics. Both off = the dynamics of the thesis; both require legacy_mode = false.
    // fifo_link_exit: cars leave a link in the order they reached its end. A car that cannot
    //   enter its next link blocks the cars behind it (MATSim QSim). Off: they can pass it.
    bool fifo_link_exit = false;
    // stuck_time (hours, 0 = off): a car that has waited this long to enter a link enters it
    //   even if the link is full or it should give way, but still needs a free gap (flow
    //   capacity). Resolves gridlocks inside the simulation (MATSim stuckTime; Berlin 30 s).
    double stuck_time = 0.0;

    // ---- Memory (thesis "Memory and learning") -----------------------------------------------
    double alpha_l_i = 0.01;
    double alpha_l_f = 0.01;
    double alpha_f_i = 0.0;
    double alpha_f_f = 0.0;
    int k_min = 0;                     // rates stay at their initial value before k_min (thesis);
                                       // ignored in legacy_mode
    int k_max = 1;

    // ---- Choice (thesis "Deciding preferences") ----------------------------------------------
    double beta_choice = 1.0;          // mu in the MNL
    bool mnl_choice = false;           // true: MNL every iteration; false: MCMC after it 0

    // ---- Car / bike ownership ----------------------------------------------------------------
    // Probability that an agent owns a car / a bike, drawn from discrete_distribution({1-p, p}).
    // Default p = 1: everybody owns both.
    double p_car = 1.0;
    double p_bike = 1.0;
};

}  // namespace mimos
