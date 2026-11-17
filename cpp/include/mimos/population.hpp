// Synthetic population and agenda (thesis section "Model input: Synthetic population &
// agenda"), plus the precomputed access/egress candidates (thesis "Pre-computing agent
// loading and unloading from networks").
//
// Agent i has lat[i].size() agenda locations and lat[i].size()-1 trips; trip il goes from
// location il to location il+1. Times are hours of the agenda day (which starts at 4:00).
#pragma once

#include <string>
#include <vector>

#include "mimos/config.hpp"
#include "mimos/logger.hpp"
#include "mimos/network.hpp"
#include "mimos/types.hpp"

namespace mimos {

struct Population {
    std::vector<std::vector<double>> lat, lon;   // agenda locations
    std::vector<std::vector<double>> start, end; // suggested activity start / end times
    std::vector<std::vector<int>> prio;          // activity priority (0 = work, highest)

    std::vector<bool> owns_car, owns_bike;       // drawn in Model (needs the RNG)

    int population_size = 0;                     // agents in the input files
    std::vector<int> source_row;                 // input line of each simulated agent (written
                                                 // to agents.csv to link results to the population)

    // Access (source) and egress (target) candidates per agent and trip, with the score and
    // time of walking or driving between the agenda location and the network node.
    AccessTable public_src, public_tgt, bike_src, bike_tgt, drive_src, drive_tgt;

    int size() const { return static_cast<int>(lat.size()); }
    int n_trips(int i) const { return static_cast<int>(lat[i].size()) - 1; }
};

// Loads cfg.n_agents agents (first rows or a random sample, see Config::sample_random) from
// <data_dir>/pop_files/ and scores their access
// candidates with the utility parameters in cfg.
// Throws std::runtime_error if files are missing or contain fewer agents than requested.
Population load_population(const Config& cfg, const PublicNetwork& pub, Logger& log);

}  // namespace mimos
