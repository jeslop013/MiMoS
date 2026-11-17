// Population loader.
//
// Agent selection: every file has one line per agent of the full population, in the same
// order. `select_rows` chooses which lines are simulated:
//   * first n_agents lines (cfg.sample_random = false);
//   * a uniform random sample without replacement (cfg.sample_random = true, sample_seed).
// All files are then read in one pass, keeping only the chosen lines (others are skipped
// without parsing). Population::source_row records the original line of each agent.
#include "mimos/population.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>

#include "mimos/csv.hpp"

namespace mimos {

namespace {

// Skips the rest of the current line.
void skip_line(std::istream& in) { in.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); }

int count_lines(const std::string& file) {
    auto in = csv::open(file);
    int n = 0;
    std::string line;
    while (csv::next_line(in, line))
        if (!line.empty()) ++n;
    return n;
}

// Uniform integer in [0, n) from raw mt19937 output, by rejection. Written out because
// std::uniform_int_distribution differs between standard libraries (MSVC vs libc++), and the
// sample must be the same on every platform for a given seed.
uint32_t bounded(std::mt19937& g, uint32_t n) {
    const uint64_t range = uint64_t(1) << 32;
    const uint64_t limit = range - (range % n);
    uint64_t x;
    do x = g(); while (x >= limit);
    return static_cast<uint32_t>(x % n);
}

// Sorted line indices of the agents to simulate.
std::vector<int> select_rows(const Config& c, int population) {
    if (c.n_agents > population)
        throw std::runtime_error("MiMoS: n_agents (" + std::to_string(c.n_agents) +
                                 ") is larger than the population (" + std::to_string(population) + " agents)");
    std::vector<int> rows;
    if (!c.sample_random) {
        for (int r = 0; r < c.n_agents; ++r) rows.push_back(r);
        return rows;
    }
    // Floyd's algorithm: a uniform sample of n_agents out of `population`, no replacement.
    std::mt19937 g(c.sample_seed);
    std::vector<char> chosen(population, 0);
    for (int j = population - c.n_agents; j < population; ++j) {
        int t = static_cast<int>(bounded(g, static_cast<uint32_t>(j + 1)));
        chosen[chosen[t] ? j : t] = 1;
    }
    for (int r = 0; r < population; ++r)
        if (chosen[r]) rows.push_back(r);
    return rows;
}

// Calls `use(line)` for every selected line of `file`, in order.
void for_selected_lines(const std::string& file, const std::vector<int>& rows,
                        const std::function<void(const std::string&)>& use) {
    auto in = csv::open(file);
    std::string line;
    int r = 0;
    for (int target : rows) {
        for (; r < target; ++r) skip_line(in);
        if (!csv::next_line(in, line)) break;
        ++r;
        use(line);
    }
}

template <typename T, typename Parse>
std::vector<std::vector<T>> read_rows(const std::string& file, const std::vector<int>& rows, Parse parse) {
    std::vector<std::vector<T>> out;
    for_selected_lines(file, rows, [&](const std::string& line) {
        if (!line.empty()) out.push_back(csv::parse_row<T>(line, parse));
    });
    return out;
}

// Reads the paired "<mode>_nodes_<side>.csv" / "<mode>_times_<side>.csv" files. Each line holds,
// for one agent, the candidate nodes of all its trips separated by -1, and the access time
// (hours) to each candidate. `score_of(time, node)` gives the utility of reaching the node.
AccessTable read_access(const std::string& nodes_file, const std::string& times_file, const std::vector<int>& rows,
                        const std::function<double(double, int)>& score_of) {
    // Both files are read in lockstep (for a city the public ones take GBs, so lines are not buffered).
    auto nodes_in = csv::open(nodes_file);
    auto times_in = csv::open(times_file);
    std::string node_line, time_line;
    int r = 0;
    AccessTable table;
    for (int target : rows) {
        for (; r < target; ++r) {
            skip_line(nodes_in);
            skip_line(times_in);
        }
        bool got_nodes = csv::next_line(nodes_in, node_line);
        bool got_times = csv::next_line(times_in, time_line);
        ++r;
        if (!got_nodes || !got_times) break;

        std::vector<std::vector<Access>> agent;
        if (!node_line.empty() && !time_line.empty()) {
            auto nodes = csv::parse_row<int>(node_line, [](const std::string& s) { return std::stoi(s); });
            auto times = csv::parse_row<double>(time_line, [](const std::string& s) { return std::stod(s); });
            std::vector<Access> trip;
            for (size_t j = 0; j < nodes.size(); ++j) {
                int node = nodes[j];
                double time = times[j];
                if (node > -0.5) {
                    trip.emplace_back(score_of(time, node), time, node);
                } else {  // -1 separates consecutive trips
                    agent.push_back(trip);
                    trip.clear();
                }
            }
            agent.push_back(trip);
        }
        table.push_back(agent);
    }
    return table;
}

}  // namespace

Population load_population(const Config& c, const PublicNetwork& pub, Logger& log) {
    Population p;
    const std::string dir = c.data_dir + "pop_files/";
    auto to_double = [](const std::string& s) { return std::stod(s); };

    MIMOS_LOG(log, INFO, "----- LOADING AGENDA OF AGENTS -----");
    p.population_size = count_lines(dir + "lat.csv");
    p.source_row = select_rows(c, p.population_size);
    const auto& rows = p.source_row;

    p.lat = read_rows<double>(dir + "lat.csv", rows, to_double);
    p.lon = read_rows<double>(dir + "lon.csv", rows, to_double);
    p.start = read_rows<double>(dir + "t_start.csv", rows, to_double);
    p.end = read_rows<double>(dir + "t_end.csv", rows, to_double);
    p.prio = read_rows<int>(dir + "prio.csv", rows, [](const std::string& s) { return std::stoi(s); });

    // ---- Access / egress candidates -------------------------------------------------------
    // Off-network movement: walking at 4 km/h (public, bike) or driving at 15 km/h (drive).
    // The arithmetic order of each expression is fixed (results depend on the rounding).
    const std::string pre = dir + "distances_pre_computation/";
    MIMOS_LOG(log, INFO, "----------------- LOAD PRE-COMPUTE DISTANCES TO NODES OF NETWORKS -------------------");

    p.public_src = read_access(pre + "public_nodes_source.csv", pre + "public_times_source.csv", rows,
                               [&](double time, int node) {
                                   double length = 4.0 * time;
                                   double score = c.beta_trav_walk * time + c.beta_d_walk * length;
                                   // Boarding constant: public transport or rental bike
                                   if (pub.itin[node] > -0.5) score += c.C_public + c.beta_m * c.ticket;
                                   else score += c.C_rent_bike + c.beta_m * c.bike_ticket;
                                   return score;
                               });
    p.public_tgt = read_access(pre + "public_nodes_target.csv", pre + "public_times_target.csv", rows,
                               [&](double time, int) {
                                   double length = 4.0 * time;
                                   return c.beta_trav_walk * time + c.beta_d_walk * length;
                               });
    p.bike_src = read_access(pre + "bike_nodes_source.csv", pre + "bike_times_source.csv", rows,
                             [&](double time, int) {
                                 double length = 4.0 * time;
                                 return c.C_bike + c.beta_trav_walk * time + c.beta_d_walk * length;
                             });
    p.bike_tgt = read_access(pre + "bike_nodes_target.csv", pre + "bike_times_target.csv", rows,
                             [&](double time, int) {
                                 double length = 4.0 * time;
                                 return c.beta_trav_walk * time + c.beta_d_walk * length;
                             });
    p.drive_src = read_access(pre + "drive_nodes_source.csv", pre + "drive_times_source.csv", rows,
                              [&](double time, int) {
                                  double length = 15.0 * time;
                                  return c.C_drive + c.beta_trav_drive * time + c.beta_d_drive * length +
                                         c.beta_m * c.gamma_d_drive * length;
                              });
    p.drive_tgt = read_access(pre + "drive_nodes_target.csv", pre + "drive_times_target.csv", rows,
                              [&](double time, int) {
                                  double length = 15.0 * time;
                                  return c.beta_trav_drive * time + c.beta_d_drive * length +
                                         c.beta_m * c.gamma_d_drive * length;
                              });

    if (p.size() != c.n_agents || static_cast<int>(p.drive_tgt.size()) != c.n_agents)
        throw std::runtime_error("MiMoS: population files have fewer rows than expected");
    MIMOS_LOG(log, INFO, "Population loaded: %d of %d agents (%s)", p.size(), p.population_size,
              c.sample_random ? "random sample" : "first rows");
    return p;
}

}  // namespace mimos
