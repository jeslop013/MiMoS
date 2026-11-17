// Output writers (thesis "Saving the results").
//
// Formats (the CSV formats of the thesis version):
//   plan_<mode>_scores[_it_K].csv            one whole-day utility per agent and line
//   exec_<mode>_paths[_it_K].csv             one agent per line: visited nodes, where agenda
//                                            locations are -il-1; -999 if the agent did not use
//                                            this mode
//   exec_<mode>_passage_times[_it_K].csv     matching times (hours)
//   agents.csv                               line k = agent k of the outputs: its row in the
//                                            population files (needed with random sampling)
// Lines end with '\n' instead of std::endl (no flush per line); the bytes written are the same.
#include <fstream>

#include "mimos/model.hpp"

namespace mimos {

namespace {

void write_scores(const std::string& file, const std::vector<double>& scores, int n, Logger& log) {
    std::ofstream out(file);
    if (!out.is_open()) {
        MIMOS_LOG(log, ERROR, "File %s could not be opened", file.c_str());
        return;
    }
    for (int i = 0; i < n; ++i) out << scores[i] << '\n';
}

void write_trajectories(const std::string& nodes_file, const std::string& times_file, const Trajectories& t,
                        const std::vector<bool>& used, int n, Logger& log) {
    std::ofstream nodes(nodes_file), times(times_file);
    if (!nodes.is_open() || !times.is_open()) {
        MIMOS_LOG(log, ERROR, "File %s or %s could not be opened", nodes_file.c_str(), times_file.c_str());
        return;
    }
    for (int i = 0; i < n; ++i) {
        if (used[i]) {
            for (size_t j = 0; j < t.nodes[i].size(); ++j) {
                nodes << t.nodes[i][j];
                times << t.times[i][j];
                if (j + 1 < t.nodes[i].size()) {
                    nodes << ",";
                    times << ",";
                }
            }
            nodes << '\n';
            times << '\n';
        } else {
            nodes << -999 << '\n';
            times << -999 << '\n';
        }
    }
}

}  // namespace

void Model::write_iteration(int it) {
    const std::string r = cfg.run_dir;
    const std::string k = "_it_" + std::to_string(it) + ".csv";
    write_scores(r + "plan_public_scores" + k, pub.scores, N, log);
    write_scores(r + "plan_drive_scores" + k, drive.scores, N, log);
    write_trajectories(r + "exec_public_paths" + k, r + "exec_public_passage_times" + k, traj_public, exec_public, N, log);
    write_trajectories(r + "exec_drive_paths" + k, r + "exec_drive_passage_times" + k, traj_drive, exec_drive, N, log);
}

void Model::write_final() {
    const std::string r = cfg.run_dir;
    {
        std::ofstream agents(r + "agents.csv");
        agents << "agent,population_row\n";
        for (int i = 0; i < N; ++i) agents << i << ',' << pop.source_row[i] << '\n';
    }
    write_scores(r + "plan_bike_scores.csv", bike.scores, N, log);
    write_scores(r + "plan_public_scores.csv", pub.scores, N, log);
    write_scores(r + "plan_drive_scores.csv", drive.scores, N, log);
    write_trajectories(r + "exec_bike_paths.csv", r + "exec_bike_passage_times.csv", traj_bike, exec_bike, N, log);
    write_trajectories(r + "exec_public_paths.csv", r + "exec_public_passage_times.csv", traj_public, exec_public, N, log);
    write_trajectories(r + "exec_drive_paths.csv", r + "exec_drive_passage_times.csv", traj_drive, exec_drive, N, log);
}

}  // namespace mimos
