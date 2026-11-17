// Stand-alone command line runner of the MiMoS core (no Python), for profiling:
//     mimos_cli <data_dir/> <run_dir/> [n_agents=20000] [log_level=2 (INFO)] [it_number=2]
// Every other parameter takes its default value (mimos/config.hpp = the thesis values).
// Normal use goes through Python: `uv run mimos run <config.toml>`.
#include <exception>
#include <iostream>
#include <string>

#include "mimos/model.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <data_dir/> <run_dir/> [n_agents] [log_level 0..5] [it_number]\n";
        return 1;
    }
    mimos::Config cfg;
    cfg.data_dir = argv[1];
    cfg.run_dir = argv[2];
    if (argc > 3) cfg.n_agents = std::stoi(argv[3]);
    if (argc > 4) cfg.log_level = static_cast<mimos::LogLevel>(std::stoi(argv[4]));
    if (argc > 5) cfg.it_number = std::stoi(argv[5]);

    try {
        mimos::Model model(cfg);
        model.run();
    } catch (const std::exception& e) {
        std::cerr << "MiMoS error: " << e.what() << "\n";
        return 2;
    }
    return 0;
}
