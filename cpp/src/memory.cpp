// Memory and learning (thesis section "Memory and learning").
//
// Short memory: delays met in this iteration, per agent, trip and network element (drive:
// link (u, av); public: stop). Elements traversed WITHOUT delay are added with 0, so good
// experiences are learned too.
// Long memory: kept across iterations. Experienced elements are updated with an exponential
// moving average (learning rate alpha_l), unused ones decay (forgetting rate alpha_f). Rates
// move linearly with the iteration k ("simulated annealing").
//
// Rate schedule (thesis eqs. alpha_l(k), alpha_f(k)):
//     alpha(k) = alpha_i                                          for k <  k_min
//              = alpha_i + (k - k_min)/(k_max - k_min) (alpha_f - alpha_i)  for k_min <= k <= k_max
//              = alpha_f                                          for k >  k_max
// legacy_mode uses the schedule of the thesis version: alpha_i + k (alpha_f - alpha_i) / k_max,
// clamped at alpha_f, without k_min.
// An element seen for the first time is stored as alpha_l * delay (an EMA starting from 0).
#include "mimos/model.hpp"

namespace mimos {

namespace {

// Learning then forgetting for one agent and trip. Key is int (public stop) or (u, av).
// Rate at iteration k (see the schedule above).
double annealed(double a_i, double a_f, int k, const Config& c) {
    if (k < c.k_min) return a_i;
    if (k >= c.k_max) return a_f;
    return a_i + static_cast<double>(k - c.k_min) / (c.k_max - c.k_min) * (a_f - a_i);
}

template <typename Key>
void update_long(std::map<Key, float>& shrt, std::map<Key, float>& lng, const Config& c, int it) {
    const int k = it;
    if (shrt.size() > 0.5) {
        for (const auto& kv : shrt) {
            double alpha_l_eff;
            if (c.legacy_mode) {
                alpha_l_eff = c.alpha_l_i + k * (c.alpha_l_f - c.alpha_l_i) / c.k_max;
                if (alpha_l_eff > c.alpha_l_f) alpha_l_eff = c.alpha_l_f;
            } else {
                alpha_l_eff = annealed(c.alpha_l_i, c.alpha_l_f, k, c);
            }
            if (lng.find(kv.first) != lng.end())
                lng[kv.first] = (1 - alpha_l_eff) * lng[kv.first] + alpha_l_eff * shrt[kv.first];
            else
                lng[kv.first] = alpha_l_eff * shrt[kv.first];
        }
    }
    if (lng.size() > 0.5) {
        for (auto& kv : lng) {
            if (shrt.find(kv.first) == shrt.end()) {  // not used today: forget a bit
                double alpha_f_eff;
                if (c.legacy_mode) {
                    alpha_f_eff = c.alpha_f_i + k * (c.alpha_f_f - c.alpha_f_i) / c.k_max;
                    if (alpha_f_eff < c.alpha_f_f) alpha_f_eff = c.alpha_f_f;
                } else {
                    alpha_f_eff = annealed(c.alpha_f_i, c.alpha_f_f, k, c);
                }
                kv.second = (1.0 - alpha_f_eff) * kv.second;
            }
        }
    }
}

}  // namespace

void Model::update_memory(int it) {
    MIMOS_LOG(log, INFO, "------------------------- UPDATING LONG MEMORY AT ITERATION: %d ------------------------", it);
    const auto& D = net.drive;
    const auto& P = net.pub;

    for (int i = 0; i < N; ++i) {
        if (exec_drive[i] == true) {
            const auto& nodes = traj_drive.nodes[i];
            for (int il = 0; il < pop.n_trips(i); ++il) {
                // Links traversed in trip il without a recorded delay are learned as 0.
                // The trajectory is one sequence for the whole day; -il-1 marks the start of
                // trip il.
                bool record = false;
                for (size_t j = 0; j + 1 < nodes.size(); ++j) {
                    int u = nodes[j], v = nodes[j + 1];
                    if (u < -0.5) {
                        record = (u == -il - 1);
                    } else if (record == true && v > -0.5) {
                        for (int av = 0; av < static_cast<int>(D.adj[u].size()); ++av) {
                            if (D.adj[u][av] == v) {
                                auto key = std::make_pair(u, av);
                                if (short_drive_mem[i][il].find(key) == short_drive_mem[i][il].end())
                                    short_drive_mem[i][il][key] = 0.0;
                                break;
                            }
                        }
                    }
                }
                update_long(short_drive_mem[i][il], long_drive_mem[i][il], cfg, it);
            }
        } else if (exec_public[i] == true) {
            const auto& nodes = traj_public.nodes[i];
            for (int il = 0; il < pop.n_trips(i); ++il) {
                // Stops where the agent boards a new itinerary without a recorded delay -> 0.
                bool record = false;
                for (size_t j = 0; j + 1 < nodes.size(); ++j) {
                    int u = nodes[j], v = nodes[j + 1];
                    int itin_u = u > -0.5 ? P.itin[u] : -999;
                    int itin_v = v > -0.5 ? P.itin[v] : -999;
                    if (u < -0.5) record = (u == -il - 1);
                    if (record == true && v > -0.5 && itin_u != itin_v)
                        if (short_public_mem[i][il].find(v) == short_public_mem[i][il].end())
                            short_public_mem[i][il][v] = 0.0;
                }
                update_long(short_public_mem[i][il], long_public_mem[i][il], cfg, it);
            }
        }
        // Bike: no interaction, no memory.
    }
}

}  // namespace mimos
