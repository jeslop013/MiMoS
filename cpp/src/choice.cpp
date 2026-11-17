// Mode choice (thesis section "Deciding preferences").
//
//  * Iteration 0 (or cfg.mnl_choice): multinomial logit P(m) = exp(mu U_m) / sum_k exp(mu U_k)
//    over the whole-day utilities of bike, public and drive, then ownership constraints.
//  * Later iterations: Metropolis-style MCMC. Propose a different available mode uniformly
//    and accept with min(1, exp(mu (U_new - U_old))), which converges to the MNL distribution.
//
// Numerical details (legacy_mode reproduces the model version of the thesis instead):
//  1. Log-sum-exp: whole-day utilities are of order -10 ... -1e6, so exp(mu U) computed
//     directly can underflow to 0 for every mode; the maximum exponent is subtracted first.
//  2. With cfg.mnl_choice = true, exec_* are reset every iteration, so an agent has exactly
//     one executed mode.
//  3. legacy_mode draws one extra random number per MCMC decision, as the thesis version did.
#include <algorithm>
#include <array>
#include <cmath>

#include "mimos/model.hpp"

namespace mimos {

namespace {
// exp(mu * s_k) normalised over the given scores; with `stable`, the maximum exponent is
// subtracted first (mathematically identical, numerically safe).
template <size_t K>
std::array<double, K> logit(const std::array<double, K>& scores, double mu, bool stable) {
    std::array<double, K> p{};
    double shift = 0.0;
    if (stable) {
        shift = mu * scores[0];
        for (double s : scores) shift = std::max(shift, mu * s);
    }
    double total = 0.0;
    for (size_t k = 0; k < K; ++k) {
        p[k] = stable ? std::exp(mu * scores[k] - shift) : std::exp(mu * scores[k]);
        total += p[k];
    }
    for (auto& x : p) x = x / total;
    return p;
}
}  // namespace

void Model::choose(int it) {
    MIMOS_LOG(log, INFO, "-------------------------------- PREFERENCES IT: %d -----------------------------------", it);
    const bool legacy = cfg.legacy_mode;
    const double mu = cfg.beta_choice;

    for (int i = 0; i < N; ++i) {
        // MNL over public, drive, bike (in this order: it fixes how the random number is used).
        auto p3 = logit<3>({pub.scores[i], drive.scores[i], bike.scores[i]}, mu, !legacy);
        double prob_public = p3[0], prob_drive = p3[1];
        double prob = uniform(mt);

        if (it == 0 || cfg.mnl_choice == true) {
            if (!legacy) {  // detail 2: a fresh draw replaces last iteration's executed mode
                exec_public[i] = false;
                exec_drive[i] = false;
                exec_bike[i] = false;
            }
            // ---- MNL over all three modes: the preferred mode ---------------------------
            if (prob < prob_public) pub.preferred[i] = true;
            else if (prob < prob_public + prob_drive) drive.preferred[i] = true;
            else bike.preferred[i] = true;

            // ---- Ownership constraints decide what is actually simulated ----------------
            if (pop.owns_car[i] && pop.owns_bike[i]) {
                if (pub.preferred[i]) exec_public[i] = true;
                else if (drive.preferred[i]) exec_drive[i] = true;
                else exec_bike[i] = true;
            } else if (pop.owns_car[i] && !pop.owns_bike[i]) {  // public vs drive
                auto p2 = logit<2>({pub.scores[i], drive.scores[i]}, mu, !legacy);
                prob = uniform(mt);
                if (prob < p2[0]) exec_public[i] = true;
                else exec_drive[i] = true;
            } else if (!pop.owns_car[i] && pop.owns_bike[i]) {  // public vs bike
                auto p2 = logit<2>({pub.scores[i], bike.scores[i]}, mu, !legacy);
                prob = uniform(mt);
                if (prob < p2[0]) exec_public[i] = true;
                else exec_bike[i] = true;
            } else {
                exec_public[i] = true;  // no private vehicle: public transport (incl. walking)
            }
        } else {
            // ---- MCMC switch from last iteration's simulated mode ----------------------
            // Mode codes: 0 public, 1 drive, 2 bike.
            double prev_plan_score = 0.0;
            std::vector<unsigned int> alternatives;
            if (exec_public[i]) {
                prev_plan_score = pub.scores[i];
                if (pop.owns_car[i] && pop.owns_bike[i]) { alternatives.push_back(1); alternatives.push_back(2); }
                else if (pop.owns_car[i]) alternatives.push_back(1);
                else if (pop.owns_bike[i]) alternatives.push_back(2);
                else alternatives.push_back(0);  // nothing else: "propose" public again
            } else if (exec_drive[i]) {
                prev_plan_score = drive.scores[i];
                alternatives.push_back(0);
                if (pop.owns_bike[i]) alternatives.push_back(2);
            } else {
                prev_plan_score = bike.scores[i];
                alternatives.push_back(0);
                if (pop.owns_car[i]) alternatives.push_back(1);
            }

            int alt_index = static_cast<int>(alternatives.size() * uniform(mt));
            int proposed_mode = alternatives[alt_index];
            double proposed_score = proposed_mode == 0 ? pub.scores[i]
                                  : proposed_mode == 1 ? drive.scores[i]
                                                       : bike.scores[i];

            // exp of a difference: no underflow problem here.
            double accept_prob = std::exp(mu * (proposed_score - prev_plan_score));
            if (accept_prob > 1.0) accept_prob = 1.0;  // min(1, exp(mu dU))

            double rand = uniform(mt);
            if (legacy) (void)uniform(mt);  // detail 3: the thesis version's extra draw (see header)

            if (rand < accept_prob) {
                exec_public[i] = proposed_mode == 0;
                exec_drive[i] = proposed_mode == 1;
                exec_bike[i] = proposed_mode == 2;
            }
        }
    }
}

}  // namespace mimos
