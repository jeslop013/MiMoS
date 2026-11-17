// Private transport micro-simulation (thesis "Private transport micro-simulation",
// Algorithm "Private transport mode simulation algorithm" and appendix sub-procedures).
//
// State per link (u, av)  [see types.hpp for the event encoding]
//   drive_link_space[u][av]   free storage; starts at the (sample-scaled) capacity. A car
//                             occupies it from entering the link until it enters the next one
//                             (or leaves the network).
//   drive_mouth_space[u][av]  free "gaps" at the entrance; starts at lanes. Entering takes one,
//                             and a gap-opening event gives it back t_s later (safety time).
//   link_q[u][av]             cars travelling the link in free flow, ordered by entry time.
//   exit_q[u][av]             (fifo_link_exit only) cars that reached the end of the link, in
//                             arrival order; only the first one may leave.
//   request_q[u][av]          cars waiting to enter the link (no space, no gap, or they gave way).
//
// Event handlers (thesis colour boxes):
//   agent, o == -1     "Exit from location"
//   agent, o == -0.5   "Entry to road network"
//   g == -1            "Gap opening at segment entry"
//   g == -0.5          retry after giving way (see rule 1 below), or stuck check
//   g ==  0            "De-saturation of segment"
//   g == +1            "Check arrival at segment end" (check-me)
//   g == +2            next car of the exit queue tries to leave (rule 5)
//
// Invariant at the end of a run without gridlock: link space == capacity and mouth space ==
// lanes for every link (checked in Model::check_invariants).
//
// Rules (rules 1-3 are off in legacy_mode, which reproduces the model version of the thesis):
//  1. RETRY. A car that gives way is parked in request_q. Giving way also schedules a RETRY
//     event t_s later; by then the conflicting cars of that window have been processed. A
//     retry only admits a car when a gap and space are free, so it never bypasses the safety
//     time. This guarantees that a car that gave way is woken up again.
//  2. CONFLICT DETECTION. An event counts as "entering u" only if its link (in, ia) ends at u.
//  3. PRIORITY VALUES are doubles: link roads keep their +0.5.
//  4. SAMPLE SCALING (Config::capacity_scale): scaled storage (min 1), and t_s / capacity_scale.
//
// Options (both off by default):
//  5. FIFO LINK EXIT (Config::fifo_link_exit). Off: a car that cannot enter its next link
//     moves out of link_q into the next link's request_q, so the cars behind it can still
//     leave the link to other links (overtaking at the link exit). On: a car that reaches the
//     end of the link joins exit_q; only the first car of exit_q tries to leave, and while it
//     waits it blocks the others (MATSim QSim). When it leaves, an EXIT event lets the next
//     car try. Its waiting time counts from its arrival at the end of the link.
//  6. STUCK TIME (Config::stuck_time). A car first in request_q that has waited stuck_time
//     enters the link without free storage and without giving way, but still takes a free gap,
//     so the flow capacity holds. Gaps always reopen, so no gridlock can last (MATSim
//     stuckTime). Link space can go negative meanwhile; it is back to capacity at the end.
#include <algorithm>
#include <cmath>
#include <deque>
#include <map>

#include "mimos/model.hpp"

namespace mimos {

namespace {

using LinkQueues = std::vector<std::vector<DriveQueue>>;
using ExitQueues = std::vector<std::vector<std::deque<DriveEvent>>>;

constexpr double GAP = -1.0, RETRY = -0.5, DESATURATION = 0.0, CHECK_ME = 1.0, EXIT = 2.0;

// Adds a delay (hours) to agent i's short memory for link (u, av).
// The two branches keep a fixed evaluation order: `m + event_t - t0` is
// (m + event_t) - t0, which rounds differently from m + (event_t - t0) because m is a float.
void remember_wait(DriveMemory& mem, int i, int il, int u, int av, double event_t, double t0) {
    auto& m = mem[i][il];
    auto key = std::make_pair(u, av);
    if (m.find(key) == m.end()) m[key] = event_t - t0;
    else m[key] = m[key] + event_t - t0;
}

void remember_jam(DriveMemory& mem, int i, int il, int u, int av, double penalty) {
    auto& m = mem[i][il];
    auto key = std::make_pair(u, av);
    if (m.find(key) == m.end()) m[key] = penalty;
    else m[key] = m[key] + penalty;
}

struct DriveContext {
    const Model& model;
    const DriveNetwork& D;
    const ModePlan& plan;
    DriveQueue& queue;
    LinkQueues& link_q;
    const ExitQueues* exit_q;  // nullptr unless fifo_link_exit
    double t_s;
};

// Intersection priority rule (thesis Fig. "Example of interactions at intersections"), shared
// by network entry, gap opening and check-me.
//
// A car wants to enter link (u, av) at event_t. It looks at the events of the next t_s hours:
// every check-me event of a link INTO u whose car also wants (u, av) is a conflict. A conflict
// costs one of the `target` free gaps if
//   * the subject is not yet on the network (subject_o < 0): it always gives way, or
//   * the conflicting car comes from a strictly higher-priority road (lower priority number).
// If the free gaps run out, the subject has no priority and must wait.
//
// The scan pops events from the queue and pushes them back afterwards. It also pops the
// conflicting cars from their link queues to inspect them, and restores them too.
bool has_priority(DriveContext& ctx, int u, int av, int target, double event_t, double subject_o,
                  double subject_prio) {
    const auto& D = ctx.D;
    const bool legacy = ctx.model.cfg.legacy_mode;
    const std::vector<int>& in_nodes = D.inv_adj[u];
    if (in_nodes[0] == -1) return true;  // nothing flows into u

    std::vector<int> in_access;  // legacy_mode membership test only (see rule 2 in the header)
    if (legacy)
        for (int in : in_nodes)
            for (int ia = 0; ia < static_cast<int>(D.adj[in].size()); ++ia)
                if (D.adj[in][ia] == u) in_access.push_back(ia);

    bool prio = true;
    bool seek = true;
    DriveQueue proxy;
    std::map<std::pair<int, int>, DriveQueue> taken;  // cars popped from link queues

    while (seek && !ctx.queue.empty()) {
        DriveEvent looking = ctx.queue.top();
        ctx.queue.pop();
        proxy.push(looking);

        if (!(std::get<0>(looking) < event_t + ctx.t_s)) {
            seek = false;  // no more events within the safety time
            continue;
        }
        // (g < 1.5 excludes the EXIT events of fifo_link_exit)
        const bool is_check_me = !std::get<1>(looking) && std::get<4>(looking) > 0.5 && std::get<4>(looking) < 1.5;
        if (!is_check_me) continue;
        const int in = std::get<2>(looking), ia = std::get<3>(looking);
        bool enters_u;
        if (legacy)
            enters_u = std::find(in_nodes.begin(), in_nodes.end(), in) != in_nodes.end() &&
                       std::find(in_access.begin(), in_access.end(), ia) != in_access.end();
        else
            enters_u = D.adj[in][ia] == u;  // the link (in, ia) itself ends at u
        if (!enters_u) continue;
        // Rule 5 (FIFO): that car will queue behind the cars already at the end of its link, so
        // it cannot enter u within this window and is no conflict.
        if (ctx.exit_q && !(*ctx.exit_q)[in][ia].empty()) continue;
        // Every pending check-me event has its car in the link queue, so this never skips in a
        // consistent state.
        if (ctx.link_q[in][ia].empty()) continue;

        DriveEvent c_agent = ctx.link_q[in][ia].top();
        ctx.link_q[in][ia].pop();
        taken[std::make_pair(in, ia)].push(c_agent);

        const int c_i = std::get<2>(c_agent), c_il = std::get<3>(c_agent);
        const size_t c_o = static_cast<size_t>(std::get<4>(c_agent));
        const auto& c_path = ctx.plan.paths[c_i][c_il];
        const int c_u = c_path[c_o + 1];
        if (c_u == c_path.back()) continue;  // the other car leaves the network at u: no conflict
        const int c_av = ctx.plan.access[c_i][c_il][c_o + 1];

        if (u == c_u && av == c_av) {  // it wants the same link
            bool yields = subject_o < 0 || ctx.model.link_priority(in, ia) < subject_prio;
            if (yields) {
                target -= 1;
                if (target == 0) {
                    prio = false;
                    seek = false;
                }
            }
        }
    }

    while (!proxy.empty()) {
        ctx.queue.push(proxy.top());
        proxy.pop();
    }
    for (auto& entry : taken)
        while (!entry.second.empty()) {
            ctx.link_q[entry.first.first][entry.first.second].push(entry.second.top());
            entry.second.pop();
        }
    return prio;
}

}  // namespace

DriveStats Model::simulate_drive(int it) {
    (void)it;
    const auto& D = net.drive;
    const auto& paths = drive.paths;
    const auto& access = drive.access;
    const bool legacy = cfg.legacy_mode;
    // Sample scaling of the flow capacity: a gap reopens after t_s / capacity_scale.
    // With capacity_scale = 1 this is exactly cfg.t_s.
    const double t_s = cfg.t_s / capacity_scale;
    // Link-exit options (load() rejects them in legacy_mode).
    const bool fifo = cfg.fifo_link_exit;
    const double stuck_time = cfg.stuck_time;
    DriveStats stats;
    stats.agents = in_drive_simulation_;

    // ---- Road state ------------------------------------------------------------------------
    drive_link_space = drive_capacity;
    drive_mouth_space = D.lanes;
    LinkQueues request_q(D.lanes.size());
    LinkQueues link_q(D.lanes.size());
    ExitQueues exit_q(fifo ? D.lanes.size() : 0);
    for (size_t u = 0; u < D.lanes.size(); ++u) {
        request_q[u].resize(D.lanes[u].size());
        link_q[u].resize(D.lanes[u].size());
        if (fifo) exit_q[u].resize(D.lanes[u].size());
    }
    // Rule 5: time each car reached the end of its current link (its waiting starts there).
    std::vector<double> queued_since(fifo ? N : 0, 0.0);

    DriveQueue& q = drive_queue_;
    DriveContext ctx{*this, D, drive, q, link_q, fifo ? &exit_q : nullptr, t_s};

    // A car left link (u, av). If the link was full and someone waits, wake them up.
    // (With stuck_time the space can be negative; de-saturation fires when it is back to 1.)
    auto free_link = [&](int u, int av, double event_t) {
        drive_link_space[u][av] = drive_link_space[u][av] + 1;
        if (drive_link_space[u][av] == 1 && request_q[u][av].size() > 0.5)
            q.push(DriveEvent(event_t, false, u, av, DESATURATION));
    };
    // Rule 6: a car joining request_q[u][av] is checked again once it could be stuck.
    auto schedule_stuck_check = [&](int u, int av, double event_t) {
        if (stuck_time > 0) q.push(DriveEvent(event_t + stuck_time, false, u, av, RETRY));
    };
    // Rule 6: the first car of a request queue has waited at least stuck_time since joining it.
    // (Small tolerance: the stuck check event is scheduled exactly stuck_time later.)
    auto is_stuck = [&](const DriveEvent& waiting, double event_t) {
        return stuck_time > 0 && event_t - std::get<0>(waiting) >= stuck_time - 1e-12;
    };
    // Rule 5: the first car of exit_q[u][av] has left the link; let the next one try now.
    auto exit_q_advance = [&](int u, int av, double event_t) {
        exit_q[u][av].pop_front();
        if (!exit_q[u][av].empty()) q.push(DriveEvent(event_t, false, u, av, EXIT));
    };
    // Arrival at the trip destination; wait there until the next planned departure.
    auto arrive = [&](int i, int il, double tiempo) {
        traj_drive.add(i, -2 - il, tiempo);
        if (il + 1.5 < drive.departure[i].size()) {
            if (tiempo < drive.departure[i][il + 1]) {
                tiempo = drive.departure[i][il + 1];
                traj_drive.add(i, -2 - il, tiempo);
            }
        }
        q.push(DriveEvent(tiempo, true, i, il + 1, -1));
    };
    // Priority of the link a car is currently on (o = its index in the path).
    auto prio_of = [&](int i, int il, double o) {
        size_t k = static_cast<size_t>(o);
        return link_priority(paths[i][il][k], access[i][il][k]);
    };
    // Rule 1: after giving way, try the link again once the conflicting cars have passed.
    auto schedule_retry = [&](int u, int av, double event_t) {
        if (!legacy) q.push(DriveEvent(event_t + t_s, false, u, av, RETRY));
    };
    // Admits the first car of request_q[u][av] (its gap is taken by the caller's accounting).
    auto admit_from_request = [&](int u, int av, double event_t, bool take_gap) {
        DriveEvent agent = request_q[u][av].top();
        request_q[u][av].pop();
        const int i = std::get<2>(agent);
        const int il = std::get<3>(agent);
        const double o = std::get<4>(agent);
        link_q[u][av].push(DriveEvent(event_t, true, i, il, std::floor(o) + 1));
        traj_drive.add(i, u, event_t);
        drive_link_space[u][av] = drive_link_space[u][av] - 1;
        if (take_gap) drive_mouth_space[u][av] = drive_mouth_space[u][av] - 1;
        double waiting_since = std::get<0>(agent);
        if (o > -0.1) {  // it was on a previous link: free it
            size_t k = static_cast<size_t>(o);
            const int p_u = paths[i][il][k], p_av = access[i][il][k];
            free_link(p_u, p_av, event_t);
            if (fifo) {  // rule 5: it was the first car of that link's exit queue
                exit_q_advance(p_u, p_av, event_t);
                waiting_since = queued_since[i];
            }
        }
        remember_wait(short_drive_mem, i, il, u, av, event_t, waiting_since);
        q.push(DriveEvent(event_t + t_s, false, u, av, GAP));
        q.push(DriveEvent(event_t + D.times[u][av], false, u, av, CHECK_ME));
    };
    // A car at the end of link (u, av) tries to leave it: to its destination (last link) or
    // into its next link. Returns true if it left, false if it now waits in the next link's
    // request_q (and still occupies (u, av)). Without fifo_link_exit it is called directly at
    // the check-me event.
    auto try_leave = [&](const DriveEvent& agent, int u, int av, double event_t) {
        const int i = std::get<2>(agent);
        const int il = std::get<3>(agent);
        const double o = std::get<4>(agent);
        const size_t k = static_cast<size_t>(o);
        // Rule 5: time spent queuing behind other cars at the end of this link.
        const bool waited = fifo && event_t > queued_since[i];

        if (o + 2.5 > paths[i][il].size()) {
            // Last link of the route: leave the network and drive to the destination.
            if (waited) {
                remember_wait(short_drive_mem, i, il, u, av, event_t, queued_since[i]);
                traj_drive.add(i, D.adj[u][av], event_t);  // when it actually leaves the network
            }
            free_link(u, av, event_t);
            int v = D.adj[u][av];
            double tiempo = event_t + geo_km(D.lat[v], D.lon[v], pop.lat[i][il + 1], pop.lon[i][il + 1]) / cfg.car_access_speed;
            arrive(i, il, tiempo);
            return true;
        }
        int n_u = paths[i][il][k + 1];
        int n_av = access[i][il][k + 1];
        if (drive_mouth_space[n_u][n_av] > 0.5 && drive_link_space[n_u][n_av] > 0.5) {
            bool prio = has_priority(ctx, n_u, n_av, drive_mouth_space[n_u][n_av], event_t, o, prio_of(i, il, o));
            if (prio) {
                if (waited) remember_wait(short_drive_mem, i, il, n_u, n_av, event_t, queued_since[i]);
                free_link(u, av, event_t);
                link_q[n_u][n_av].push(DriveEvent(event_t, true, i, il, std::floor(o) + 1));
                traj_drive.add(i, n_u, event_t);
                drive_link_space[n_u][n_av] = drive_link_space[n_u][n_av] - 1;
                drive_mouth_space[n_u][n_av] = drive_mouth_space[n_u][n_av] - 1;
                q.push(DriveEvent(event_t + t_s, false, n_u, n_av, GAP));
                q.push(DriveEvent(event_t + D.times[n_u][n_av], false, n_u, n_av, CHECK_ME));
                return true;
            }
            // gives way at the intersection
            request_q[n_u][n_av].push(DriveEvent(event_t, true, i, il, o));
            traj_drive.add(i, n_u, event_t);
            schedule_retry(n_u, n_av, event_t);
            schedule_stuck_check(n_u, n_av, event_t);
            return false;
        }
        // Next link full or no gap: wait at the end of this link, which stays occupied; a
        // pending gap opening / de-saturation (or the stuck check) will admit the car.
        // Without fifo_link_exit the car has left link_q, so cars behind it can still exit the
        // link to other links (rule 5 makes them wait).
        request_q[n_u][n_av].push(DriveEvent(event_t, true, i, il, o));
        traj_drive.add(i, n_u, event_t);
        schedule_stuck_check(n_u, n_av, event_t);
        return false;
    };
    // Rule 5: the first car of exit_q[u][av] tries to leave; if it does, the next one follows.
    auto attempt_exit = [&](int u, int av, double event_t) {
        if (exit_q[u][av].empty()) return;
        const DriveEvent agent = exit_q[u][av].front();
        if (try_leave(agent, u, av, event_t)) exit_q_advance(u, av, event_t);
    };

    MIMOS_LOG(log, INFO, "DRIVE EXECUTION");
    while (q.empty() == false && in_drive_simulation_ > 0.5) {
        DriveEvent next = q.top();
        q.pop();
        ++stats.events;
        double event_t = std::get<0>(next);
        progress_clock_.store(event_t, std::memory_order_relaxed);

        if (std::get<1>(next)) {
            const int i = std::get<2>(next);
            const int il = std::get<3>(next);
            const double o = std::get<4>(next);

            if (o < -0.6) {
                // ---- Exit from location -------------------------------------------------
                traj_drive.add(i, -1 - il, event_t);
                if (il + 1.5 < pop.lat[i].size()) {
                    if (paths[i][il].size() > 1.5) {  // drive: go to the first node of the route
                        int u = paths[i][il][0];
                        event_t = event_t + geo_km(pop.lat[i][il], pop.lon[i][il], D.lat[u], D.lon[u]) / cfg.car_access_speed;
                        traj_drive.add(i, u, event_t);
                        q.push(DriveEvent(event_t, true, i, il, -0.5));
                    } else {  // no road route: walk
                        double tiempo = event_t + geo_km(pop.lat[i][il], pop.lon[i][il], pop.lat[i][il + 1], pop.lon[i][il + 1]) / 4.0;
                        arrive(i, il, tiempo);
                    }
                } else {  // end of the day
                    --in_drive_simulation_;
                    finished_exec[i] = finished_exec[i] + 1;
                    ++stats.finished;
                }
            } else {
                // ---- Entry to road network (o == -0.5) -------------------------------------
                int u = paths[i][il][0];
                int av = access[i][il][0];
                if (drive_mouth_space[u][av] > 0.5 && drive_link_space[u][av] > 0.5) {
                    bool prio = has_priority(ctx, u, av, drive_mouth_space[u][av], event_t, o, 0);
                    if (prio) {
                        link_q[u][av].push(DriveEvent(event_t, true, i, il, 0));
                        traj_drive.add(i, u, event_t);
                        drive_link_space[u][av] = drive_link_space[u][av] - 1;
                        drive_mouth_space[u][av] = drive_mouth_space[u][av] - 1;
                        q.push(DriveEvent(event_t + t_s, false, u, av, GAP));
                        q.push(DriveEvent(event_t + D.times[u][av], false, u, av, CHECK_ME));
                    } else {  // gives way to cars already on the network
                        request_q[u][av].push(DriveEvent(event_t, true, i, il, o));
                        traj_drive.add(i, u, event_t);
                        schedule_retry(u, av, event_t);
                        schedule_stuck_check(u, av, event_t);
                    }
                } else {  // a pending gap opening or de-saturation will admit it
                    request_q[u][av].push(DriveEvent(event_t, true, i, il, -0.5));
                    traj_drive.add(i, u, event_t);
                    schedule_stuck_check(u, av, event_t);
                }
            }
            continue;
        }

        const int u = std::get<2>(next);
        const int av = std::get<3>(next);
        const double g = std::get<4>(next);

        if (g < -0.75) {
            // ---- Gap opening at the link entrance ----------------------------------------
            // Rule 6: a stuck car may enter a full link.
            if (request_q[u][av].size() > 0.5 &&
                (drive_link_space[u][av] > 0.5 || is_stuck(request_q[u][av].top(), event_t))) {
                const DriveEvent& agent = request_q[u][av].top();
                const int i = std::get<2>(agent);
                const int il = std::get<3>(agent);
                const double o = std::get<4>(agent);
                // +1: this event is about to give a gap back
                double subject_prio = o < 0 ? 0 : prio_of(i, il, o);
                bool prio = drive_link_space[u][av] > 0.5 &&
                            has_priority(ctx, u, av, drive_mouth_space[u][av] + 1, event_t, o, subject_prio);
                if (!prio && is_stuck(agent, event_t)) {  // rule 6: no more giving way
                    prio = true;
                    ++stats.stuck_moves;
                }
                if (prio) {
                    // The opened gap is used immediately, so the mouth count does not change.
                    admit_from_request(u, av, event_t, /*take_gap=*/false);
                } else {
                    // Still gives way: the gap goes back to the mouth (the car stays first in
                    // the queue).
                    traj_drive.add(i, u, event_t);
                    drive_mouth_space[u][av] = drive_mouth_space[u][av] + 1;
                    schedule_retry(u, av, event_t);
                }
            } else {
                drive_mouth_space[u][av] = drive_mouth_space[u][av] + 1;
            }
        } else if (g < -0.25) {
            // ---- Retry after giving way (rule 1, never scheduled in legacy_mode) -----------
            // Same rule as a car entering with the gaps currently free. If there is no free gap
            // or no space, a pending gap opening / de-saturation will handle the queue.
            // Rule 6: also the stuck check; a stuck car needs a gap but no space or priority.
            if (request_q[u][av].size() > 0.5 && drive_mouth_space[u][av] > 0.5 &&
                (drive_link_space[u][av] > 0.5 || is_stuck(request_q[u][av].top(), event_t))) {
                const DriveEvent& agent = request_q[u][av].top();
                const int i = std::get<2>(agent);
                const int il = std::get<3>(agent);
                const double o = std::get<4>(agent);
                double subject_prio = o < 0 ? 0 : prio_of(i, il, o);
                bool prio = drive_link_space[u][av] > 0.5 &&
                            has_priority(ctx, u, av, drive_mouth_space[u][av], event_t, o, subject_prio);
                if (!prio && is_stuck(agent, event_t)) {
                    prio = true;
                    ++stats.stuck_moves;
                }
                if (prio)
                    admit_from_request(u, av, event_t, /*take_gap=*/true);
                else
                    schedule_retry(u, av, event_t);
            }
        } else if (g > 1.5) {
            // ---- Exit attempt of the next car at the end of the link (rule 5) -------------
            attempt_exit(u, av, event_t);
        } else if (g > 0.5) {
            // ---- Check-me: the first car of the link reaches its end ----------------------
            if (link_q[u][av].size() > 0.5) {
                DriveEvent agent = link_q[u][av].top();
                link_q[u][av].pop();
                const int i = std::get<2>(agent);
                traj_drive.add(i, D.adj[u][av], event_t);
                if (fifo) {
                    // Rule 5: queue at the end of the link; only the first car tries to leave.
                    queued_since[i] = event_t;
                    exit_q[u][av].push_back(agent);
                    if (exit_q[u][av].size() == 1) attempt_exit(u, av, event_t);
                } else {
                    try_leave(agent, u, av, event_t);
                }
            }
        } else {
            // ---- De-saturation: a full link has space again -------------------------------
            if (request_q[u][av].empty()) continue;  // nobody waits any more
            // No priority rule: the road is congested, so first-come-first-served.
            if (drive_mouth_space[u][av] > 0.5 && drive_link_space[u][av] > 0.5)
                admit_from_request(u, av, event_t, /*take_gap=*/true);
            // else: no gap or no space; the pending gap opening will admit the car.
        }
    }

    // ---- Pending gap openings -------------------------------------------------------------
    // The loop stops as soon as every agent has finished its day. Cars that entered
    // a link less than t_s before that still have a gap opening in the queue, so that link's
    // mouth would end one gap short and check_invariants() would report a false violation.
    // Apply those gap openings; the remaining events
    // (check-me of cars that already left, retries, de-saturations) concern no agent any more.
    // Outputs do not change: no agent is involved.
    if (in_drive_simulation_ <= 0.5)
        while (!q.empty()) {
            const DriveEvent e = q.top();
            q.pop();
            if (!std::get<1>(e) && std::get<4>(e) < -0.75)
                drive_mouth_space[std::get<2>(e)][std::get<3>(e)] += 1;
        }

    // ---- Gridlock --------------------------------------------------------------------------
    // Cars still inside links or waiting to enter one never finished. Each gets
    // cfg.jam_penalty hours of delay on that link in its short memory ("memory shock").
    for (size_t u = 0; u < link_q.size(); ++u)
        for (size_t av = 0; av < link_q[u].size(); ++av)
            while (link_q[u][av].size() > 0.5) {
                DriveEvent agent = link_q[u][av].top();
                link_q[u][av].pop();
                int i = std::get<2>(agent), il = std::get<3>(agent);
                traj_drive.add(i, traj_drive.nodes[i].back(), traj_drive.times[i].back() + cfg.jam_penalty);
                remember_jam(short_drive_mem, i, il, static_cast<int>(u), static_cast<int>(av), cfg.jam_penalty);
                ++stats.jammed_in_links;
            }
    std::vector<bool> counted(fifo ? N : 0, false);  // fix 5: count each car once
    for (size_t u = 0; u < request_q.size(); ++u)
        for (size_t av = 0; av < request_q[u].size(); ++av)
            while (request_q[u][av].size() > 0.5) {
                DriveEvent agent = request_q[u][av].top();
                request_q[u][av].pop();
                int i = std::get<2>(agent), il = std::get<3>(agent);
                remember_jam(short_drive_mem, i, il, static_cast<int>(u), static_cast<int>(av), cfg.jam_penalty);
                ++stats.jammed_in_requests;
                if (fifo) counted[i] = true;
            }
    // Fix 5: cars queued at the end of a link. The first one of each queue is also waiting in
    // its next link's request_q (counted above); the others are counted as cars in links.
    for (size_t u = 0; u < exit_q.size(); ++u)
        for (size_t av = 0; av < exit_q[u].size(); ++av)
            for (const DriveEvent& agent : exit_q[u][av]) {
                int i = std::get<2>(agent), il = std::get<3>(agent);
                if (counted[i]) continue;
                traj_drive.add(i, traj_drive.nodes[i].back(), traj_drive.times[i].back() + cfg.jam_penalty);
                remember_jam(short_drive_mem, i, il, static_cast<int>(u), static_cast<int>(av), cfg.jam_penalty);
                ++stats.jammed_in_links;
            }
    if (stats.jammed_in_links + stats.jammed_in_requests > 0)
        MIMOS_LOG(log, WARNING, "Drive gridlock: %d cars in links, %d waiting to enter", stats.jammed_in_links, stats.jammed_in_requests);
    if (stats.stuck_moves > 0)
        MIMOS_LOG(log, INFO, "Stuck time: %d cars moved on after waiting %.0f s", stats.stuck_moves, stuck_time * 3600);
    return stats;
}

}  // namespace mimos
