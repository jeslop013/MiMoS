// Public transport simulation (thesis "Public transport simulation", Algorithm "Public
// transport mode simulation algorithm" and its appendix sub-procedures).
//
// Vehicles are not simulated as moving objects: they follow the schedules exactly. The only
// interaction is occupancy (space_map: vehicle id -> free seats). An agent who finds a full
// vehicle takes a ticket from the stop (stop_queue_ticket) and retries at the next vehicle;
// the ticket is the event's tie-breaker, so boarding at a stop is first-in-first-out.
//
// Event sub-procedures (thesis colour boxes):
//   o < 0                          "Exit from location"
//   boarding < 1                   "Unloading" of vehicle -boarding
//   vehicle found, space           "Propagate down transport itinerary"
//   vehicle found, full            "Saturation of transport"
//   no vehicle left today          "Absence of transport"
// Delays caused by interaction (waiting for a later vehicle, walking after missing the last one)
// are written to short_public_mem[i][il][stop] (thesis "Short memory").
#include "mimos/model.hpp"

namespace mimos {

namespace {
// Adds `delay` hours to agent i's short memory for stop `node` of trip il.
void remember(PublicMemory& mem, int i, int il, int node, double delay) {
    auto& m = mem[i][il];
    if (m.find(node) == m.end()) m[node] = delay;
    else m[node] = m[node] + delay;
}
}  // namespace

PublicStats Model::simulate_public(int it) {
    (void)it;
    const auto& P = net.pub;
    const auto& paths = pub.paths;
    const auto& access = pub.access;
    PublicStats stats;
    stats.agents = in_public_simulation_;

    // Every vehicle starts empty.
    space_map.clear();
    for (size_t u = 0; u < P.size(); ++u)
        if (P.stop[u] == 0 && P.mode[u] > -0.5) {
            int cap = public_capacity(static_cast<int>(u));
            for (int v : P.id[u]) space_map[v] = cap;
        }
    std::vector<int> stop_queue_ticket(P.size(), 1);

    auto walk_hours = [&](double la1, double lo1, double la2, double lo2) { return geo_km(la1, lo1, la2, lo2) / 4.0; };
    // Arrival at the trip destination: record it, then wait for the next departure if early.
    auto arrive_at_destination = [&](int i, int il, double tiempo) {
        traj_public.add(i, -2 - il, tiempo);
        if (il + 1.5 < pub.departure[i].size())
            if (tiempo < pub.departure[i][il + 1]) tiempo = pub.departure[i][il + 1];
        public_queue_.push(PublicEvent(tiempo, 1000000, i, il + 1, -1));
    };

    MIMOS_LOG(log, INFO, "PUBLIC EXECUTION");
    while (public_queue_.empty() == false && in_public_simulation_ > -0.5) {
        PublicEvent next = public_queue_.top();
        public_queue_.pop();
        ++stats.events;

        double event_t = std::get<0>(next);
        progress_clock_.store(event_t, std::memory_order_relaxed);
        int in_transport = std::get<1>(next);
        int i = std::get<2>(next);
        int il = std::get<3>(next);
        int o = std::get<4>(next);
        double next_t = event_t;

        // Every event records where the agent is.
        if (o < 0) traj_public.add(i, -1 - il, event_t);
        else traj_public.add(i, paths[i][il][o], event_t);

        if (o < 0) {
            // ---- Exit from location -------------------------------------------------------
            if (il + 1.5 > pop.lat[i].size()) {  // final location of the day
                in_public_simulation_ = in_public_simulation_ - 1;
                finished_exec[i] = finished_exec[i] + 1;
                ++stats.finished;
            } else if (paths[i][il].size() < 0.5) {  // no public route: walk the whole trip
                double tiempo = event_t + walk_hours(pop.lat[i][il], pop.lon[i][il], pop.lat[i][il + 1], pop.lon[i][il + 1]);
                arrive_at_destination(i, il, tiempo);
            } else {  // walk to the first stop of the route
                int first = paths[i][il][0];
                public_queue_.push(PublicEvent(event_t + walk_hours(pop.lat[i][il], pop.lon[i][il], P.lat[first], P.lon[first]),
                                               1000000, i, il, 0));
            }
            continue;
        }

        if (in_transport < 1) {
            // ---- Unloading of vehicle -in_transport ----------------------------------------
            space_map[-in_transport] = space_map[-in_transport] + 1;
            if (o + 1.5 > paths[i][il].size()) {  // last stop of the route: walk to destination
                int node = paths[i][il][o];
                arrive_at_destination(i, il, event_t + walk_hours(P.lat[node], P.lon[node], pop.lat[i][il + 1], pop.lon[i][il + 1]));
            } else {  // transfer to the next node of the route
                public_queue_.push(PublicEvent(event_t + P.times[paths[i][il][o]][access[i][il][o]], 1000000, i, il, o + 1));
            }
            continue;
        }

        // ---- Agent at a stop, waiting for a vehicle ----------------------------------------
        int u = paths[i][il][o];
        bool found = false;
        int found_id = -1;

        // "Stepping stop" (thesis "Routing routine"): the route only passes through this stop
        // to transfer again, without boarding. Treated as if no vehicle came, so the agent
        // walks on to the next itinerary or the destination.
        bool stepping_stop = false;
        if (o + 1.5 < paths[i][il].size()) {
            if (P.itin[u] != P.itin[paths[i][il][o + 1]]) {
                MIMOS_LOG(log, WARNING, "STEPPING STOP! Agent %d jumping across stops %d - %d without actually taking them. Mark them as finding no transport", i, u, paths[i][il][o + 1]);
                stepping_stop = true;
            }
        } else {
            MIMOS_LOG(log, WARNING, "STEPPING STOP ON LAST STOP (Node %d)? Look it up!", paths[i][il][o]);
            stepping_stop = true;
        }

        if (stepping_stop == false) {
            if (P.mode[u] > -0.5) {  // next scheduled vehicle
                int ii = -1;
                for (double sch : P.sch[u]) {
                    ++ii;
                    if (sch >= event_t) {
                        next_t = sch;
                        found = true;
                        found_id = P.id[u][ii];
                        break;
                    }
                }
            } else {
                found = true;  // rental bikes are always available
            }
        }

        // in_transport < 1e5 means the agent was already queued here (it holds a ticket), so
        // the time since then is a delay caused by other agents.
        const bool was_queued = in_transport < 1e5;

        if (found == false) {
            // ---- Absence of transport: skip to the next itinerary or walk to destination ----
            int itin1 = P.itin[u];
            int o1 = o;
            int v = paths[i][il][o1];
            bool straight_to_dest = false;
            while (itin1 == P.itin[v]) {
                ++o1;
                if (o1 + 1.5 > paths[i][il].size()) {
                    straight_to_dest = true;
                    break;
                }
                v = paths[i][il][o1];
            }

            int here = paths[i][il][o];
            if (straight_to_dest == true) {
                double walk = walk_hours(P.lat[here], P.lon[here], pop.lat[i][il + 1], pop.lon[i][il + 1]);
                arrive_at_destination(i, il, event_t + walk);
                if (was_queued) remember(short_public_mem, i, il, here, walk);
            } else {
                int there = paths[i][il][o1];
                double walk = walk_hours(P.lat[here], P.lon[here], P.lat[there], P.lon[there]);
                public_queue_.push(PublicEvent(event_t + walk, 1000000, i, il, o1));
                if (was_queued) remember(short_public_mem, i, il, here, walk);
            }
            continue;
        }

        // ---- A vehicle arrives at next_t --------------------------------------------------
        traj_public.add(i, paths[i][il][o], next_t);
        if (was_queued) {
            // Written out (not via remember()) to keep a fixed evaluation order
            // (m + next_t) - event_t, which rounds differently from m + (next_t - event_t).
            auto& m = short_public_mem[i][il];
            int node = paths[i][il][o];
            if (m.find(node) == m.end()) m[node] = next_t - event_t;
            else m[node] = m[node] + next_t - event_t;
        }

        bool space = true;
        if (P.mode[u] > -0.5)
            if (space_map[found_id] == 0) space = false;

        if (!space) {
            // ---- Saturation of transport: wait for the next vehicle, keeping queue order ---
            if (in_transport > -0.5 && in_transport < 1e5) {
                public_queue_.push(PublicEvent(next_t + 1.0 / 3600.0, in_transport, i, il, o));
            } else {
                public_queue_.push(PublicEvent(next_t + 1.0 / 3600.0, stop_queue_ticket[paths[i][il][o]], i, il, o));
                stop_queue_ticket[paths[i][il][o]] = stop_queue_ticket[paths[i][il][o]] + 1;
            }
            continue;
        }

        // ---- Propagate down the itinerary -------------------------------------------------
        if (P.mode[u] > -0.5) space_map[found_id] = space_map[found_id] - 1;

        int itin1 = P.itin[u];
        int o1 = o;
        int uu = paths[i][il][o1];
        int av = access[i][il][o1];
        double prop_time = 0.0;
        while (itin1 == P.itin[paths[i][il][o1 + 1]]) {
            uu = paths[i][il][o1];
            av = access[i][il][o1];
            prop_time = prop_time + P.times[uu][av];
            traj_public.add(i, P.adj[uu][av], next_t + prop_time);
            ++o1;
            if (o1 + 1.5 > paths[i][il].size()) break;
        }

        // This tests the mode of `uu`, the start node of the LAST link travelled, not the
        // boarding stop. Both are on the same itinerary, so they have the same mode.
        if (P.mode[uu] > -0.5) {
            // Vehicle: unload at the last stop of this itinerary on the route.
            public_queue_.push(PublicEvent(next_t + prop_time, -found_id, i, il, o1));
        } else if (o1 + 1.5 > paths[i][il].size()) {
            // Rental bike to the end of the route, then walk to the destination.
            int node = paths[i][il][o1];
            arrive_at_destination(i, il, event_t + prop_time + walk_hours(P.lat[node], P.lon[node], pop.lat[i][il + 1], pop.lon[i][il + 1]));
        } else {
            // Rental bike, then walk to the next stop of the route.
            // The walk takes the network time of this transfer link (times_public), as planned;
            // legacy_mode uses the straight-line walk at 4 km/h of the thesis version.
            int a = paths[i][il][o1], b = paths[i][il][o1 + 1];
            double transfer = cfg.legacy_mode ? walk_hours(P.lat[a], P.lon[a], P.lat[b], P.lon[b])
                                              : P.times[a][access[i][il][o1]];
            public_queue_.push(PublicEvent(event_t + prop_time + transfer, 1000000, i, il, o1 + 1));
        }
    }
    return stats;
}

}  // namespace mimos
