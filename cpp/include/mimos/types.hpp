// Shared types of the MiMoS core: event tuples, access/egress candidates, geo helper.
#pragma once

#include <cmath>
#include <functional>
#include <queue>
#include <tuple>
#include <vector>

namespace mimos {

// ------------------------------------------------------------------------------------------
// Events (thesis "Simulation of transportation modes"). Both simulators are discrete-event:
// a std::priority_queue with std::greater pops the SMALLEST tuple first, so the tuple
// field order IS the tie-breaking rule. Do not reorder fields.
// ------------------------------------------------------------------------------------------

// Private transport event: (time, agent_bool, u|i, av|il, g|o)
//   agent_bool == true  -> an agent at an agenda location (o == -1) or about to enter the
//                          network at the first node of its trip (o == -0.5), with i = agent,
//                          il = trip index. Inside link queues, o = index of the link in
//                          the planned path (link = path[o] -> path[o+1]).
//   agent_bool == false -> a road event on link (u, av), where av is the position of v in
//                          adj_drive[u]: g = -1 gap opening, g = -0.5 retry,
//                          g = 0 de-saturation, g = +1 check-me (first car reaches the end
//                          of the link), g = +2 exit attempt of the next car in the link's
//                          exit queue (fifo_link_exit only).
//   At equal times road events (false) are processed before agent events (true), and for
//   the same link gap (-1) < retry (-0.5) < de-saturation (0) < check-me (+1) < exit (+2).
using DriveEvent = std::tuple<double, bool, int, int, double>;
using DriveQueue = std::priority_queue<DriveEvent, std::vector<DriveEvent>, std::greater<DriveEvent>>;

// Public transport event: (time, boarding, i, il, o)
//   boarding >= 1e5 (1e6 is used) -> agent walking or arriving to a stop;
//   0 < boarding < 1e5            -> agent queued at a stop, value = its stop ticket (FIFO);
//   boarding < 1                  -> agent inside the vehicle with id -boarding, about to unload.
//   o < 0 -> agent at an agenda location; o >= 0 -> index of the current node in the planned
//   path. Unloading (negative boarding) sorts before boarding at equal times, so vehicles
//   free their space before new passengers board.
using PublicEvent = std::tuple<double, int, int, int, int>;
using PublicQueue = std::priority_queue<PublicEvent, std::vector<PublicEvent>, std::greater<PublicEvent>>;

// Access (source) or egress (target) candidate of a trip: (score, time, node).
// Precomputed before MiMoS runs (docs/input_format.md), scored on load.
using Access = std::tuple<double, double, int>;
using AccessTable = std::vector<std::vector<std::vector<Access>>>;  // [agent][trip][candidate]

// ------------------------------------------------------------------------------------------
// Great-circle distance in km.
// haversine() takes FLOAT arguments, as in the thesis version, so the double coordinates are
// rounded to float first. Used only in legacy_mode; the default is haversine_d (below).
// ------------------------------------------------------------------------------------------
constexpr double PI = 3.14159265358979323846;
constexpr double EARTH_RADIUS_KM = 6372.79756085;

inline double haversine(float lat1, float lon1, float lat2, float lon2) {
    // Written as (x * PI) / 180 to keep the rounding of the thesis version.
    lat1 = lat1 * PI / 180;
    lon1 = lon1 * PI / 180;
    lat2 = lat2 * PI / 180;
    lon2 = lon2 * PI / 180;
    double h = (std::pow(std::sin((1.0 / 2) * (lat2 - lat1)), 2)) +
               ((std::cos(lat1)) * (std::cos(lat2)) * (std::pow(std::sin((1.0 / 2) * (lon2 - lon1)), 2)));
    double temp = 2 * std::asin(std::min(1.0, std::sqrt(h)));
    return EARTH_RADIUS_KM * temp;
}

// Great-circle distance in km computed entirely in double precision (default).
inline double haversine_d(double lat1, double lon1, double lat2, double lon2) {
    const double r = PI / 180;
    lat1 *= r;
    lon1 *= r;
    lat2 *= r;
    lon2 *= r;
    double h = std::pow(std::sin(0.5 * (lat2 - lat1)), 2) +
               std::cos(lat1) * std::cos(lat2) * std::pow(std::sin(0.5 * (lon2 - lon1)), 2);
    return EARTH_RADIUS_KM * 2 * std::asin(std::min(1.0, std::sqrt(h)));
}

// Walking speed (km/h) used for every off-network or transfer walk.
// The car speed between an agenda location and the road network is Config::car_access_speed.
constexpr double WALK_SPEED = 4.0;

}  // namespace mimos
