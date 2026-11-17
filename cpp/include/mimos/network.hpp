// Transport networks (thesis section "Transportation networks" / "Constructing the networks").
//
// All networks use the thesis' adjacency-list format: row index = node id, and every
// per-edge attribute file mirrors adj_*.csv entry by entry. A link is identified by
// (u, av) where av is the position of v inside adj[u] ("access v"); this avoids searching
// adj[u] for v whenever an edge attribute is needed. Rows of nodes without neighbours hold -1.
//
// Loaded from <data_dir>/net_files/{public,bike,drive}_network/. Units after loading:
// times in hours, lengths in km (files store minutes and metres).
#pragma once

#include <string>
#include <vector>

#include "mimos/logger.hpp"

namespace mimos {

// Public transport: buses, trains and rental bikes. Nodes are stops; edges are the next stop
// of the same itinerary (always the FIRST entry of a row) or walking transfers to the closest
// stop of other itineraries.
struct PublicNetwork {
    // Node attributes (nodes_public.csv: mode, itinerary, stop, lat, lon, name, fee, circular)
    std::vector<int> mode;      // -1 rental bike; 0 urban bus; 1 regional bus; 2 train;
                                // 3 and 4 other urban and regional bus services
    std::vector<int> itin;      // itinerary id (-1 for rental bikes)
    std::vector<int> stop;      // order within the itinerary (0 = header)
    std::vector<double> lat, lon;
    std::vector<std::string> name;
    std::vector<double> fee;
    std::vector<bool> circ;     // circular itinerary
    // Edges
    std::vector<std::vector<int>> adj;
    std::vector<std::vector<double>> times;    // hours
    std::vector<std::vector<double>> lengths;  // km
    // Temporal dynamics
    std::vector<std::vector<double>> sch;      // times at which a vehicle passes the stop (hours)
    std::vector<std::vector<int>> id;          // vehicle id for each entry of sch

    size_t size() const { return lat.size(); }
};

// Bicycle network (OSM).
struct BikeNetwork {
    std::vector<std::string> osm;
    std::vector<double> lat, lon;
    std::vector<std::vector<int>> adj;
    std::vector<std::vector<double>> times;    // hours
    // Raw cycleway flag from cycleway_bike.csv (1 = cycleway, 0 = shared road, -1 = no link).
    std::vector<std::vector<int>> cycle_raw;
    // Interpreted flag, set by Model::load(): value > 0 (legacy_mode: value >= 0).
    std::vector<std::vector<bool>> cycle;

    size_t size() const { return lat.size(); }
};

// Road network for private vehicles (OSM).
struct DriveNetwork {
    std::vector<std::string> osm;
    std::vector<double> lat, lon;
    std::vector<std::vector<int>> adj;
    std::vector<std::vector<int>> inv_adj;     // nodes with an edge INTO the row node
    std::vector<std::vector<double>> times;    // free-flow time, hours
    std::vector<std::vector<double>> lengths;  // km
    std::vector<std::vector<int>> lanes;       // = gaps at the link "mouth"
    std::vector<std::vector<int>> capacity;    // storage: cars that fit (lanes * L / 4.5 m)
    std::vector<std::vector<int>> velocity;    // km/h
    // Road hierarchy, lower = higher priority (0 roundabout ... 6 residential; link roads
    // +0.5). legacy_mode compares the values truncated to int.
    std::vector<std::vector<double>> priority;

    size_t size() const { return lat.size(); }
};

struct Networks {
    PublicNetwork pub;
    BikeNetwork bike;
    DriveNetwork drive;

    // Size of the largest network: the routing work arrays are allocated with this size
    // and shared by the three modes.
    size_t largest() const;
};

// Loads the three networks from <data_dir>/net_files/. Throws std::runtime_error if a file
// cannot be opened.
Networks load_networks(const std::string& data_dir, Logger& log);

}  // namespace mimos
