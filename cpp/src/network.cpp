// Network loaders. The per-file conversions are documented on each call.
#include "mimos/network.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "mimos/csv.hpp"

namespace mimos {

size_t Networks::largest() const {
    // Ties resolve in a fixed order.
    size_t largest = bike.size();
    if (pub.size() > drive.size() && pub.size() > bike.size()) largest = pub.size();
    else if (drive.size() > bike.size()) largest = drive.size();
    return largest;
}

namespace {

// Times are stored in minutes; negative entries (no neighbour) become -1.
std::vector<std::vector<double>> minutes_to_hours(const std::string& file) {
    return csv::read_rows<double>(file, [](const std::string& s) {
        double v = std::stod(s);
        return v < 0 ? -1.0 : v / 60.0;
    });
}

// Lengths are stored in metres; negative entries become -1.
std::vector<std::vector<double>> metres_to_km(const std::string& file) {
    return csv::read_rows<double>(file, [](const std::string& s) {
        double v = std::stod(s);
        return v < 0 ? -1.0 : v / 1000.0;
    });
}

// Integer edge attribute, parsed through stod and truncated to int.
std::vector<std::vector<int>> truncated_ints(const std::string& file) {
    return csv::read_rows<int>(file, [](const std::string& s) { return static_cast<int>(std::stod(s)); });
}

std::vector<std::vector<int>> ints(const std::string& file) {
    return csv::read_rows<int>(file, [](const std::string& s) { return std::stoi(s); });
}

std::vector<std::vector<double>> doubles(const std::string& file) {
    return csv::read_rows<double>(file, [](const std::string& s) { return std::stod(s); });
}

}  // namespace

Networks load_networks(const std::string& data_dir, Logger& log) {
    Networks n;
    const std::string pub = data_dir + "net_files/public_network/";
    const std::string bike = data_dir + "net_files/bike_network/";
    const std::string drv = data_dir + "net_files/drive_network/";

    // ---- Public ------------------------------------------------------------------------
    MIMOS_LOG(log, INFO, "----- LOADING PUBLIC NETWORK -----");
    for (const auto& f : csv::read_fields(pub + "nodes_public.csv", 8)) {
        n.pub.mode.push_back(std::stoi(f[0]));
        n.pub.itin.push_back(std::stoi(f[1]));
        n.pub.stop.push_back(std::stoi(f[2]));
        n.pub.lat.push_back(std::stod(f[3]));
        n.pub.lon.push_back(std::stod(f[4]));
        n.pub.name.push_back(f[5]);
        n.pub.fee.push_back(std::stod(f[6]));
        n.pub.circ.push_back(!(std::stoi(f[7]) < 1));
    }
    n.pub.adj = ints(pub + "adj_public.csv");
    n.pub.times = minutes_to_hours(pub + "times_public.csv");
    n.pub.lengths = metres_to_km(pub + "lengths_public.csv");
    n.pub.sch = doubles(pub + "schedules_public.csv");
    n.pub.id = ints(pub + "transport_id_public.csv");

    // ---- Bike --------------------------------------------------------------------------
    MIMOS_LOG(log, INFO, "----- LOADING BIKE NETWORK -----");
    for (const auto& f : csv::read_fields(bike + "nodes_bike.csv", 3)) {
        n.bike.osm.push_back(f[0]);
        n.bike.lat.push_back(std::stod(f[1]));
        n.bike.lon.push_back(std::stod(f[2]));
    }
    n.bike.adj = ints(bike + "adj_bike.csv");
    n.bike.times = minutes_to_hours(bike + "times_bike.csv");
    n.bike.cycle_raw = ints(bike + "cycleway_bike.csv");  // interpreted in Model::load()

    // ---- Drive -------------------------------------------------------------------------
    MIMOS_LOG(log, INFO, "----- LOADING DRIVE NETWORK -----");
    for (const auto& f : csv::read_fields(drv + "nodes_drive.csv", 3)) {
        n.drive.osm.push_back(f[0]);
        n.drive.lat.push_back(std::stod(f[1]));
        n.drive.lon.push_back(std::stod(f[2]));
    }
    // The csv reader strips '\r' and skips blank rows, so a single-value row such as "1" is
    // parsed as 1 on every platform.
    n.drive.adj = ints(drv + "adj_drive.csv");
    n.drive.inv_adj = ints(drv + "inv_adj_drive.csv");
    n.drive.times = minutes_to_hours(drv + "times_drive.csv");
    n.drive.lengths = metres_to_km(drv + "lengths_drive.csv");
    n.drive.lanes = truncated_ints(drv + "lanes_drive.csv");
    n.drive.capacity = truncated_ints(drv + "capacity_drive.csv");
    n.drive.velocity = truncated_ints(drv + "velocity_drive.csv");
    n.drive.priority = doubles(drv + "priority_drive.csv");  // keeps the .5 of link roads

    MIMOS_LOG(log, INFO, "Networks loaded: public %zu nodes, bike %zu nodes, drive %zu nodes",
              n.pub.size(), n.bike.size(), n.drive.size());
    return n;
}

}  // namespace mimos
