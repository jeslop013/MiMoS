// Python bindings of the MiMoS core (module mimos._core), built by scikit-build-core.
// The Python package (src/mimos/) wraps these in a friendlier API; see mimos.model.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mimos/model.hpp"

namespace py = pybind11;
using namespace mimos;

namespace {
py::array_t<double> to_numpy(const std::vector<double>& v) { return py::array_t<double>(v.size(), v.data()); }
py::array_t<bool> to_numpy(const std::vector<bool>& v) {
    py::array_t<bool> a(v.size());
    auto m = a.mutable_unchecked<1>();
    for (size_t i = 0; i < v.size(); ++i) m(i) = v[i];
    return a;
}
}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "MiMoS C++ core: networks, routing, mode choice, event-driven simulators, memory";

    py::enum_<LogLevel>(m, "LogLevel")
        .value("TRACE", TRACE).value("DEBUG", DEBUG).value("INFO", INFO)
        .value("WARNING", WARNING).value("ERROR", ERROR).value("CRITICAL", CRITICAL);

    // Every Config field is exposed read/write under its C++ name.
#define F(name) .def_readwrite(#name, &Config::name)
    py::class_<Config>(m, "Config")
        .def(py::init<>())
        F(data_dir) F(run_dir) F(n_agents) F(it_number) F(print_it) F(seed) F(log_level) F(threads) F(legacy_mode) F(capacity_scale) F(k_min) F(sample_random) F(sample_seed) F(C_transfer) F(daily_cost_drive) F(daily_cost_public)
        F(beta_m) F(beta_transfer) F(C_limit) F(C_public) F(C_rent_bike) F(C_walk) F(C_bike) F(C_drive)
        F(beta_trav_public) F(beta_d_public) F(gamma_d_public)
        F(beta_trav_rent_bike) F(beta_d_rent_bike) F(gamma_d_rent_bike)
        F(beta_trav_walk) F(beta_d_walk)
        F(beta_trav_bike) F(beta_d_bike) F(gamma_d_bike) F(beta_trav_no_cycleway)
        F(beta_trav_drive) F(beta_trav_delay) F(beta_d_drive) F(gamma_d_drive)
        F(beta_dur) F(beta_late) F(jam_penalty) F(departure_correction)
        F(city_bus_cap) F(reg_bus_cap) F(cercanias_cap) F(ticket) F(bike_ticket)
        F(t_s) F(car_access_speed) F(fifo_link_exit) F(stuck_time) F(alpha_l_i) F(alpha_l_f) F(alpha_f_i) F(alpha_f_f) F(k_max)
        F(beta_choice) F(mnl_choice) F(p_car) F(p_bike);
#undef F

    py::class_<DriveStats>(m, "DriveStats")
        .def_readonly("events", &DriveStats::events)
        .def_readonly("agents", &DriveStats::agents)
        .def_readonly("finished", &DriveStats::finished)
        .def_readonly("jammed_in_links", &DriveStats::jammed_in_links)
        .def_readonly("jammed_in_requests", &DriveStats::jammed_in_requests)
        .def_readonly("stuck_moves", &DriveStats::stuck_moves)
        .def("__repr__", [](const DriveStats& s) {
            return "DriveStats(events=" + std::to_string(s.events) + ", agents=" + std::to_string(s.agents) +
                   ", finished=" + std::to_string(s.finished) + ", jammed_in_links=" + std::to_string(s.jammed_in_links) +
                   ", jammed_in_requests=" + std::to_string(s.jammed_in_requests) +
                   ", stuck_moves=" + std::to_string(s.stuck_moves) + ")";
        });

    py::class_<PublicStats>(m, "PublicStats")
        .def_readonly("events", &PublicStats::events)
        .def_readonly("agents", &PublicStats::agents)
        .def_readonly("finished", &PublicStats::finished)
        .def("__repr__", [](const PublicStats& s) {
            return "PublicStats(events=" + std::to_string(s.events) + ", agents=" + std::to_string(s.agents) +
                   ", finished=" + std::to_string(s.finished) + ")";
        });

    using release = py::call_guard<py::gil_scoped_release>;
    py::class_<Model>(m, "Model")
        .def(py::init<Config>(), py::arg("config"))
        .def("load", &Model::load, release(), "Load networks, agenda, ownership and access candidates")
        .def("plan_bike", &Model::plan_bike, release(), "Plan bike routes (once; no interaction)")
        .def("start_iteration", &Model::start_iteration, release())
        .def("plan", &Model::plan, release(), "Plan public and drive routes for this iteration")
        .def("choose", &Model::choose, release(), "MNL (iteration 0) / MCMC mode choice")
        .def("prepare_execution", &Model::prepare_execution, release())
        .def("simulate_public", &Model::simulate_public, release())
        .def("simulate_drive", &Model::simulate_drive, release())
        .def_property_readonly("progress_agents", &Model::progress_agents,
                               "agents started in the running plan()/plan_bike() (progress bar)")
        .def_property_readonly("progress_clock", &Model::progress_clock,
                               "simulated clock (hours) of the running public/drive simulation (progress bar)")
        .def("update_memory", &Model::update_memory, release())
        .def("check_invariants", &Model::check_invariants, release(),
             "True if every agent finished and all vehicle/road capacities were restored")
        .def("write_iteration", &Model::write_iteration, release())
        .def("write_final", &Model::write_final, release())
        .def("run", &Model::run, release(), "Whole cycle (load ... write_final)")
        .def_readonly("config", &Model::cfg)
        .def_readonly("n_agents", &Model::N)
        .def_readonly("capacity_scale", &Model::capacity_scale)
        .def_property_readonly("population_size", [](const Model& s) { return s.pop.population_size; })
        .def_property_readonly("population_rows", [](const Model& s) { return s.pop.source_row; })
        // Results, copied into Python objects
        .def_property_readonly("bike_scores", [](const Model& s) { return to_numpy(s.bike.scores); })
        .def_property_readonly("public_scores", [](const Model& s) { return to_numpy(s.pub.scores); })
        .def_property_readonly("drive_scores", [](const Model& s) { return to_numpy(s.drive.scores); })
        .def_property_readonly("exec_bike", [](const Model& s) { return to_numpy(s.exec_bike); })
        .def_property_readonly("exec_public", [](const Model& s) { return to_numpy(s.exec_public); })
        .def_property_readonly("exec_drive", [](const Model& s) { return to_numpy(s.exec_drive); })
        .def_property_readonly("finished_exec", [](const Model& s) { return s.finished_exec; })
        .def_property_readonly("drive_paths", [](const Model& s) { return s.drive.paths; })
        .def_property_readonly("public_paths", [](const Model& s) { return s.pub.paths; })
        .def_property_readonly("drive_trajectories", [](const Model& s) { return py::make_tuple(s.traj_drive.nodes, s.traj_drive.times); })
        .def_property_readonly("public_trajectories", [](const Model& s) { return py::make_tuple(s.traj_public.nodes, s.traj_public.times); })
        // Per-trip results of the last iteration: mode code 0 walk, 1 car, 2 pt, 3 bike
        .def("trip_modes", &Model::trip_modes)
        .def("trip_durations", &Model::trip_durations)
        .def("trip_departures", &Model::trip_departures)
        .def("trip_distances_km", &Model::trip_distances_km)
        // Planned whole-day times (hours) and executed scores
        .def_property_readonly("bike_times", [](const Model& s) { return to_numpy(s.bike.times); })
        .def_property_readonly("public_times", [](const Model& s) { return to_numpy(s.pub.times); })
        .def_property_readonly("drive_times", [](const Model& s) { return to_numpy(s.drive.times); })
        .def("executed_scores", [](const Model& s) { return to_numpy(s.executed_scores()); })
        .def("drive_link_traversals", [](const Model& s) {
            auto t = s.drive_link_traversals();
            py::dict d;
            d["agent"] = py::array_t<int>(t.agent.size(), t.agent.data());
            d["trip"] = py::array_t<int>(t.trip.size(), t.trip.data());
            d["u"] = py::array_t<int>(t.u.size(), t.u.data());
            d["av"] = py::array_t<int>(t.av.size(), t.av.data());
            d["t_enter"] = to_numpy(t.t_enter);
            d["t_end"] = to_numpy(t.t_end);
            d["t_exit"] = to_numpy(t.t_exit);
            return d;
        });
}
