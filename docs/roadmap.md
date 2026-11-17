# Roadmap

The main directions for the next versions of MiMoS. The order inside each group is not a schedule. Released changes are listed in the [CHANGELOG](../CHANGELOG.md).

## Travel behaviour

- **Mode choice per trip and per tour**, with the car as a vehicle that must come back home.
- **Departure-time choice** and **route choice** among several routes.
- **Car and bike ownership** and transport passes from the attributes of the population.

## Networks and modes

- **Public transport from GTFS timetables**, with a fast timetable router.
- **Traffic signals** at intersections, and **buses in road traffic**.
- **A walking network** for walk trips and access to stops.
- **Taxi and ride-hailing** as a new mode.
- **Traffic from outside the area**: visitors, commuters from outside and freight.

## Policy scenarios and emissions

- **Low-emission zones**: access rules by vehicle class, and the response of the agents.
- **More emission sources**: cold starts and non-exhaust particles.
- **Scenario comparison**: maps and curves of the differences between two runs.

## Usability

- **Importers for OpenStreetMap and GTFS**, and a checker for the input files.
- **Faster simulation** and loading for large cities.
- **A Python package on PyPI** with prebuilt binaries, so that no C++ compiler is needed.
- **Overview reports of the inputs** of a scenario, before a run.
