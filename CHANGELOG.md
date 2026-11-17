# Changelog

## [0.1.0] - 2026-11-17: first public release
- First public version, under the GPL-3.0-or-later licence (`LICENSE`).
- Agent-based model of urban mobility with a C++17 core, a Python API and a command line (`mimos`).
- Public transport, car and bicycle simulation; routing, mode choice and learning until a stochastic user equilibrium.
- Hot exhaust emissions per link and hour, HTML reports, videos, and calibration of the mode constants to a survey (`mimos calibrate`).
- A made-up example town that needs no data (`mimos demo`): organic streets, a ring road, a river with bridges and bus lines on the roads. A demo gallery in `docs/gallery/`.
- `CITATION.cff` and a GitHub Actions workflow that runs the tests on Windows, Linux and macOS.
