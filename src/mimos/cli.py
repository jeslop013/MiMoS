"""Command line interface.

    uv run mimos run <config.toml> [options]           # run the model
    uv run mimos calibrate <config.toml> <targets.toml>    # fit the mode constants to a survey
    uv run mimos report <run_dir> [--targets FILE] [--png] [--serve] [--no-tiles]   # HTML report
    uv run mimos emissions <run_dir> [--seed N]       # hot exhaust emissions of a run
    uv run mimos video <run_dir> [--what vehicles traffic emissions]   # MP4 videos of a run
"""

from __future__ import annotations

import argparse
import sys

from mimos.config import load_config, load_view
from mimos.model import run


def _serve(path) -> None:
    """Serves the report's folder on localhost and opens it in the browser (Ctrl+C to stop)."""
    import functools
    import http.server
    import webbrowser
    from pathlib import Path

    path = Path(path).resolve()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(path.parent))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/{path.name}"
        print(f"serving {url} (Ctrl+C to stop)")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def _view_args(p) -> None:
    p.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"),
                   help="map centre (default: the [view] saved with the run, e.g. Plaza Mayor)")
    p.add_argument("--radius-km", type=float, help="half-height of the map view in km (default: [view] or 6)")
    p.add_argument("--view-from", help="take [view] from this TOML file (e.g. the run's scenario config)")


def _view_from(args) -> dict | None:
    view = dict(load_view(args.view_from) or {}) if args.view_from else {}
    if args.center:
        view["center_lat"], view["center_lon"] = args.center
    if args.radius_km:
        view["radius_km"] = args.radius_km
    return view or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mimos", description="MiMoS agent-based transport model")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the iterative cycle with a TOML configuration")
    p_run.add_argument("config", help="TOML configuration file")
    p_run.add_argument("--n-agents", type=int, help="override [run] n_agents")
    p_run.add_argument("--iterations", type=int, dest="it_number", help="override [run] it_number")
    p_run.add_argument("--run-dir", "--results-dir", dest="run_dir",
                       help="override [run] run_dir (folder of this run's outputs; --results-dir: old name)")
    p_run.add_argument("--data-dir", help="override [run] data_dir")
    p_run.add_argument("--log-level", help="TRACE, DEBUG, INFO, WARNING, ERROR or CRITICAL")
    p_run.add_argument("--seed", type=int)
    p_run.add_argument("--legacy", action="store_true", default=None, dest="legacy_mode",
                       help="reproduce the model version of the thesis (legacy_mode = true)")
    p_run.add_argument("--print-it", type=int, help="override [run] print_it (per-iteration CSVs every N iterations)")
    p_run.add_argument("--targets", help="survey targets file: adds per-trip shares within its max_trip_km to iterations.csv")
    p_run.add_argument("--no-progress", action="store_true",
                       help="no progress bar (it is only drawn on a terminal anyway, not in logs or pipes)")

    p_rep = sub.add_parser("report", help="write <run_dir>/report/report.html (needs: uv sync --extra viz)")
    p_rep.add_argument("run_dir")
    p_rep.add_argument("--targets", help="survey targets TOML (adds the survey comparison)")
    p_rep.add_argument("--out", help="output file (default <run_dir>/report/report.html)")
    p_rep.add_argument("--serve", action="store_true",
                       help="open the report from a local web server (http://localhost) instead of a file: "
                            "browsers that block map tiles on file:// pages load them this way")
    p_rep.add_argument("--png", action="store_true",
                       help="also write every figure as a PNG (2x) to <run_dir>/report/figures/")
    p_rep.add_argument("--png-size", nargs=2, type=int, default=(1920, 1080), metavar=("W", "H"),
                       help="PNG size in pixels before the 2x scale (default 1920 1080)")
    p_rep.add_argument("--png-font", type=int, default=16, help="PNG font size (default 16)")
    _view_args(p_rep)
    p_rep.add_argument("--no-tiles", action="store_true",
                       help="draw maps on the model's road network instead of online map tiles: works when the file is "
                            "opened directly (double-click), so use it for reports sent to others")

    p_emi = sub.add_parser("emissions", help="hot exhaust emissions of a run -> <run_dir>/emissions/")
    p_emi.add_argument("run_dir")
    p_emi.add_argument("--seed", type=int, default=0, help="seed of the vehicle draw (default 0)")
    p_emi.add_argument("--fleet", help="fleet CSV (default: <data_dir>/fleet.csv of the run's scenario, else the packaged example fleet)")
    p_emi.add_argument("--factors", help="hot emission factors CSV (default: the packaged EEA factors, mimos/data/hot_ef.csv)")

    p_vid = sub.add_parser("video", help="MP4 videos of a run -> <run_dir>/videos/ (needs: uv sync --extra viz)")
    p_vid.add_argument("run_dir")
    p_vid.add_argument("--what", nargs="+", default=["vehicles", "traffic", "emissions"],
                       choices=["vehicles", "traffic", "emissions"])
    p_vid.add_argument("--start", default="05:00", help="start, clock time HH:MM of the simulated day (default 05:00)")
    p_vid.add_argument("--end", default="24:00", help="end, HH:MM (default 24:00; past midnight e.g. 25:30)")
    p_vid.add_argument("--speed", type=float,
                       help="simulated seconds per second of video: 1 = real time, 10 = ten times faster, "
                            "60 = a minute per second (default: 1800 vehicles, 9000 traffic/emissions)")
    p_vid.add_argument("--fps", type=int, default=30, help="frames per second of the video (default 30)")
    p_vid.add_argument("--tag", default="", help="suffix for the file names, e.g. _0800_slow")
    _view_args(p_vid)

    p_cal = sub.add_parser("calibrate", help="fit the mode constants to a survey's per-trip modal split")
    p_cal.add_argument("config", help="TOML configuration file")
    p_cal.add_argument("targets", help="survey targets TOML (shares, scope, durations)")
    p_cal.add_argument("--n-agents", type=int)
    p_cal.add_argument("--iterations", type=int, default=3, help="iterations per calibration round (default 3)")
    p_cal.add_argument("--rounds", type=int, default=12)
    p_cal.add_argument("--tol", type=float, default=0.01, help="stop when every share is within tol (default 0.01)")
    p_cal.add_argument("--eta", type=float, default=0.7, help="damping of the update (default 0.7)")
    p_cal.add_argument("--max-step", type=float, default=2.0, help="largest change of a constant per round")
    p_cal.add_argument("--final-iterations", type=int, default=10, help="validation run (0 = skip)")
    p_cal.add_argument("--run-dir", help="output folder (default: 'calibration' next to the config's run_dir)")
    p_cal.add_argument("--start", nargs="*", default=[], metavar="NAME=VALUE",
                       help="starting constants instead of the config's, e.g. C_drive=-1.7")

    for name, help_text in (("example", "write the synthetic example town (data + config.toml), no external data needed"),
                            ("demo", "example town -> run -> emissions -> report (-> video): try MiMoS in ~1 minute")):
        p_ex = sub.add_parser(name, help=help_text)
        p_ex.add_argument("--out", default="examples/town", help="output folder (default examples/town)")
        p_ex.add_argument("--agents", type=int, default=5000, help="simulated agents (default 5000)")
        p_ex.add_argument("--scale", type=float, default=0.1,
                          help="sample share they represent (default 0.1: a town of 10 x agents; 1 = everybody)")
        p_ex.add_argument("--iterations", type=int, default=10, help="iterations in the config (default 10)")
        p_ex.add_argument("--seed", type=int, default=0, help="seed of the generated town (default 0)")
        if name == "demo":
            p_ex.add_argument("--video", action="store_true", help="also render a vehicles video of the morning peak")

    args = parser.parse_args(argv)
    if args.command in ("example", "demo"):
        from mimos.scenarios import town

        config = town(args.out, n_agents=args.agents, seed=args.seed, capacity_scale=args.scale,
                      iterations=args.iterations)
        if args.command == "example":
            print(f"example town written: {config}\nnext:\n  uv run mimos run {config.as_posix()}\n"
                  f"  uv run mimos report {(config.parent / 'run').as_posix()} --no-tiles")
            return 0
        from mimos.emissions import run_emissions
        from mimos.viz.report import build_report

        run(load_config(config), view=load_view(config))
        results = config.parent / "run"
        print(run_emissions(results))
        print(build_report(results, tiles=False))  # a made-up town: draw its own roads, no map tiles
        if args.video:
            from mimos.viz.video import make_videos

            for path in make_videos(results, ("vehicles",), 7.5, 8.5, speed=60, fps=30):
                print(path)
        print(f"open {(results / 'report' / 'report.html').as_posix()} in a browser")
        return 0
    if args.command == "calibrate":
        from mimos.calibration import calibrate

        print(calibrate(args.config, args.targets, n_agents=args.n_agents, iterations=args.iterations,
                        rounds=args.rounds, tol=args.tol, eta=args.eta, max_step=args.max_step,
                        final_iterations=args.final_iterations, run_dir=args.run_dir, start=args.start))
        return 0
    if args.command == "video":
        from mimos.viz.video import make_videos, parse_clock

        for path in make_videos(args.run_dir, args.what, parse_clock(args.start), parse_clock(args.end),
                                args.speed, args.fps, view=_view_from(args), tag=args.tag):
            print(path)
        return 0
    if args.command == "emissions":
        from pathlib import Path

        from mimos.emissions import Settings, run_emissions

        s = Settings(seed=args.seed)
        if args.fleet:
            s.fleet = Path(args.fleet)
        if args.factors:
            s.factors = Path(args.factors)
        print(run_emissions(args.run_dir, s))
        return 0
    if args.command == "report":
        from mimos.viz.report import build_report  # plotly/pandas are optional dependencies

        path = build_report(args.run_dir, args.targets, args.out, tiles=not args.no_tiles, png=args.png,
                            view=_view_from(args), png_size=tuple(args.png_size), png_font=args.png_font)
        print(path)
        if args.serve:
            _serve(path)
        return 0
    if args.command == "run":
        overrides = {
            "n_agents": args.n_agents,
            "it_number": args.it_number,
            "run_dir": args.run_dir,
            "data_dir": args.data_dir,
            "log_level": args.log_level,
            "seed": args.seed,
            "legacy_mode": args.legacy_mode,
            "print_it": args.print_it,
        }
        cfg = load_config(args.config, **overrides)
        max_km = None
        if args.targets:
            from mimos.analysis import load_targets

            max_km = load_targets(args.targets)["scope"].get("max_trip_km")
        _, report = run(cfg, view=load_view(args.config), max_km=max_km, progress=not args.no_progress)
        return 0 if all(r.invariants_ok for r in report.iterations) else 3
    return 1


if __name__ == "__main__":
    sys.exit(main())
