"""The HTML report builds from a toy run and contains every section (plots are not inspected)."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from mimos import config_from_dict, run
from mimos.viz.report import build_report
from scenario import write_corridor

TARGETS = Path(__file__).resolve().parent / "fixtures" / "targets_demo.toml"


def test_report_from_corridor_run(tmp_path):
    n = 12
    data = write_corridor(tmp_path / "data", n_agents=n)
    # half the agents can only drive well, the rest take the bus: both road and pt sections fill
    cfg = config_from_dict({"data_dir": str(data), "run_dir": str(tmp_path / "out"), "log_level": "WARNING",
                            "it_number": 2, "n_agents": n, "C_drive": -0.5, "C_public": -0.5,
                            "beta_trav_walk": -60.0})
    model, _ = run(cfg, verbose=False)
    links = np.load(tmp_path / "out" / "drive_links.npz")
    assert len(links["u"]) == 2 * 3 * int(np.asarray(model.exec_drive).sum())  # 2 trips x 3 links per driver
    # moving part of each traversal = free-flow time of the corridor links (1 minute)
    assert np.allclose((links["t_end"] - links["t_enter"]) * 60, 1.0, atol=1e-6)

    path = build_report(tmp_path / "out", targets=TARGETS)
    page = path.read_text(encoding="utf-8")
    assert re.findall(r"<h2 id='(\w+)'>", page) == ["run", "relaxation", "trips", "roads", "public", "emissions"]
    assert page.count("Plotly.newPlot") >= 6
    assert "No emissions computed" in page
    offline = build_report(tmp_path / "out", targets=TARGETS, out=tmp_path / "offline.html", tiles=False)
    # tile maps have a map centre in their layout; offline maps have lon/lat axes instead
    off = offline.read_text(encoding="utf-8")
    assert re.search(r'"center":\{"l(at|on)"', page) and '"text":"longitude"' not in page
    assert '"text":"longitude"' in off and not re.search(r'"center":\{"l(at|on)"', off)


def test_png_export_and_videos(tmp_path):
    """PNG export (needs kaleido + Chrome/Edge) and the MP4 videos, on a short toy run."""
    import pytest

    from mimos.emissions import run_emissions
    from mimos.viz.video import make_videos

    n = 8
    data = write_corridor(tmp_path / "data", n_agents=n)
    cfg = config_from_dict({"data_dir": str(data), "run_dir": str(tmp_path / "out"), "log_level": "WARNING",
                            "it_number": 1, "n_agents": n, "C_drive": 0.0, "beta_trav_walk": -60.0})
    run(cfg, verbose=False)
    run_emissions(tmp_path / "out")
    videos = make_videos(tmp_path / "out", start=7.9, end=8.3, speed=600, fps=10)
    assert [p.name for p in videos] == ["vehicles.mp4", "traffic.mp4", "emissions_NOx.mp4"]
    assert all(p.stat().st_size > 1000 for p in videos)

    pytest.importorskip("kaleido")
    # kaleido drives a headless Chrome/Edge. Without a usable browser it can raise or hang
    # (seen on CI machines), so the export runs in a child process with a time limit.
    code = ("import sys; from pathlib import Path; from mimos.viz.report import build_report; "
            "build_report(Path(sys.argv[1]), targets=Path(sys.argv[2]), png=True, tiles=False)")
    try:
        res = subprocess.run([sys.executable, "-c", code, str(tmp_path / "out"), str(TARGETS)],
                             capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        pytest.skip("static export unavailable: kaleido did not finish within 180 s")
    if res.returncode != 0:  # no Chrome/Edge available for kaleido
        pytest.skip(f"static export unavailable: {res.stderr.strip().splitlines()[-1:]}")
    pngs = sorted((tmp_path / "out" / "report" / "figures").glob("*.png"))
    assert len(pngs) >= 8 and any("roads_map" in p.name for p in pngs)
