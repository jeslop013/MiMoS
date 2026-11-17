"""Terminal progress bar for `mimos run` (disable with --no-progress).

The long C++ stages release the GIL, so a background thread can poll the model's counters
twice a second while they run: agents started in plan()/plan_bike() (``Model.progress_agents``)
and the simulated clock of simulate_public()/simulate_drive() (``Model.progress_clock``, hours;
the bar spans the day 00:00-24:00, agendas that run past midnight show 100%). One line on
stderr is rewritten in place and cleared when the stage ends, so the per-iteration summary
lines stay as they were. Only drawn when stderr is a terminal (not in logs or pipes).

    iteration 3/30  drive   [##########----------]  13:40 simulated | 4m12s | run left ~6h05m
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager

WIDTH = 24  # characters of the bar


def _hms(seconds: float) -> str:
    s = int(max(seconds, 0))
    if s >= 3600:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def _clock(h: float) -> str:
    return f"{int(h):02d}:{int(round(h % 1 * 60)) % 60:02d}"


class Progress:
    """Progress of a run: ``with progress.stage("drive", it): model.simulate_drive(it)``."""

    def __init__(self, model, n_iterations: int, enabled: bool = True):
        self.model, self.n_it = model, n_iterations
        self.enabled = enabled and sys.stderr.isatty()
        self.it_seconds: list[float] = []  # completed iterations
        self.it_start = time.perf_counter()

    def start_iteration(self) -> None:
        self.it_start = time.perf_counter()

    def end_iteration(self) -> None:
        self.it_seconds.append(time.perf_counter() - self.it_start)

    def _left(self, it: int | None) -> str:
        """Time left in the run, from the mean of the completed iterations."""
        if it is None or not self.it_seconds:
            return ""
        mean = sum(self.it_seconds) / len(self.it_seconds)
        left = (self.n_it - it - 1) * mean + max(0.0, mean - (time.perf_counter() - self.it_start))
        return f" | run left ~{_hms(left)}"

    def _line(self, name: str, it: int | None, t0: float) -> str:
        head = f"iteration {it + 1}/{self.n_it}  " if it is not None else ""
        if name in ("plan", "bike"):
            n = max(self.model.n_agents, 1)
            frac = min(self.model.progress_agents / n, 1.0)
            what = f"{min(self.model.progress_agents, n):,}/{n:,} agents"
        elif name in ("public", "drive"):
            h = self.model.progress_clock
            frac = min(max(h / 24.0, 0.0), 1.0)
            what = f"{_clock(h)} simulated"
        else:  # no counter (loading): elapsed time only
            frac, what = None, ""
        bar = "" if frac is None else "[" + "#" * round(frac * WIDTH) + "-" * (WIDTH - round(frac * WIDTH)) + "]  "
        return f"{head}{name:<7} {bar}{what} | {_hms(time.perf_counter() - t0)}{self._left(it)}"

    def _write(self, text: str) -> None:
        try:
            sys.stderr.write("\r" + text.ljust(100)[:120] + ("\r" if not text else ""))
            sys.stderr.flush()
        except (OSError, ValueError):  # terminal gone: stop drawing, never stop the run
            self.enabled = False

    @contextmanager
    def stage(self, name: str, it: int | None = None):
        if not self.enabled:
            yield
            return
        stop = threading.Event()
        t0 = time.perf_counter()

        def loop() -> None:
            while not stop.wait(0.5):
                self._write(self._line(name, it, t0))

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
            self._write("")  # clear the line
