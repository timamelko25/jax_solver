"""Performance benchmarking utilities for flow simulators.

Provides tools to measure wall-clock time vs dt and number of steps.
"""

import time
import statistics
from typing import List, Dict

import jax.numpy as jnp

from .properties import FlowParams
from .saturation_solver import simulate_1d_buckley_leverett


def _run_simulation(dt: float, nx: int, t_max: float) -> float:
    """Run a single 1D Buckley-Leverett simulation and return wall time.

    Args:
        dt: Time step size
        nx: Number of grid cells
        t_max: Total simulation time

    Returns:
        Wall-clock time in seconds
    """
    params = FlowParams()
    L = 1.0
    dx = L / nx
    u = 1.0
    Sw_init = jnp.full(nx, params.Swc)
    nt = int(t_max / dt)

    # JIT warmup: run once
    _ = simulate_1d_buckley_leverett(
        Sw_init, u, params, dt, dx, nt, scheme="upwind", Sw_inj=1.0
    )

    start = time.perf_counter()
    _ = simulate_1d_buckley_leverett(
        Sw_init, u, params, dt, dx, nt, scheme="upwind", Sw_inj=1.0
    )
    end = time.perf_counter()

    return end - start


def benchmark_time_vs_dt(
    dt_values: List[float], nx: int = 100, t_max: float = 0.01
) -> List[Dict]:
    """Benchmark wall-clock time as a function of dt.

    For each dt value, runs simulation with that dt (same nx and t_max).
    Uses warmup (1 run) + 3 timed runs, takes median time.

    Args:
        dt_values: List of dt values to benchmark (e.g. [0.001, 0.0005, 0.0001])
        nx: Number of grid cells (default: 100)
        t_max: Total simulation time (default: 0.01)

    Returns:
        List of dicts [{'dt': dt, 'wall_time': time, 'n_steps': n_steps}, ...]
    """
    results = []

    for dt in dt_values:
        n_steps = int(t_max / dt)

        # Warmup run
        _run_simulation(dt, nx, t_max)

        # 3 timed runs
        times = []
        for _ in range(3):
            elapsed = _run_simulation(dt, nx, t_max)
            times.append(elapsed)

        wall_time = statistics.median(times)

        results.append(
            {
                "dt": dt,
                "wall_time": wall_time,
                "n_steps": n_steps,
            }
        )

    return results


def benchmark_time_vs_nsteps(
    n_steps_list: List[int], nx: int = 100, dt: float = 0.0001
) -> List[Dict]:
    """Benchmark wall-clock time as a function of number of steps.

    For each n_steps value, runs simulation with that t_max (same dt).
    Uses warmup (1 run) + 3 timed runs, takes median time.

    Args:
        n_steps_list: List of number of steps to benchmark (e.g. [50, 100, 200, 400])
        nx: Number of grid cells (default: 100)
        dt: Fixed time step size (default: 0.0001)

    Returns:
        List of dicts [{'n_steps': n_steps, 'wall_time': time, 't_max': t_max}, ...]
    """
    results = []

    for n_steps in n_steps_list:
        t_max = n_steps * dt

        # Warmup run
        _run_simulation(dt, nx, t_max)

        # 3 timed runs
        times = []
        for _ in range(3):
            elapsed = _run_simulation(dt, nx, t_max)
            times.append(elapsed)

        wall_time = statistics.median(times)

        results.append(
            {
                "n_steps": n_steps,
                "wall_time": wall_time,
                "t_max": t_max,
            }
        )

    return results
