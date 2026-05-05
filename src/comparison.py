"""Algorithm and scenario comparison for flow simulations.

This module provides comparison utilities:
- compare_schemes(): Compare different numerical schemes (upwind vs TVD vs Rusanov)
- compare_scenarios(): Compare different simulation scenarios (1D vs 2D-SPE10 vs 2D-synthetic)
"""

import jax.numpy as jnp
from typing import Dict, List, Literal, Optional, Any
import os

from .properties import FlowParams
from .saturation_solver import simulate_1d_buckley_leverett
from .solver import SimulationConfig2D, IMPESSolver2D


def compare_schemes(
    schemes: List[Literal["upwind", "tvd", "rusanov"]],
    params: Optional[FlowParams] = None,
    nx: int = 100,
    L: float = 100.0,
    dt: float = 0.0005,
    t_max: float = 0.05,
    u: float = 1e-4,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Compare different numerical schemes on 1D Buckley-Leverett problem.

    Runs each scheme with identical parameters and returns results for comparison.

    Args:
        schemes: List of scheme names to compare
                 e.g., ["upwind", "tvd", "rusanov"]
        params: Flow parameters (default: FlowParams())
        nx: Number of grid cells (default: 100)
        L: Domain length in meters (default: 100.0)
        dt: Time step (default: 0.0005)
        t_max: Maximum simulation time (default: 0.05)
        u: Darcy velocity (default: 1e-4)

    Returns:
        Dict mapping scheme name to results dict with keys:
        - Sw_history: Saturation history (n_times+1, nx)
        - time_array: Time at each saved step
        - x: Grid cell centers (nx,)
    """
    if params is None:
        params = FlowParams()

    dx = L / nx
    nt = int(t_max / dt)

    Sw_init = jnp.full(nx, params.Swc + params.Sor)

    n_times = nt + 1
    time_array = jnp.linspace(0, t_max, n_times)

    x_array = jnp.linspace(0, L, nx) + 0.5 * dx

    results = {}

    for scheme in schemes:
        Sw_history = simulate_1d_buckley_leverett(
            Sw_init=Sw_init,
            u=u,
            params=params,
            dt=dt,
            dx=dx,
            nt=nt,
            scheme=scheme,
            Sw_inj=1.0 - params.Sor,
            bc_type="dirichlet",
        )

        results[scheme] = {
            "Sw_history": Sw_history,
            "time_array": time_array,
            "x": x_array,
            "t_max": t_max,
        }

    return results


def compare_scenarios(
    scenarios: List[Literal["1d", "2d-spe10", "2d-synthetic"]],
    params: Optional[FlowParams] = None,
    t_max: float = 0.005,
    dt: float = 0.001,
) -> Dict[str, Dict[str, Any]]:
    """Compare different simulation scenarios.

    Runs 1D Buckley-Leverett and various 2D configurations.

    Args:
        scenarios: List of scenario names to compare
                   e.g., ["1d", "2d-spe10", "2d-synthetic"]
        params: Flow parameters (default: FlowParams())
        t_max: Maximum simulation time (default: 0.005)
        dt: Time step (default: 0.001)

    Returns:
        Dict mapping scenario name to results dict with keys:
        - For all scenarios: Sw_history, time_array, nx, ny
        - For 1D: x
        - For 2D: x, y
        - On error: status="error", message="..."

    Note:
        2D scenarios may return error status if data files are unavailable.
    """
    if params is None:
        params = FlowParams()

    results = {}

    if "1d" in scenarios:
        try:
            nx = 100
            L = 100.0
            dx = L / nx
            nt = int(t_max / dt)

            Sw_init_1d = jnp.full(nx, params.Swc)
            x_1d = jnp.linspace(0, L, nx) + 0.5 * dx
            time_array = jnp.linspace(0, t_max, nt + 1)

            Sw_history_1d = simulate_1d_buckley_leverett(
                Sw_init=Sw_init_1d,
                u=1e-4,
                params=params,
                dt=dt,
                dx=dx,
                nt=nt,
                scheme="upwind",
                Sw_inj=1.0 - params.Sor,
            )

            results["1d"] = {
                "Sw_history": Sw_history_1d,
                "time_array": time_array,
                "x": x_1d,
                "y": None,
                "nx": nx,
                "ny": 1,
            }
        except Exception as e:
            results["1d"] = {
                "status": "error",
                "message": f"1D simulation failed: {str(e)}",
            }

    if "2d-spe10" in scenarios:
        try:
            data_dir = "data"
            spe10_file = os.path.join(data_dir, "perm_case1.dat")

            if not os.path.exists(spe10_file):
                results["2d-spe10"] = {
                    "status": "error",
                    "message": f"SPE10 data not found at {spe10_file}",
                }
            else:
                from .benchmarks import load_spe10_case1

                spe10_data = load_spe10_case1(data_dir)

                nx = 100
                ny = 20
                Lx = 100.0
                Ly = 20.0

                perm_x = spe10_data["perm_x"].T
                perm_y = spe10_data["perm_y"].T

                config = SimulationConfig2D(
                    nx=nx,
                    ny=ny,
                    Lx=Lx,
                    Ly=Ly,
                    dt=dt,
                    t_max=t_max,
                    q_injection=1e-4,
                    save_interval=10,
                )

                solver = IMPESSolver2D(
                    params=params,
                    config=config,
                    perm_x=perm_x,
                    perm_y=perm_y,
                )

                Sw_init_2d = jnp.full((ny, nx), params.Swc)

                time_array, Sw_history_2d, _ = solver.run(Sw_init_2d)

                results["2d-spe10"] = {
                    "Sw_history": Sw_history_2d,
                    "time_array": time_array,
                    "x": solver.x,
                    "y": solver.y,
                    "nx": nx,
                    "ny": ny,
                }
        except FileNotFoundError as e:
            results["2d-spe10"] = {
                "status": "error",
                "message": f"SPE10 data file not found: {str(e)}",
            }
        except Exception as e:
            results["2d-spe10"] = {
                "status": "error",
                "message": f"2D SPE10 simulation failed: {str(e)}",
            }

    if "2d-synthetic" in scenarios:
        try:
            nx = 50
            ny = 30
            Lx = 50.0
            Ly = 30.0

            perm_x = jnp.ones((ny, nx)) * 1e-12
            perm_y = jnp.ones((ny, nx)) * 1e-12

            config = SimulationConfig2D(
                nx=nx,
                ny=ny,
                Lx=Lx,
                Ly=Ly,
                dt=dt,
                t_max=t_max,
                q_injection=1e-4,
                save_interval=10,
            )

            solver = IMPESSolver2D(
                params=params,
                config=config,
                perm_x=perm_x,
                perm_y=perm_y,
            )

            Sw_init_2d = jnp.full((ny, nx), params.Swc)

            time_array, Sw_history_2d, _ = solver.run(Sw_init_2d)

            results["2d-synthetic"] = {
                "Sw_history": Sw_history_2d,
                "time_array": time_array,
                "x": solver.x,
                "y": solver.y,
                "nx": nx,
                "ny": ny,
            }
        except Exception as e:
            results["2d-synthetic"] = {
                "status": "error",
                "message": f"2D synthetic simulation failed: {str(e)}",
            }

    return results


def compute_scheme_differences(
    schemes_results: Dict[str, Dict[str, jnp.ndarray]],
    time_index: int = -1,
) -> Dict[str, float]:
    """Compute difference metrics between schemes at a given time.

    Args:
        schemes_results: Output from compare_schemes()
        time_index: Which time step to compare (default: last)

    Returns:
        Dict with difference metrics between schemes
    """
    scheme_names = list(schemes_results.keys())
    if len(scheme_names) < 2:
        return {}

    metrics = {}

    ref_scheme = scheme_names[0]
    ref_saturation = schemes_results[ref_scheme]["Sw_history"][time_index]

    for scheme in scheme_names[1:]:
        comp_saturation = schemes_results[scheme]["Sw_history"][time_index]

        diff = jnp.abs(comp_saturation - ref_saturation)
        max_diff = float(jnp.max(diff))
        mean_diff = float(jnp.mean(diff))

        metrics[f"{ref_scheme}_vs_{scheme}_max_diff"] = max_diff
        metrics[f"{ref_scheme}_vs_{scheme}_mean_diff"] = mean_diff

    return metrics
