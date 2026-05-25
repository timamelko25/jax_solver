"""Convergence analysis for Buckley-Leverett solver."""

import numpy as onp
import jax.numpy as jnp

from .properties import FlowParams, welge_construct
from .saturation_solver import simulate_1d_buckley_leverett


def welge_analytical_solution(
    params: FlowParams,
    L: float,
    time_array: jnp.ndarray,
    nx: int = 2000,
    Sw_inj: float = 1.0,
    Sw_init: float = None,
    u: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if Sw_init is None:
        Sw_init = params.Swc

    S_shock, v_shock = welge_construct(Sw_inj, Sw_init, params)

    v_actual = u * v_shock

    dx = L / nx
    x_array = jnp.linspace(dx / 2, L - dx / 2, nx)

    n_times = len(time_array)
    Sw_history_analytical = jnp.zeros((n_times, nx))

    for i, t in enumerate(time_array):
        x_shock = jnp.clip(v_actual * t, 0.0, L)
        Sw_profile = jnp.where(x_array < x_shock, Sw_inj, Sw_init)
        Sw_history_analytical = Sw_history_analytical.at[i, :].set(Sw_profile)

    return x_array, Sw_history_analytical


def compute_l1_error(Sw_num: jnp.ndarray, Sw_ana: jnp.ndarray) -> float:
    return float(onp.mean(onp.abs(onp.array(Sw_num) - onp.array(Sw_ana))))


def compute_l2_error(Sw_num: jnp.ndarray, Sw_ana: jnp.ndarray) -> float:
    return float(onp.sqrt(onp.mean((onp.array(Sw_num) - onp.array(Sw_ana)) ** 2)))


def compute_max_error(Sw_num: jnp.ndarray, Sw_ana: jnp.ndarray) -> float:
    return float(onp.max(onp.abs(onp.array(Sw_num) - onp.array(Sw_ana))))


def run_grid_refinement_study(
    params: FlowParams = None,
    base_nx: int = 50,
    refine_factors: list = None,
    t_compare: float = 0.03,
    L: float = 100.0,
    base_dt: float = 0.001,
    t_max: float = 0.05,
    Sw_inj: float = 1.0,
    Sw_init: float = None,
    scheme: str = "upwind",
) -> list:
    if params is None:
        params = FlowParams()
    if refine_factors is None:
        refine_factors = [2, 4]
    if Sw_init is None:
        Sw_init = params.Swc

    results = []

    u = 1.0

    for factor in refine_factors:
        nx = base_nx * factor
        dx = L / nx
        dt = base_dt / factor

        Sw_init_arr = jnp.full(nx, Sw_init)

        nt = int(t_max / dt)

        Sw_history = simulate_1d_buckley_leverett(
            Sw_init_arr,
            u,
            params,
            dt,
            dx,
            nt,
            scheme=scheme,
            Sw_inj=Sw_inj,
        )

        n_steps = int(t_compare / dt)
        if n_steps > nt:
            n_steps = nt
        Sw_num = onp.array(Sw_history[n_steps, :])

        time_array = jnp.array([t_compare])
        x_ana, Sw_ana = welge_analytical_solution(
            params,
            L,
            time_array,
            nx=nx,
            Sw_inj=Sw_inj,
            Sw_init=Sw_init,
            u=u,
        )
        Sw_ana = onp.array(Sw_ana[0, :])

        l1_err = compute_l1_error(Sw_num, Sw_ana)
        l2_err = compute_l2_error(Sw_num, Sw_ana)
        max_err = compute_max_error(Sw_num, Sw_ana)

        results.append(
            {
                "nx": nx,
                "dx": dx,
                "l1": l1_err,
                "l2": l2_err,
                "max": max_err,
            }
        )

    return results


def run_temporal_refinement_study(
    params: FlowParams = None,
    base_dt: float = 0.001,
    refine_factors: list = None,
    nx: int = 100,
    t_max: float = 0.05,
    t_compare: float = 0.03,
    L: float = 100.0,
    Sw_inj: float = 1.0,
    Sw_init: float = None,
    scheme: str = "upwind",
) -> list:
    if params is None:
        params = FlowParams()
    if refine_factors is None:
        refine_factors = [1, 2, 4, 8]
    if Sw_init is None:
        Sw_init = params.Swc

    results = []

    dx = L / nx
    Sw_init_arr = jnp.full(nx, Sw_init)
    u = 1.0

    finest_factor = max(refine_factors)
    dt_finest = base_dt / finest_factor
    nt_finest = int(t_max / dt_finest)

    Sw_history_finest = simulate_1d_buckley_leverett(
        Sw_init_arr,
        u,
        params,
        dt_finest,
        dx,
        nt_finest,
        scheme=scheme,
        Sw_inj=Sw_inj,
    )

    n_steps_ref = int(t_compare / dt_finest)
    if n_steps_ref > nt_finest:
        n_steps_ref = nt_finest
    Sw_ref = onp.array(Sw_history_finest[n_steps_ref, :])

    for factor in refine_factors:
        dt = base_dt / factor
        nt = int(t_max / dt)

        Sw_history = simulate_1d_buckley_leverett(
            Sw_init_arr,
            u,
            params,
            dt,
            dx,
            nt,
            scheme=scheme,
            Sw_inj=Sw_inj,
        )

        n_steps = int(t_compare / dt)
        if n_steps > nt:
            n_steps = nt
        Sw_num = onp.array(Sw_history[n_steps, :])

        l1_err = compute_l1_error(Sw_num, Sw_ref)
        l2_err = compute_l2_error(Sw_num, Sw_ref)
        max_err = compute_max_error(Sw_num, Sw_ref)

        results.append(
            {
                "dt": dt,
                "l1": l1_err,
                "l2": l2_err,
                "max": max_err,
            }
        )

    return results
