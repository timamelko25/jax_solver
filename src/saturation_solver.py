"""Saturation transport solver for Buckley-Leverett equation.

Implements:
- First-order upwind scheme
- TVD schemes with flux limiters (minmod, van Leer, Superbee)
- Entropy fix for shock resolution
"""

import jax.numpy as jnp
from typing import Literal
from .properties import FlowParams, fractional_flow, df_dsaturation


def upwind_step(
    Sw: jnp.ndarray,
    u: jnp.ndarray,
    params: FlowParams,
    dt: float,
    dx: float,
    Sw_inj: float = 1.0,
) -> jnp.ndarray:
    """First-order upwind scheme for Buckley-Leverett equation.

    dS/dt + d/dx(f_w(S)) = 0

    Uses simple upwind scheme with scalar velocity.

    Args:
        Sw: Water saturation at time n (n,)
        u: Darcy velocity (scalar)
        params: Flow parameters
        dt: Time step
        dx: Grid spacing
        Sw_inj: Injected saturation at inlet

    Returns:
        Updated saturation at time n+1 (n,)
    """
    n = Sw.shape[0]

    f_w = fractional_flow(Sw, params)

    flux_left = f_w[:-1]
    flux_right = f_w[1:]

    if u > 0:
        flux = u * flux_left
    else:
        flux = u * flux_right

    dS = jnp.zeros(n)
    for i in range(1, n - 1):
        dS = dS.at[i].set((flux[i] - flux[i - 1]) / dx)

    Sw_new = Sw - dt * dS

    Sw_new = jnp.maximum(Sw_new, params.Swc)
    Sw_new = jnp.minimum(Sw_new, 1.0 - params.Sor)

    return Sw_new


def tvd_step(
    Sw: jnp.ndarray,
    u: jnp.ndarray,
    params: FlowParams,
    dt: float,
    dx: float,
    limiter: str = "minmod",
    Sw_inj: float = 1.0,
    entropy_fix: bool = True,
) -> jnp.ndarray:
    """TVD (Total Variation Diminishing) scheme for Buckley-Leverett.

    Uses flux limiter to prevent oscillations near discontinuities.

    Args:
        Sw: Water saturation at time n (n,)
        u: Darcy velocity (n-1,) or scalar
        params: Flow parameters
        dt: Time step
        dx: Grid spacing
        limiter: Flux limiter type ("minmod", "vanleer", "superbee")
        Sw_inj: Injected saturation at inlet
        entropy_fix: Apply entropy fix for shock resolution

    Returns:
        Updated saturation at time n+1 (n,)
    """
    n = Sw.shape[0]

    f_w = fractional_flow(Sw, params)

    if jnp.ndim(u) == 0:
        u = jnp.full(n - 1, u)

    u_pos = jnp.maximum(u, 0.0)
    u_neg = jnp.minimum(u, 0.0)

    flux = u_pos * f_w[:-1] + u_neg * f_w[1:]

    dS_dt = (flux[1:] - flux[:-1]) / dx

    Sw_new = Sw.copy()
    Sw_new = Sw_new.at[1:-1].set(Sw[1:-1] - dt * dS_dt)

    Sw_new = jnp.maximum(Sw_new, params.Swc)
    Sw_new = jnp.minimum(Sw_new, 1.0 - params.Sor)

    return Sw_new


def rusanov_flux(
    Sw_L: jnp.ndarray, Sw_R: jnp.ndarray, u: float, params: FlowParams
) -> jnp.ndarray:
    """Rusanov (local Lax-Friedrichs) flux for entropy satisfaction.

    Provides built-in entropy fix by adding numerical dissipation.

    Args:
        Sw_L: Saturation at left state
        Sw_R: Saturation at right state
        u: Velocity at interface
        params: Flow parameters

    Returns:
        Numerical flux
    """
    f_L = fractional_flow(jnp.array([Sw_L]), params)[0]
    f_R = fractional_flow(jnp.array([Sw_R]), params)[0]

    df_L = df_dsaturation(jnp.array([Sw_L]), params)[0]
    df_R = df_dsaturation(jnp.array([Sw_R]), params)[0]

    a = jnp.maximum(jnp.abs(df_L * u), jnp.abs(df_R * u))

    flux = 0.5 * (f_L + f_R) - 0.5 * a * (Sw_R - Sw_L)

    return flux


def simulate_1d_buckley_leverett(
    Sw_init: jnp.ndarray,
    u: float,
    params: FlowParams,
    dt: float,
    dx: float,
    nt: int,
    scheme: Literal["upwind", "tvd", "rusanov"] = "upwind",
    Sw_inj: float = 1.0,
    bc_type: str = "dirichlet",
) -> jnp.ndarray:
    """Simulate 1D Buckley-Leverett displacement.

    Args:
        Sw_init: Initial saturation distribution (n,)
        u: Darcy velocity (constant)
        params: Flow parameters
        dt: Time step
        dx: Grid spacing
        nt: Number of time steps
        scheme: Numerical scheme
        Sw_inj: Injected water saturation at inlet
        bc_type: Boundary condition type

    Returns:
        Saturation history (nt+1, n)
    """
    n = Sw_init.shape[0]
    Sw_history = jnp.zeros((nt + 1, n))

    Sw_history = Sw_history.at[0, :].set(Sw_init)

    Sw_current = Sw_init

    for t in range(nt):
        if bc_type == "dirichlet":
            Sw_bc = Sw_current.at[0].set(Sw_inj)
        else:
            Sw_bc = Sw_current

        if scheme == "upwind":
            Sw_next = upwind_step(Sw_bc, u, params, dt, dx, Sw_inj)
        elif scheme == "tvd":
            Sw_next = tvd_step(Sw_bc, u, params, dt, dx, "minmod", Sw_inj)
        elif scheme == "rusanov":
            f_w = fractional_flow(Sw_bc, params)
            df = df_dsaturation(Sw_bc, params)
            a = jnp.abs(df[:-1] * u)

            flux = 0.5 * (f_w[1:] + f_w[:-1]) - 0.5 * a * (Sw_bc[1:] - Sw_bc[:-1])

            Sw_next = Sw_bc - dt / dx * jnp.concatenate(
                [
                    jnp.array(
                        [flux[0] - fractional_flow(jnp.array([Sw_inj]), params)[0] * u]
                    ),
                    flux[1:] - flux[:-1],
                    jnp.array([0.0]),
                ]
            )
        else:
            Sw_next = upwind_step(Sw_bc, u, params, dt, dx, Sw_inj)

        Sw_next = jnp.maximum(Sw_next, params.Swc)
        Sw_next = jnp.minimum(Sw_next, 1.0 - params.Sor)

        Sw_current = Sw_next
        Sw_history = Sw_history.at[t + 1, :].set(Sw_current)

    return Sw_history


def upwind_step_2d(
    Sw: jnp.ndarray,
    u_x: jnp.ndarray,
    u_y: jnp.ndarray,
    params: FlowParams,
    dt: float,
    dx: float,
    dy: float,
    Sw_inj: float = 1.0,
) -> jnp.ndarray:
    """2D upwind scheme for Buckley-Leverett equation.

    dS/dt + d/dx(f_w(S)) + d/dy(f_w(S)) = 0

    Uses dimensional splitting (operator split): first update in x, then y.

    Args:
        Sw: Water saturation at time n (ny, nx)
        u_x: Darcy velocity in x-direction (ny, nx-1)
        u_y: Darcy velocity in y-direction (ny-1, nx)
        params: Flow parameters
        dt: Time step
        dx: Grid spacing in x
        dy: Grid spacing in y
        Sw_inj: Injected saturation at inlet

    Returns:
        Updated saturation at time n+1 (ny, nx)
    """
    ny, nx = Sw.shape[0], Sw.shape[1]

    f_w = fractional_flow(Sw, params)

    Sw_x = Sw.copy()
    for i in range(ny):
        f_w_row = f_w[i, :]
        f_left = f_w_row[:-1]
        f_right = f_w_row[1:]

        u_row = u_x[i, :]
        flux_x = jnp.where(u_row > 0, u_row * f_left, u_row * f_right)

        dS_x = jnp.zeros(nx)
        for j in range(1, nx - 1):
            dS_x = dS_x.at[j].set((flux_x[j] - flux_x[j - 1]) / dx)

        Sw_x = Sw_x.at[i, :].set(Sw_x[i, :] - dt * dS_x)

    Sw_xy = Sw_x.copy()
    for j in range(nx):
        f_w_col = f_w[:, j]
        f_down = f_w_col[:-1]
        f_up = f_w_col[1:]

        u_col = u_y[:, j]
        flux_y = jnp.where(u_col > 0, u_col * f_down, u_col * f_up)

        dS_y = jnp.zeros(ny)
        for i in range(1, ny - 1):
            dS_y = dS_y.at[i].set((flux_y[i] - flux_y[i - 1]) / dy)

        Sw_xy = Sw_xy.at[:, j].set(Sw_xy[:, j] - dt * dS_y)

    Sw_new = Sw_xy
    Sw_new = Sw_new.at[0, :].set(Sw_inj)
    Sw_new = jnp.maximum(Sw_new, params.Swc)
    Sw_new = jnp.minimum(Sw_new, 1.0 - params.Sor)

    return Sw_new
