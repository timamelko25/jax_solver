"""Pressure solver for elliptic equation in porous media.

Solves: div(lambda(S) * grad(p)) = q

Using finite difference discretization and iterative solvers.
"""

import jax.numpy as jnp
from jax import jit
from typing import Optional
from .properties import FlowParams, compute_mobility


@jit
def solve_pressure_1d(
    Sw: jnp.ndarray,
    params: FlowParams,
    q: jnp.ndarray,
    dx: float,
    p_bc_left: float = 1e5,
    p_bc_right: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> jnp.ndarray:
    """Solve pressure equation in 1D using Gauss-Seidel iteration.

    Solves: d/dx(lambda(S) * dp/dx) = q

    Args:
        Sw: Water saturation (n,)
        params: Flow parameters
        q: Source term (n,)
        dx: Grid spacing
        p_bc_left: Pressure at left boundary
        p_bc_right: Pressure at right boundary
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Pressure field (n,)
    """
    n = Sw.shape[0]

    _, _, lambda_total = compute_mobility(Sw, params)

    p = jnp.zeros(n)
    p = p.at[0].set(p_bc_left)
    p = p.at[-1].set(p_bc_right)

    alpha = lambda_total * params.k / (params.phi * dx**2)

    for _ in range(max_iter):
        p_old = p

        p_left = p[:-2]
        p_right = p[2:]

        lambda_left = 0.5 * (lambda_total[:-1] + lambda_total[1:])[:-1]
        lambda_right = 0.5 * (lambda_total[:-1] + lambda_total[1:])[1:]

        rhs = q[1:-1]

        denom = (
            lambda_left
            + lambda_right
            + jnp.where(alpha[1:-1] > 0, 1.0 / alpha[1:-1], 1e10)
        )

        p_new = (
            lambda_left * p_left
            + lambda_right * p_right
            + rhs / jnp.where(alpha[1:-1] > 0, alpha[1:-1], 1e10)
        ) / denom

        p = p.at[1:-1].set(p_new)

        error = jnp.max(jnp.abs(p - p_old))
        if error < tol:
            break

    return p


@jit
def compute_velocity_1d(
    p: jnp.ndarray, lambda_total: jnp.ndarray, lambda_w: jnp.ndarray, dx: float
) -> jnp.ndarray:
    """Compute Darcy velocity from pressure field.

    u = -lambda * grad(p)

    Args:
        p: Pressure field
        lambda_total: Total mobility
        lambda_w: Water mobility
        dx: Grid spacing

    Returns:
        Total velocity and water velocity
    """
    dp_dx = (p[1:] - p[:-1]) / dx

    u_total = -lambda_total[1:] * dp_dx
    u_w = -lambda_w[1:] * dp_dx

    u_total = 0.5 * (u_total[:-1] + u_total[1:])
    u_w = 0.5 * (u_w[:-1] + u_w[1:])

    return u_total, u_w


def compute_velocity_2d(
    p: jnp.ndarray,
    lambda_total: jnp.ndarray,
    lambda_w: jnp.ndarray,
    dx: float,
    dy: float,
) -> tuple:
    """Compute Darcy velocity from 2D pressure field.

    u = -lambda * grad(p)

    Args:
        p: Pressure field (ny, nx)
        lambda_total: Total mobility (ny, nx)
        lambda_w: Water mobility (ny, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y

    Returns:
        Tuple of (u_total_x, u_total_y, u_w_x, u_w_y)
        - u_total_x: (ny, nx-1)
        - u_total_y: (ny-1, nx)
        - u_w_x: (ny, nx-1)
        - u_w_y: (ny-1, nx)
    """
    dp_dx = (p[:, 1:] - p[:, :-1]) / dx
    dp_dy = (p[1:, :] - p[:-1, :]) / dy

    u_total_x = -lambda_total[:, :-1] * dp_dx
    u_total_y = -lambda_total[:-1, :] * dp_dy

    u_w_x = -lambda_w[:, :-1] * dp_dx
    u_w_y = -lambda_w[:-1, :] * dp_dy

    return u_total_x, u_total_y, u_w_x, u_w_y


def solve_pressure_2d(
    Sw: jnp.ndarray,
    perm_x: jnp.ndarray,
    perm_y: jnp.ndarray,
    params: FlowParams,
    q: jnp.ndarray,
    dx: float,
    dy: float,
    p_bc: Optional[jnp.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> jnp.ndarray:
    """Solve pressure equation in 2D using Gauss-Seidel iteration.

    Solves: d/dx(lambda_x * dp/dx) + d/dy(lambda_y * dp/dy) = q

    Args:
        Sw: Water saturation (ny, nx)
        perm_x: Permeability in x-direction (ny, nx)
        perm_y: Permeability in y-direction (ny, nx)
        params: Flow parameters
        q: Source term (ny, nx)
        dx, dy: Grid spacing
        p_bc: Boundary conditions (optional)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Pressure field (ny, nx)
    """
    ny, nx = Sw.shape[0], Sw.shape[1]

    lambda_w, lambda_o, lambda_total = compute_mobility(Sw, params)

    lambda_x = lambda_total * perm_x
    lambda_y = lambda_total * perm_y

    p = jnp.zeros((ny, nx))
    if p_bc is not None:
        p = p.at[:].set(p_bc)

    for _ in range(max_iter):
        p_old = p

        p_left = p[1:-1, :-2]
        p_right = p[1:-1, 2:]
        p_down = p[:-2, 1:-1]
        p_up = p[2:, 1:-1]

        lx_center = lambda_x[1:-1, 1:-1]
        lx_left = lambda_x[1:-1, :-2]
        lx_right = lambda_x[1:-1, 2:]

        ly_center = lambda_y[1:-1, 1:-1]
        ly_down = lambda_y[:-2, 1:-1]
        ly_up = lambda_y[2:, 1:-1]

        hx = dx**2
        hy = dy**2

        denom = 2 * (lx_center / hx + ly_center / hy)
        denom = jnp.where(denom > 0, denom, 1e10)

        p_new = (
            lx_left * p_left / hx
            + lx_right * p_right / hx
            + ly_down * p_down / hy
            + ly_up * p_up / hy
            - q[1:-1, 1:-1]
        ) / denom

        p = p.at[1:-1, 1:-1].set(p_new)

        error = jnp.max(jnp.abs(p - p_old))
        if error < tol:
            break

    return p
