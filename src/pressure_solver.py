"""Pressure solver for 2D and 3D IMPES."""

from typing import Optional
import jax.numpy as jnp
from jax import jit, lax
from src.properties import FlowParams, compute_mobility


def solve_pressure_1d(
    Sw: jnp.ndarray,
    perm: jnp.ndarray,
    params: FlowParams,
    q: jnp.ndarray,
    dx: float,
    p_bc: Optional[jnp.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> jnp.ndarray:
    """Solve pressure equation in 1D using Gauss-Seidel iteration.

    Solves: d/dx(lambda * dp/dx) = q

    Args:
        Sw: Water saturation (nx,)
        perm: Permeability (nx,)
        params: Flow parameters
        q: Source term (nx,)
        dx: Grid spacing
        p_bc: Boundary conditions (optional)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Pressure field (nx,)
    """
    nx = Sw.shape[0]

    lambda_w, lambda_o, lambda_total = compute_mobility(Sw, params)
    lambda_eff = lambda_total * perm

    p = jnp.zeros(nx)
    if p_bc is not None:
        p = p.at[:].set(p_bc)

    h2 = dx**2

    for _ in range(max_iter):
        p_old = p

        for i in range(1, nx - 1):
            p_left = p[i - 1]
            p_right = p[i + 1]
            l_center = lambda_eff[i]
            l_left = lambda_eff[i - 1]
            l_right = lambda_eff[i + 1]

            denom = 2.0 * l_center / h2
            denom = jnp.where(denom > 0, denom, 1e10)

            p_new = (l_left * p_left / h2 + l_right * p_right / h2 - q[i]) / denom

            p = p.at[i].set(p_new)

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


@jit
def _solve_pressure_2d_jitted(p, lambda_x, lambda_y, q, dx, dy, max_iter, tol):
    hx = dx**2
    hy = dy**2

    def cond_func(state):
        i, p, converged = state
        return jnp.logical_and(i < max_iter, jnp.logical_not(converged))

    def body_func(state):
        i, p, converged = state
        p_old = p

        p_left = p[1:-1, :-2]
        p_right = p[1:-1, 2:]
        p_down = p[:-2, 1:-1]
        p_up = p[2:, 1:-1]

        lx_left = lambda_x[1:-1, :-2]
        lx_right = lambda_x[1:-1, 2:]
        ly_down = lambda_y[:-2, 1:-1]
        ly_up = lambda_y[2:, 1:-1]

        denom = 2.0 * (lambda_x[1:-1, 1:-1] / hx + lambda_y[1:-1, 1:-1] / hy)
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
        converged = error < tol

        return i + 1, p, converged

    init_state = (0, p, False)
    _, p_final, _ = lax.while_loop(cond_func, body_func, init_state)
    return p_final


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
    """Solve pressure equation in 2D using vectorized Jacobi iteration.

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

    p = _solve_pressure_2d_jitted(p, lambda_x, lambda_y, q, dx, dy, max_iter, tol)

    return p


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
    """
    dp_dx = (p[:, 1:] - p[:, :-1]) / dx
    dp_dy = (p[1:, :] - p[:-1, :]) / dy

    u_total_x = -lambda_total[:, :-1] * dp_dx
    u_total_y = -lambda_total[:-1, :] * dp_dy

    u_w_x = -lambda_w[:, :-1] * dp_dx
    u_w_y = -lambda_w[:-1, :] * dp_dy

    return u_total_x, u_total_y, u_w_x, u_w_y


@jit
def _solve_pressure_3d_jitted(
    p, lambda_x, lambda_y, lambda_z, q, dx, dy, dz, max_iter, tol
):
    """Vectorised 3D Jacobi iteration for pressure solver."""
    hx = dx**2
    hy = dy**2
    hz = dz**2

    nz, ny, nx = p.shape

    def cond_func(state):
        i, p, converged = state
        return jnp.logical_and(i < max_iter, jnp.logical_not(converged))

    def body_func(state):
        i, p, converged = state
        p_old = p

        # Interior points: use actual neighbours
        # x-neighbours: left = p[i,j,k-1], right = p[i,j,k+1]
        p_left_int = jnp.zeros_like(p)
        p_left_int = p_left_int.at[:, :, 1:].set(p[:, :, :-1])
        p_left_int = p_left_int.at[:, :, 0].set(p[:, :, 0])  # Neumann BC

        p_right_int = jnp.zeros_like(p)
        p_right_int = p_right_int.at[:, :, :-1].set(p[:, :, 1:])
        p_right_int = p_right_int.at[:, :, -1].set(p[:, :, -1])  # Neumann BC

        # y-neighbours
        p_down_int = jnp.zeros_like(p)
        p_down_int = p_down_int.at[:, 1:, :].set(p[:, :-1, :])
        p_down_int = p_down_int.at[:, 0, :].set(p[:, 0, :])  # Neumann BC

        p_up_int = jnp.zeros_like(p)
        p_up_int = p_up_int.at[:, :-1, :].set(p[:, 1:, :])
        p_up_int = p_up_int.at[:, -1, :].set(p[:, -1, :])  # Neumann BC

        # z-neighbours
        p_back_int = jnp.zeros_like(p)
        p_back_int = p_back_int.at[1:, :, :].set(p[:-1, :, :])
        p_back_int = p_back_int.at[0, :, :].set(p[0, :, :])  # Neumann BC

        p_front_int = jnp.zeros_like(p)
        p_front_int = p_front_int.at[:-1, :, :].set(p[1:, :, :])
        p_front_int = p_front_int.at[-1, :, :].set(p[-1, :, :])  # Neumann BC

        lx = lambda_x
        ly = lambda_y
        lz = lambda_z

        denom = 2.0 * (lx / hx + ly / hy + lz / hz)
        denom = jnp.where(denom > 0, denom, 1e10)

        p_new = (
            lx * p_left_int / hx
            + lx * p_right_int / hx
            + ly * p_down_int / hy
            + ly * p_up_int / hy
            + lz * p_back_int / hz
            + lz * p_front_int / hz
            - q
        ) / denom

        p = p_new
        error = jnp.max(jnp.abs(p - p_old))
        converged = error < tol

        return i + 1, p, converged

    init_state = (0, p, False)
    _, p_final, _ = lax.while_loop(
        lambda s: jnp.logical_and(s[0] < max_iter, jnp.logical_not(s[2])),
        body_func,
        init_state,
    )
    return p_final


def solve_pressure_3d(
    Sw: jnp.ndarray,
    perm_x: jnp.ndarray,
    perm_y: jnp.ndarray,
    perm_z: jnp.ndarray,
    params: FlowParams,
    q: jnp.ndarray,
    dx: float,
    dy: float,
    dz: float,
    p_bc: Optional[jnp.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> jnp.ndarray:
    """Solve pressure equation in 3D using vectorised Jacobi iteration.

    Solves: d/dx(lambda_x * dp/dx) + d/dy(lambda_y * dp/dy) + d/dz(lambda_z * dp/dz) = q.

    Uses vectorised Jacobi iteration with Neumann BC (zero gradient) at all boundaries.

    Args:
        Sw: Water saturation (nz, ny, nx)
        perm_x: Permeability in x-direction (nz, ny, nx)
        perm_y: Permeability in y-direction (nz, ny, nx)
        perm_z: Permeability in z-direction (nz, ny, nx)
        params: Flow parameters
        q: Source term (nz, ny, nx)
        dx, dy, dz: Grid spacing
        p_bc: Boundary conditions (optional)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Pressure field (nz, ny, nx)
    """
    nz, ny, nx = Sw.shape[0], Sw.shape[1], Sw.shape[2]

    lambda_w, lambda_o, lambda_total = compute_mobility(Sw, params)

    lambda_x = lambda_total * perm_x
    lambda_y = lambda_total * perm_y
    lambda_z = lambda_total * perm_z

    p = jnp.zeros((nz, ny, nx))
    if p_bc is not None:
        p = p.at[:].set(p_bc)

    p = _solve_pressure_3d_jitted(
        p, lambda_x, lambda_y, lambda_z, q, dx, dy, dz, max_iter, tol
    )

    return p


def compute_velocity_3d(
    p: jnp.ndarray,
    lambda_total: jnp.ndarray,
    lambda_w: jnp.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> tuple:
    """Compute Darcy velocity from 3D pressure field.

    u = -lambda * grad(p)

    Args:
        p: Pressure field (nz, ny, nx)
        lambda_total: Total mobility (nz, ny, nx)
        lambda_w: Water mobility (nz, ny, nx)
        dx, dy, dz: Grid spacing

    Returns:
        Tuple of (u_total, u_w) - magnitude at cell centers.
    """
    # Gradients at cell centers using central differences
    dp_dx = jnp.zeros_like(p)
    dp_dx = dp_dx.at[:, :, 1:-1].set((p[:, :, 2:] - p[:, :, :-2]) / (2 * dx))

    dp_dy = jnp.zeros_like(p)
    dp_dy = dp_dy.at[:, 1:-1, :].set((p[:, 2:, :] - p[:, :-2, :]) / (2 * dy))

    dp_dz = jnp.zeros_like(p)
    dp_dz = dp_dz.at[1:-1, :, :].set((p[2:, :, :] - p[:-2, :, :]) / (2 * dz))

    # Velocity magnitude at cell centers
    u_total_x = -lambda_total * dp_dx
    u_total_y = -lambda_total * dp_dy
    u_total_z = -lambda_total * dp_dz

    u_w_x = -lambda_w * dp_dx
    u_w_y = -lambda_w * dp_dy
    u_w_z = -lambda_w * dp_dz

    # Magnitude
    u_total = jnp.sqrt(u_total_x**2 + u_total_y**2 + u_total_z**2)
    u_w = jnp.sqrt(u_w_x**2 + u_w_y**2 + u_w_z**2)

    return u_total, u_w
