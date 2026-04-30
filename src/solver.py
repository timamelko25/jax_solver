"""IMPES solver for coupled two-phase flow.

Implements Implicit Pressure - Explicit Saturation algorithm:
1. Solve pressure equation (elliptic) implicitly
2. Update saturation (hyperbolic) explicitly
"""

import jax.numpy as jnp
from jax import jit
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

from .properties import FlowParams, fractional_flow, compute_mobility
from .pressure_solver import (
    solve_pressure_3d,
    compute_velocity_3d,
    solve_pressure_1d,
    compute_velocity_1d,
    solve_pressure_2d,
    compute_velocity_2d,
)
from .saturation_solver import upwind_step, tvd_step, upwind_step_2d, upwind_step_3d


@dataclass
class SimulationConfig:
    """Configuration for IMPES simulation."""

    nx: int = 100
    L: float = 100.0
    dt: float = 0.001
    t_max: float = 0.1
    p_inlet: float = 1e5
    p_outlet: float = 0.0
    q_injection: float = 1e-4
    scheme: str = "upwind"
    save_interval: int = 10


class IMPESSolver:
    """IMPES solver for two-phase flow in porous media."""

    def __init__(self, params: FlowParams, config: SimulationConfig):
        """Initialize IMPES solver.

        Args:
            params: Flow parameters
            config: Simulation configuration
        """
        self.params = params
        self.config = config

        self.dx = config.L / config.nx

        self.x = jnp.linspace(0, config.L, config.nx)

    def compute_time_step(self, Sw: jnp.ndarray, u: jnp.ndarray) -> float:
        """Compute stable time step using CFL condition.

        Args:
            Sw: Current saturation
            u: Darcy velocity

        Returns:
            Stable time step
        """
        df = jnp.abs(jnp.gradient(fractional_flow(Sw, self.params), self.dx))

        cfl_number = 0.4
        dt_cfl = cfl_number * self.dx / (jnp.max(df * jnp.abs(u)) + 1e-10)

        return jnp.minimum(dt_cfl, self.config.dt)

    def step(
        self, Sw: jnp.ndarray, dt: Optional[float] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform one IMPES time step.

        Args:
            Sw: Current water saturation
            dt: Time step (if None, uses config)

        Returns:
            Tuple of (new_saturation, pressure)
        """
        if dt is None:
            dt = self.config.dt

        q = jnp.zeros(self.config.nx)
        q = q.at[0].set(self.config.q_injection / self.dx)

        p = solve_pressure_1d(
            Sw, self.params, q, self.dx, self.config.p_inlet, self.config.p_outlet
        )

        lambda_w, _, lambda_total = compute_mobility(Sw, self.params)

        u_total, u_w = compute_velocity_1d(p, lambda_total, lambda_w, self.dx)

        Sw_bc = Sw.at[0].set(1.0 - self.params.Sor)

        if self.config.scheme == "upwind":
            Sw_new = upwind_step(Sw_bc, u_total, self.params, dt, self.dx, 1.0)
        else:
            Sw_new = tvd_step(Sw_bc, u_total, self.params, dt, self.dx)

        Sw_new = jnp.maximum(Sw_new, self.params.Swc)
        Sw_new = jnp.minimum(Sw_new, 1.0 - self.params.Sor)

        return Sw_new, p

    def run(
        self, Sw_init: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run full simulation.

        Args:
            Sw_init: Initial saturation (if None, uses connate water)

        Returns:
            Tuple of (time_array, saturation_history, pressure_history)
        """
        if Sw_init is None:
            Sw_init = jnp.full(self.config.nx, self.params.Swc)

        n_steps = int(self.config.t_max / self.config.dt)
        n_save = n_steps // self.config.save_interval + 1

        time_history = jnp.zeros(n_save)
        Sw_history = jnp.zeros((n_save, self.config.nx))
        p_history = jnp.zeros((n_save, self.config.nx))

        time_history = time_history.at[0].set(0.0)
        Sw_history = Sw_history.at[0, :].set(Sw_init)

        Sw_current = Sw_init

        save_idx = 1

        for t in range(n_steps):
            Sw_new, p = self.step(Sw_current)

            Sw_current = Sw_new

            if (t + 1) % self.config.save_interval == 0:
                time_history = time_history.at[save_idx].set((t + 1) * self.config.dt)
                Sw_history = Sw_history.at[save_idx, :].set(Sw_current)
                p_history = p_history.at[save_idx, :].set(p)
                save_idx += 1

        return time_history[:save_idx], Sw_history[:save_idx], p_history[:save_idx]


@dataclass
class SimulationConfig2D:
    """Configuration for 2D IMPES simulation."""

    nx: int = 60
    ny: int = 30
    Lx: float = 100.0
    Ly: float = 50.0
    dt: float = 0.001
    t_max: float = 0.01
    p_inlet: float = 1e5
    p_outlet: float = 0.0
    q_injection: float = 1e-4
    scheme: str = "upwind"
    save_interval: int = 10


class IMPESSolver2D:
    """2D IMPES solver for two-phase flow in porous media."""

    def __init__(
        self,
        params: FlowParams,
        config: SimulationConfig2D,
        perm_x: Optional[jnp.ndarray] = None,
        perm_y: Optional[jnp.ndarray] = None,
    ):
        """Initialize 2D IMPES solver.

        Args:
            params: Flow parameters
            config: Simulation configuration
            perm_x: Permeability in x-direction (ny, nx)
            perm_y: Permeability in y-direction (ny, nx)
        """
        self.params = params
        self.config = config

        self.dx = config.Lx / config.nx
        self.dy = config.Ly / config.ny

        self.x = jnp.linspace(0, config.Lx, config.nx) + 0.5 * self.dx
        self.y = jnp.linspace(0, config.Ly, config.ny) + 0.5 * self.dy

        if perm_x is None:
            perm_x = jnp.ones((config.ny, config.nx)) * 1e-12
        if perm_y is None:
            perm_y = jnp.ones((config.ny, config.nx)) * 1e-12

        self.perm_x = perm_x
        self.perm_y = perm_y

    def step(
        self, Sw: jnp.ndarray, dt: Optional[float] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform one IMPES time step.

        Args:
            Sw: Current water saturation (ny, nx)
            dt: Time step (if None, uses config)

        Returns:
            Tuple of (new_saturation, pressure)
        """
        if dt is None:
            dt = self.config.dt

        ny, nx = Sw.shape[0], Sw.shape[1]

        q = jnp.zeros((ny, nx))
        q = q.at[:, 0].set(self.config.q_injection / self.dx)

        p = solve_pressure_2d(
            Sw,
            self.perm_x,
            self.perm_y,
            self.params,
            q,
            self.dx,
            self.dy,
            p_bc=None,
            max_iter=100,
            tol=1e-6,
        )

        lambda_w, lambda_o, lambda_total = compute_mobility(Sw, self.params)

        u_tx, u_ty, u_wx, u_wy = compute_velocity_2d(
            p, lambda_total, lambda_w, self.dx, self.dy
        )

        u_combined_x = jnp.maximum(u_tx, 0.0)
        u_combined_y = jnp.maximum(u_ty, 0.0)

        Sw_bc = Sw.copy()
        Sw_bc = Sw_bc.at[:, 0].set(1.0 - self.params.Sor)

        Sw_new = upwind_step_2d(
            Sw_bc,
            u_combined_x,
            u_combined_y,
            self.params,
            dt,
            self.dx,
            self.dy,
            1.0 - self.params.Sor,
        )

        Sw_new = jnp.maximum(Sw_new, self.params.Swc)
        Sw_new = jnp.minimum(Sw_new, 1.0 - self.params.Sor)

        return Sw_new, p

    def run(
        self, Sw_init: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run full 2D simulation.

        Args:
            Sw_init: Initial saturation (if None, uses connate water)

        Returns:
            Tuple of (time_array, saturation_history, pressure_history)
        """
        if Sw_init is None:
            Sw_init = jnp.full((self.config.ny, self.config.nx), self.params.Swc)

        nx, ny = self.config.nx, self.config.ny
        n_steps = int(self.config.t_max / self.config.dt)
        n_save = n_steps // self.config.save_interval + 1

        time_history = jnp.zeros(n_save)
        Sw_history = jnp.zeros((n_save, ny, nx))
        p_history = jnp.zeros((n_save, ny, nx))

        time_history = time_history.at[0].set(0.0)
        Sw_history = Sw_history.at[0, :, :].set(Sw_init)

        Sw_current = Sw_init

        save_idx = 1

        for t in range(n_steps):
            Sw_new, p = self.step(Sw_current)

            Sw_current = Sw_new

            if (t + 1) % self.config.save_interval == 0:
                time_history = time_history.at[save_idx].set((t + 1) * self.config.dt)
                Sw_history = Sw_history.at[save_idx, :, :].set(Sw_current)
                p_history = p_history.at[save_idx, :, :].set(p)
                save_idx += 1

        return time_history[:save_idx], Sw_history[:save_idx], p_history[:save_idx]


def simulate(
    params: FlowParams,
    Sw_init: jnp.ndarray,
    config: SimulationConfig,
    return_solver: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run IMPES simulation.

    Args:
        params: Flow parameters
        Sw_init: Initial saturation
        config: Simulation config
        return_solver: If True, returns solver object

    Returns:
        Tuple of (time, saturation, pressure) or (solver, time, saturation, pressure)
    """
    solver = IMPESSolver(params, config)
    time, Sw, p = solver.run(Sw_init)

    if return_solver:
        return solver, time, Sw, p
    return time, Sw, p


@jit
def residual_pinn(
    Sw_pred: jnp.ndarray, Sw_true: jnp.ndarray, params: FlowParams, dx: float
) -> float:
    """Compute residual for PINN training.

    Args:
        Sw_pred: Predicted saturation
        Sw_true: True saturation (for supervised loss)
        params: Flow parameters
        dx: Grid spacing

    Returns:
        Loss value
    """
    f_w_pred = fractional_flow(Sw_pred, params)

    dS_dx = jnp.gradient(Sw_pred, dx)
    df_dS = jnp.gradient(f_w_pred, dx)

    conservation_residual = dS_dx + df_dS

    mse_data = jnp.mean((Sw_pred - Sw_true) ** 2)
    mse_physics = jnp.mean(conservation_residual**2)

    return mse_data + 0.1 * mse_physics


def create_pinn_loss(
    params: FlowParams, dx: float, weight_physics: float = 0.1
) -> Callable:
    """Create PINN loss function for training.

    Args:
        params: Flow parameters
        dx: Grid spacing
        weight_physics: Weight for physics-based loss

    Returns:
        Loss function
    """

    def loss_fn(Sw_pred: jnp.ndarray, Sw_true: jnp.ndarray) -> float:
        return residual_pinn(Sw_pred, Sw_true, params, dx)

    return loss_fn


@dataclass
class SimulationConfig3D:
    """Configuration for 3D IMPES simulation."""

    nz: int = 43
    ny: int = 30
    nx: int = 60
    Lz: float = 43.0
    Ly: float = 30.0
    Lx: float = 60.0
    dt: float = 0.001
    t_max: float = 0.01
    p_inlet: float = 1e5
    p_outlet: float = 0.0
    q_injection: float = 1e-4
    scheme: str = "upwind"
    save_interval: int = 10


class IMPESSolver3D:
    """3D IMPES solver for two-phase flow in porous media."""

    def __init__(
        self,
        params: FlowParams,
        config: SimulationConfig3D,
        perm_x: Optional[jnp.ndarray] = None,
        perm_y: Optional[jnp.ndarray] = None,
        perm_z: Optional[jnp.ndarray] = None,
    ):
        self.params = params
        self.config = config

        self.dx = config.Lx / config.nx
        self.dy = config.Ly / config.ny
        self.dz = config.Lz / config.nz

        self.x = jnp.linspace(0, config.Lx, config.nx) + 0.5 * self.dx
        self.y = jnp.linspace(0, config.Ly, config.ny) + 0.5 * self.dy
        self.z = jnp.linspace(0, config.Lz, config.nz) + 0.5 * self.dz

        if perm_x is None:
            perm_x = jnp.ones((config.nz, config.ny, config.nx)) * 1e-12
        if perm_y is None:
            perm_y = jnp.ones((config.nz, config.ny, config.nx)) * 1e-12
        if perm_z is None:
            perm_z = jnp.ones((config.nz, config.ny, config.nx)) * 1e-12

        self.perm_x = perm_x
        self.perm_y = perm_y
        self.perm_z = perm_z

    def step(
        self, Sw: jnp.ndarray, dt: Optional[float] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if dt is None:
            dt = self.config.dt

        nz, ny, nx = Sw.shape[0], Sw.shape[1], Sw.shape[2]

        q = jnp.zeros((nz, ny, nx))
        q = q.at[:, :, 0].set(self.config.q_injection / self.dx / self.dy)

        p = solve_pressure_3d(
            Sw,
            self.perm_x,
            self.perm_y,
            self.perm_z,
            self.params,
            q,
            self.dx,
            self.dy,
            self.dz,
            p_bc=None,
            max_iter=100,
            tol=1e-6,
        )

        lambda_w, lambda_o, lambda_total = compute_mobility(Sw, self.params)

        u_total, u_w = compute_velocity_3d(
            p, lambda_total, lambda_w, self.dx, self.dy, self.dz
        )

        # Use magnitue for upwind scheme
        u_combined = jnp.maximum(u_total, 0.0)

        Sw_bc = Sw.copy()
        Sw_bc = Sw_bc.at[:, :, 0].set(1.0 - self.params.Sor)

        Sw_new = upwind_step_3d(
            Sw_bc,
            u_combined,
            self.params,
            dt,
            self.dx,
            self.dy,
            self.dz,
            1.0 - self.params.Sor,
        )

        Sw_new = jnp.maximum(Sw_new, self.params.Swc)
        Sw_new = jnp.minimum(Sw_new, 1.0 - self.params.Sor)

        return Sw_new, p

    def run(
        self, Sw_init: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if Sw_init is None:
            Sw_init = jnp.full(
                (self.config.nz, self.config.ny, self.config.nx), self.params.Swc
            )

        nz, ny, nx = self.config.nz, self.config.ny, self.config.nx
        n_steps = int(self.config.t_max / self.config.dt)
        n_save = n_steps // self.config.save_interval + 1

        time_history = jnp.zeros(n_save)
        Sw_history = jnp.zeros((n_save, nz, ny, nx))
        p_history = jnp.zeros((n_save, nz, ny, nx))

        time_history = time_history.at[0].set(0.0)
        Sw_history = Sw_history.at[0, :, :, :].set(Sw_init)

        Sw_current = Sw_init
        save_idx = 1

        for t in range(n_steps):
            Sw_new, p = self.step(Sw_current)

            Sw_current = Sw_new

            if (t + 1) % self.config.save_interval == 0:
                time_history = time_history.at[save_idx].set((t + 1) * self.config.dt)
                Sw_history = Sw_history.at[save_idx, :, :, :].set(Sw_current)
                p_history = p_history.at[save_idx, :, :, :].set(p)
                save_idx += 1

        return time_history[:save_idx], Sw_history[:save_idx], p_history[:save_idx]
