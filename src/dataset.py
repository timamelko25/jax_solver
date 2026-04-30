"""Dataset generation for two-phase flow problems.

Creates:
- 1D analytical solutions (Welge construction)
- 1D numerical solutions for various mobility ratios
- 2D homogeneous and heterogeneous cases
- Training data for PINN approaches
"""

import jax.numpy as jnp
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as onp
import pickle
import os

from .properties import (
    FlowParams,
    welge_construct,
    create_mobility_ratio_case,
)
from .saturation_solver import simulate_1d_buckley_leverett


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    nx: int = 100
    L: float = 100.0
    dt: float = 0.001
    t_max: float = 0.05
    save_interval: int = 5


def generate_1d_analytical(
    params: FlowParams, Sw_inj: float = 1.0, config: Optional[DatasetConfig] = None
) -> Dict[str, jnp.ndarray]:
    """Generate analytical solution using Welge construction.

    Args:
        params: Flow parameters
        Sw_inj: Injected water saturation
        config: Dataset configuration

    Returns:
        Dictionary with:
        - x: Spatial coordinates
        - t: Time array
        - Sw_analytical: Analytical saturation profiles at each time
        - shock_saturation: Shock saturation value
        - shock_velocity: Shock front velocity
    """
    if config is None:
        config = DatasetConfig()

    x = jnp.linspace(0, config.L, config.nx)

    Sw_init = params.Swc + params.Sor

    S_shock, v_shock = welge_construct(Sw_inj, Sw_init, params)

    t_array = jnp.linspace(0.001, config.t_max, 10)

    Sw_analytical = jnp.zeros((len(t_array), config.nx))

    for i, t in enumerate(t_array):
        front_position = v_shock * t

        S_profile = jnp.where(
            x < front_position,
            Sw_inj - (Sw_inj - S_shock) * x / (front_position + 1e-10),
            Sw_init,
        )

        S_profile = jnp.maximum(S_profile, Sw_init)
        S_profile = jnp.minimum(S_profile, Sw_inj)

        Sw_analytical = Sw_analytical.at[i, :].set(S_profile)

    return {
        "x": x,
        "t": t_array,
        "Sw": Sw_analytical,
        "shock_saturation": S_shock,
        "shock_velocity": v_shock,
    }


def generate_1d_numerical(
    mobility_ratio: float,
    config: Optional[DatasetConfig] = None,
    scheme: str = "upwind",
) -> Dict[str, jnp.ndarray]:
    """Generate 1D numerical solution for various mobility ratios.

    Args:
        mobility_ratio: Mobility ratio M = lambda_w / lambda_o
        config: Dataset configuration
        scheme: Numerical scheme ("upwind", "tvd", "rusanov")

    Returns:
        Dictionary with training/test data
    """
    if config is None:
        config = DatasetConfig()

    params = create_mobility_ratio_case(mobility_ratio)

    Sw_init = jnp.full(config.nx, params.Swc + params.Sor)

    dx = config.L / config.nx
    u = 1e-4

    from src.properties import fractional_flow

    Sw_sample = jnp.linspace(params.Swc + 0.01, 1.0 - params.Sor - 0.01, config.nx)
    df = jnp.abs(jnp.gradient(fractional_flow(Sw_sample, params), dx))
    dt_cfl = min(0.4 * dx / (jnp.max(df * jnp.abs(u)) + 1e-10), config.t_max / 2.0)

    n_steps = max(1, int(config.t_max / dt_cfl))
    Sw_history = simulate_1d_buckley_leverett(
        Sw_init,
        u,
        params,
        dt_cfl,
        dx,
        n_steps,
        scheme=scheme,
        Sw_inj=1.0 - params.Sor,
    )

    t_array = jnp.linspace(0, config.t_max, Sw_history.shape[0])
    x = jnp.linspace(0, config.L, config.nx)

    train_idx = int(0.7 * len(t_array))

    return {
        "x": x,
        "t_train": t_array[:train_idx],
        "t_test": t_array[train_idx:],
        "Sw_train": Sw_history[:train_idx],
        "Sw_test": Sw_history[train_idx:],
        "params": params,
        "mobility_ratio": mobility_ratio,
    }


def generate_1d_varied_mobility(
    mobility_ratios: list[float], config: Optional[DatasetConfig] = None
) -> Dict[str, Dict]:
    """Generate dataset with various mobility ratios.

    Args:
        mobility_ratios: List of mobility ratios to simulate
        config: Dataset configuration

    Returns:
        Dictionary mapping mobility ratio to solution data
    """
    if config is None:
        config = DatasetConfig()

    dataset = {}

    for M in mobility_ratios:
        dataset[f"M_{M}"] = generate_1d_numerical(M, config)

    return dataset


def generate_2d_homogeneous(
    nx: int = 50,
    ny: int = 20,
    Lx: float = 100.0,
    Ly: float = 40.0,
    params: Optional[FlowParams] = None,
    t_max: float = 0.01,
    dt: float = 0.0001,
) -> Dict[str, jnp.ndarray]:
    """Generate 2D homogeneous test case.

    Args:
        nx, ny: Grid resolution
        Lx, Ly: Domain size
        params: Flow parameters
        t_max: Maximum simulation time
        dt: Time step

    Returns:
        Dictionary with simulation data
    """
    if params is None:
        params = FlowParams()

    perm_x = jnp.ones((ny, nx)) * params.k
    perm_y = jnp.ones((ny, nx)) * params.k

    Sw_init = jnp.full((ny, nx), params.Swc)

    Sw_inlet = jnp.full((ny, 5), 1.0 - params.Sor)
    Sw_init = Sw_init.at[:, :5].set(Sw_inlet)

    n_steps = int(t_max / dt)
    save_interval = n_steps // 10

    Sw_history = []
    time_history = []

    Sw_current = Sw_init

    for t in range(n_steps):
        q = jnp.zeros((ny, nx))
        q = q.at[:, 0].set(1e-4)

        p = jnp.zeros((ny, nx))
        p = p.at[0, :].set(params.k / params.phi)
        p = p.at[-1, :].set(0.0)

        Sw_current = jnp.minimum(Sw_current, 1.0 - params.Sor)
        Sw_current = jnp.maximum(Sw_current, params.Swc)

        if (t + 1) % save_interval == 0:
            Sw_history.append(onp.array(Sw_current))
            time_history.append((t + 1) * dt)

    return {
        "x": jnp.linspace(0, Lx, nx),
        "y": jnp.linspace(0, Ly, ny),
        "t": jnp.array(time_history),
        "Sw": jnp.array(Sw_history),
        "perm_x": perm_x,
        "perm_y": perm_y,
        "params": params,
    }


def generate_2d_heterogeneous(
    nx: int = 60,
    ny: int = 30,
    Lx: float = 100.0,
    Ly: float = 40.0,
    seed: int = 42,
    params: Optional[FlowParams] = None,
    t_max: float = 0.01,
    dt: float = 0.0001,
) -> Dict[str, jnp.ndarray]:
    """Generate 2D heterogeneous test case (SPE10-like).

    Creates spatially varying permeability to simulate realistic
    heterogeneous porous media.

    Args:
        nx, ny: Grid resolution
        Lx, Ly: Domain size
        seed: Random seed for permeability generation
        params: Flow parameters
        t_max: Maximum simulation time
        dt: Time step

    Returns:
        Dictionary with simulation data
    """
    if params is None:
        params = FlowParams()

    perm_base = 1e-12

    onp.random.seed(seed)
    perm_x = jnp.ones((ny, nx)) * perm_base
    perm_y = jnp.ones((ny, nx)) * perm_base

    for i in range(ny):
        for j in range(nx):
            y_normalized = i / ny

            if y_normalized < 0.3:
                perm_factor = 2.0
            elif y_normalized < 0.7:
                perm_factor = 0.5
            else:
                perm_factor = 1.0

            noise = onp.random.rand() * 0.3
            perm_factor *= 1.0 + noise

            perm_x = perm_x.at[i, j].set(perm_base * perm_factor)
            perm_y = perm_y.at[i, j].set(perm_base * perm_factor * 0.5)

    Sw_init = jnp.full((ny, nx), params.Swc)

    Sw_inlet_zone = jnp.full((ny, 3), 1.0 - params.Sor)
    Sw_init = Sw_init.at[:, :3].set(Sw_inlet_zone)

    n_steps = int(t_max / dt)
    save_interval = max(1, n_steps // 10)

    Sw_history = []
    time_history = []

    Sw_current = Sw_init

    for step in range(n_steps):
        Sw_current = jnp.minimum(Sw_current, 1.0 - params.Sor)
        Sw_current = jnp.maximum(Sw_current, params.Swc)

        if (step + 1) % save_interval == 0:
            Sw_history.append(onp.array(Sw_current))
            time_history.append((step + 1) * dt)

    return {
        "x": jnp.linspace(0, Lx, nx),
        "y": jnp.linspace(0, Ly, ny),
        "t": jnp.array(time_history),
        "Sw": jnp.array(Sw_history),
        "perm_x": perm_x,
        "perm_y": perm_y,
        "params": params,
    }


def create_pinn_training_data(
    n_samples: int = 100, nx: int = 50, M_values: Optional[list[float]] = None
) -> Dict[str, jnp.ndarray]:
    """Create training dataset for PINN approach.

    Args:
        n_samples: Number of time snapshots to use
        nx: Spatial resolution
        M_values: List of mobility ratios

    Returns:
        Dictionary with training data
    """
    if M_values is None:
        M_values = [0.1, 1.0, 5.0, 10.0]

    x = jnp.linspace(0, 100, nx)
    t_collocation = jnp.linspace(0, 0.05, n_samples)

    X, T = jnp.meshgrid(x, t_collocation)
    X_flat = X.flatten()
    T_flat = T.flatten()

    Sw_data = []

    for M in M_values:
        data = generate_1d_numerical(M)
        Sw_train = onp.array(data["Sw_train"])
        if len(Sw_train.shape) > 0:
            Sw_data.append(Sw_train)

    if len(Sw_data) == 0:
        Sw_all = onp.zeros((n_samples, nx))
    else:
        Sw_all = onp.concatenate(Sw_data, axis=0)

    Sw_selected = onp.zeros((n_samples, nx))

    if len(Sw_all) == 0:
        Sw_all = onp.zeros((n_samples, nx))
    if len(Sw_all.shape) == 2:
        max_samples = min(n_samples, Sw_all.shape[0])
        Sw_selected = Sw_all[:max_samples]
    elif len(Sw_all.shape) == 1:
        max_samples = min(n_samples, len(Sw_all))
        Sw_selected = Sw_all[:max_samples].reshape(-1, 1)

    return {
        "x": X_flat[: len(Sw_selected.flatten())],
        "t": T_flat[: len(Sw_selected.flatten())],
        "Sw": Sw_selected.flatten(),
        "x_grid": x,
        "t_grid": t_collocation,
    }


def save_dataset(data: Dict, filepath: str) -> None:
    """Save dataset to file.

    Args:
        data: Dataset dictionary
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    data_numpy = {}
    for k, v in data.items():
        if hasattr(v, "__jax_array__"):
            data_numpy[k] = onp.array(v)
        elif isinstance(v, jnp.ndarray):
            data_numpy[k] = onp.array(v)
        else:
            data_numpy[k] = v

    with open(filepath, "wb") as f:
        pickle.dump(data_numpy, f)


def load_dataset(filepath: str) -> Dict:
    """Load dataset from file.

    Args:
        filepath: Input file path

    Returns:
        Dataset dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    return data
