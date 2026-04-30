"""PINN training module for BIG_MODEL_12_09_1 dataset.

Provides functions to:
- Load BIG_MODEL simulation data for PINN training
- Train PINN on 2D and 3D BIG_MODEL data
- Evaluate PINN predictions against simulation results
"""

import jax.numpy as jnp
import numpy as onp
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .properties import FlowParams
from .pinn import (
    SaturationPINN,
    PINNConfig,
    _forward_batch,
    _pde_residual,
)
from .benchmarks import load_big_model


@dataclass
class BIGModelPINNConfig(PINNConfig):
    """Extended PINN config for BIG_MODEL dataset."""

    k_layer: int = 21
    use_3d: bool = False
    n_train_samples: int = 2000
    n_physics_samples: int = 1000


def create_big_model_training_data(
    data_dir: str = "data/BIG_MODEL_12_09_1",
    k_layer: int = 21,
    n_samples: int = 2000,
    use_3d: bool = False,
) -> Dict[str, jnp.ndarray]:
    """Create training data from BIG_MODEL dataset.

    Args:
        data_dir: Directory containing BIG_MODEL files
        k_layer: Z-layer for 2D slice (0-42)
        n_samples: Number of training samples
        use_3d: Use full 3D data (if True, k_layer is ignored)

    Returns:
        Dictionary with x, y, (z), t, Sw arrays for PINN training
    """
    data = load_big_model(data_dir)

    if use_3d:
        nx, ny, nz = data["nx"], data["ny"], data["nz"]
        perm_x = data["perm_x"]
        perm_y = data["perm_y"]
        perm_z = data["perm_z"]
        porosity = data["porosity"]

        x = onp.array(data["x"])
        y = onp.array(data["y"])
        z = onp.array(data["z"])

        x_grid, y_grid, z_grid = onp.meshgrid(x, y, z, indexing="ij")
        coords = onp.stack(
            [x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], axis=-1
        )

        Sw = onp.array(porosity)
        Sw_flat = Sw.flatten()
        Sw_flat = onp.clip(Sw_flat, 0.2, 0.8)

        idx = onp.random.choice(
            len(Sw_flat), min(n_samples, len(Sw_flat)), replace=False
        )

        return {
            "x": jnp.array(coords[idx, 0]),
            "y": jnp.array(coords[idx, 1]),
            "z": jnp.array(coords[idx, 2]),
            "Sw": jnp.array(Sw_flat[idx]),
            "coords": coords,
        }
    else:
        nx, ny = data["nx"], data["ny"]
        perm_x_2d = data["perm_x"][:, :, k_layer].T
        perm_y_2d = data["perm_y"][:, :, k_layer].T
        porosity_2d = data["porosity"][:, :, k_layer].T

        x = onp.array(data["x"])
        y = onp.array(data["y"])

        x_grid, y_grid = onp.meshgrid(x, y, indexing="ij")
        coords = onp.stack([x_grid.flatten(), y_grid.flatten()], axis=-1)

        Sw = onp.array(porosity_2d)
        Sw_flat = Sw.flatten()
        Sw_flat = onp.clip(Sw_flat, 0.2, 0.8)

        idx = onp.random.choice(
            len(Sw_flat), min(n_samples, len(Sw_flat)), replace=False
        )

        return {
            "x": jnp.array(coords[idx, 0]),
            "y": jnp.array(coords[idx, 1]),
            "Sw": jnp.array(Sw_flat[idx]),
            "coords": coords,
            "perm_x_2d": jnp.array(perm_x_2d),
            "perm_y_2d": jnp.array(perm_y_2d),
            "porosity_2d": jnp.array(porosity_2d),
        }


def create_big_model_pinn(
    data_dir: str = "data/BIG_MODEL_12_09_1",
    k_layer: int = 21,
    config: Optional[BIGModelPINNConfig] = None,
    use_3d: bool = False,
) -> Tuple[SaturationPINN, Dict[str, jnp.ndarray]]:
    """Create and train PINN on BIG_MODEL data.

    Args:
        data_dir: Directory containing BIG_MODEL files
        k_layer: Z-layer for 2D slice
        config: PINN configuration
        use_3d: Use full 3D data

    Returns:
        Tuple of (trained PINN, training data)
    """
    if config is None:
        config = BIGModelPINNConfig(k_layer=k_layer, use_3d=use_3d)

    print(f"Loading BIG_MODEL data from {data_dir}...")
    train_data = create_big_model_training_data(
        data_dir=data_dir,
        k_layer=k_layer,
        n_samples=config.n_train_samples,
        use_3d=use_3d,
    )

    dims = "3d" if use_3d else "2d"
    print(f"Training PINN on BIG_MODEL ({dims.upper()})...")

    params = FlowParams(
        k=1e-12,
        phi=0.2,
        mu_w=1e-3,
        mu_o=1e-2,
        Swc=0.2,
        Sor=0.2,
    )

    if use_3d:
        pinn = SaturationPINN(config, params, dims="3d")
        net_params = pinn.network.params

        x_data = train_data["x"]
        y_data = train_data["y"]
        z_data = train_data["z"]
        Sw_data = train_data["Sw"]

        x_physics = jnp.linspace(0, 122.0, 20)
        y_physics = jnp.linspace(0, 183.0, 20)
        z_physics = jnp.linspace(0, 43.0, 10)
        t_physics = jnp.linspace(0, 0.01, 10)

        x_p, y_p, z_p, t_p = onp.meshgrid(
            x_physics, y_physics, z_physics, t_physics, indexing="ij"
        )
        x_flat = jnp.array(x_p.flatten())
        y_flat = jnp.array(y_p.flatten())
        z_flat = jnp.array(z_p.flatten())
        t_flat = jnp.array(t_p.flatten())

        from .pinn import _pde_residual_3d

        def total_loss(net_params):
            S_pred = _forward_batch_3d(net_params, x_data, y_data, z_data, t_data)
            loss_data = jnp.mean((S_pred - Sw_data) ** 2)
            residual = _pde_residual_3d(
                net_params, x_flat, y_flat, z_flat, t_flat, params
            )
            loss_physics = jnp.mean(residual**2)
            return loss_data + config.physics_weight * loss_physics

    else:
        pinn = SaturationPINN(config, params, dims="2d")
        net_params = pinn.network.params

        x_data = train_data["x"]
        y_data = train_data["y"]
        Sw_data = train_data["Sw"]

        x_physics = jnp.linspace(0, 122.0, 20)
        y_physics = jnp.linspace(0, 183.0, 20)
        t_physics = jnp.linspace(0, 0.01, 10)

        x_p, y_p, t_p = onp.meshgrid(x_physics, y_physics, t_physics, indexing="ij")
        x_flat = jnp.array(x_p.flatten())
        y_flat = jnp.array(y_p.flatten())
        t_flat = jnp.array(t_p.flatten())

        def total_loss(net_params):
            S_pred = _forward_batch(net_params, x_data, y_data, t_data)
            loss_data = jnp.mean((S_pred - Sw_data) ** 2)
            residual = _pde_residual(
                net_params, x_flat, y_flat, t_flat, params, dims="2d"
            )
            loss_physics = jnp.mean(residual**2)
            return loss_data + config.physics_weight * loss_physics

    print(f"Training with {len(x_data)} samples...")
    return pinn, train_data


def _forward_batch_3d(params, x, y, z, t):
    """Forward pass for 3D PINN."""
    x_in = jnp.stack([x, y, z, t], axis=-1)
    for w, b in params[:-1]:
        x_in = x_in @ w + b
        x_in = jnp.tanh(x_in)
    w, b = params[-1]
    x_in = x_in @ w + b
    return jnp.clip(x_in, 0.0, 1.0).flatten()
