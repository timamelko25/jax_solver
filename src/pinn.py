"""Physics-Informed Neural Networks (PINN) for two-phase flow.

This module implements:
- MLP/ResNet architecture for saturation prediction
- Physics-informed loss with PDE residual
- Boundary condition enforcement
- Entropy conditions as soft constraints
"""

import jax.numpy as jnp
import numpy as onp
from jax import value_and_grad
from jax import random
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from .properties import fractional_flow, FlowParams


@dataclass
class PINNConfig:
    """Configuration for PINN training."""

    n_layers: int = 4
    hidden_dim: int = 64
    activation: str = "tanh"
    learning_rate: float = 1e-3
    n_iterations: int = 10000
    batch_size: int = 1000
    physics_weight: float = 0.1
    bc_weight: float = 1.0
    entropy_weight: float = 0.05


class SimpleMLP:
    """Simple MLP implementation without external dependencies.

    Uses jax.numpy for all operations.
    """

    def __init__(self, layer_sizes: list, activation: str = "tanh"):
        """Initialize MLP."""
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.params = self._init_params()

    def _init_params(self) -> list:
        """Initialize network parameters."""
        params = []
        key = random.PRNGKey(42)
        for i in range(len(self.layer_sizes) - 1):
            key, subkey = random.split(key)
            w = random.normal(
                subkey, (self.layer_sizes[i], self.layer_sizes[i + 1])
            ) * jnp.sqrt(2.0 / self.layer_sizes[i])
            b = jnp.zeros(self.layer_sizes[i + 1])
            params.append((w, b))
        return params

    def _activation(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply activation function."""
        if self.activation_name == "tanh":
            return jnp.tanh(x)
        elif self.activation_name == "relu":
            return jnp.maximum(0, x)
        elif self.activation_name == "sigmoid":
            return 1 / (1 + jnp.exp(-x))
        return jnp.tanh(x)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass."""
        for w, b in self.params[:-1]:
            x = x @ w + b
            x = self._activation(x)
        w, b = self.params[-1]
        x = x @ w + b
        return jnp.clip(x, 0.0, 1.0)


class SaturationPINN:
    """PINN for Buckley-Leverett equation."""

    def __init__(
        self, config: Optional[PINNConfig] = None, params=None, dims: str = "2d"
    ):
        if config is None:
            config = PINNConfig()
        self.config = config
        self.flow_params = params
        self.dims = dims
        input_dim = 3 if dims == "2d" else 2
        self.network = SimpleMLP(
            layer_sizes=[input_dim] + [config.hidden_dim] * config.n_layers + [1],
            activation=config.activation,
        )
        self.opt_state = None

    def predict(self, x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Predict saturation at (x, y, t)."""
        return self.network(jnp.stack([x, y, t], axis=-1)).flatten()

    def pde_residual(self, x: jnp.ndarray, t: jnp.ndarray, params) -> jnp.ndarray:
        """PDE residual: dS/dt + df/dx using finite differences."""
        dt_eps = 1e-4
        dx_eps = 1e-2
        S_pred = self.predict(x, t)
        S_plus_t = self.predict(x, t + dt_eps)
        S_plus_x = self.predict(x + dx_eps, t)
        S_minus_x = self.predict(x - dx_eps, t)
        fw_plus = fractional_flow(S_plus_x, params)
        fw_minus = fractional_flow(S_minus_x, params)
        dS_dt = (S_plus_t - S_pred) / dt_eps
        df_dx = (fw_plus - fw_minus) / (2 * dx_eps)
        return dS_dt + df_dx

    def boundary_loss(
        self, x_bc: jnp.ndarray, t_bc: jnp.ndarray, S_bc: jnp.ndarray
    ) -> float:
        """Boundary condition MSE loss."""
        return jnp.mean((self.predict(x_bc, t_bc) - S_bc) ** 2)

    def entropy_loss(self, x: jnp.ndarray, t: jnp.ndarray, params) -> float:
        """Entropy condition: enforce dS/dt >= 0 near shock."""
        dt_eps = 1e-4
        S_pred = self.predict(x, t)
        S_plus = self.predict(x, t + dt_eps)
        dS_dt = (S_plus - S_pred) / dt_eps
        violation = jnp.maximum(-dS_dt, 0)
        return jnp.mean(violation**2)

    def compute_loss(
        self,
        x_data: jnp.ndarray,
        t_data: jnp.ndarray,
        S_data: jnp.ndarray,
        x_physics: jnp.ndarray,
        t_physics: jnp.ndarray,
        x_bc: jnp.ndarray,
        t_bc: jnp.ndarray,
        S_bc: jnp.ndarray,
        params,
    ) -> Tuple[float, Dict]:
        """Total physics-informed loss."""
        loss_data = jnp.mean((self.predict(x_data, t_data) - S_data) ** 2)
        residual = self.pde_residual(x_physics, t_physics, params)
        loss_physics = jnp.mean(residual**2)
        loss_bc = self.boundary_loss(x_bc, t_bc, S_bc)
        loss_entropy = self.entropy_loss(x_physics, t_physics, params)
        total = (
            loss_data
            + self.config.physics_weight * loss_physics
            + self.config.bc_weight * loss_bc
            + self.config.entropy_weight * loss_entropy
        )
        return total, {
            "total": total,
            "data": loss_data,
            "physics": loss_physics,
            "bc": loss_bc,
            "entropy": loss_entropy,
        }


def create_training_data(
    n_points: int = 1000, dims: str = "2d", nx: int = 50, params=None
) -> Dict[str, jnp.ndarray]:
    """Create training data by running simulation directly.

    Args:
        n_points: Number of sample points to return
        dims: "1d" or "2d" (default: 2d)
        nx: Grid cells in x for 1D mode
        params: Flow parameters

    Returns:
        Dictionary with x, y (2D only), t, Sw arrays
    """
    if dims == "2d":
        return create_training_data_2d(n_points, params=params)
    else:
        return create_training_data_1d(n_points, nx, params=params)


def create_training_data_2d(
    n_points: int = 1000, use_spe10: bool = True, params=None
) -> Dict[str, jnp.ndarray]:
    """Create 2D training data using IMPES solver.

    Args:
        n_points: Number of sample points to return
        use_spe10: Use SPE10 real data (default True)
        params: Flow parameters

    Returns:
        Dictionary with x, y, t, Sw arrays
    """
    from .solver import SimulationConfig2D, IMPESSolver2D

    if params is None:
        params = FlowParams()

    if use_spe10:
        from .benchmarks import load_spe10_case1

        data = load_spe10_case1("data")
        nx, ny = int(data["nx"]), int(data["ny"])
        perm_x = data["perm_x"].T
        perm_y = data["perm_y"].T
    else:
        nx, ny = 20, 10
        perm_x = perm_y = None

    config = SimulationConfig2D(
        nx=nx,
        ny=ny,
        Lx=float(nx),
        Ly=float(ny),
        dt=0.001,
        t_max=0.005,
        save_interval=5,
    )
    solver = IMPESSolver2D(params, config, perm_x, perm_y)

    Sw_init = jnp.full((ny, nx), params.Swc)
    Sw_init = Sw_init.at[:, :3].set(1.0 - params.Sor)
    time_arr, Sw_hist, _ = solver.run(Sw_init)

    x = onp.array(solver.x)
    y = onp.array(solver.y)
    t_arr = onp.array(time_arr)

    x_grid, y_grid, t_grid = onp.meshgrid(x, y, t_arr, indexing="ij")
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    t_flat = t_grid.flatten()
    Sw_flat = onp.array(Sw_hist).flatten()

    n_total = min(n_points, len(x_flat))
    idx = onp.random.choice(len(x_flat), n_total, replace=False)

    return {
        "x": jnp.array(x_flat[idx]),
        "y": jnp.array(y_flat[idx]),
        "t": jnp.array(t_flat[idx]),
        "Sw": jnp.array(Sw_flat[idx]),
    }


def create_training_data_1d(
    n_points: int = 1000, nx: int = 50, params=None
) -> Dict[str, jnp.ndarray]:
    """Create 1D training data by running simulation directly.

    Uses a fixed dt to ensure non-zero time steps.
    """
    from .saturation_solver import simulate_1d_buckley_leverett

    if params is None:
        params = FlowParams()

    L = 100.0
    dx = L / nx
    u = 1e-4
    dt = 0.001
    t_max = 0.05
    nt = int(t_max / dt)

    Sw_init = jnp.full(nx, params.Swc + params.Sor)
    Sw_history = simulate_1d_buckley_leverett(
        Sw_init, u, params, dt, dx, nt, scheme="upwind", Sw_inj=1.0 - params.Sor
    )

    x = jnp.linspace(0, L, nx)
    t_array = jnp.linspace(0, t_max, Sw_history.shape[0])

    x_np = onp.array(x)
    t_np = onp.array(t_array)
    Sw_np = onp.array(Sw_history)

    x_grid, t_grid = onp.meshgrid(x_np, t_np, indexing="ij")
    x_flat = x_grid.flatten()
    t_flat = t_grid.flatten()
    S_flat = Sw_np.flatten()

    n_total = min(n_points, len(x_flat))
    idx = onp.random.choice(len(x_flat), n_total, replace=False)

    return {
        "x": jnp.array(x_flat[idx]),
        "t": jnp.array(t_flat[idx]),
        "Sw": jnp.array(S_flat[idx]),
    }


def _forward_pass(
    params: tuple, x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray
) -> jnp.ndarray:
    """Pure forward pass with explicit params. x, y, t must have same shape (N,)."""
    x_in = jnp.stack([x, y, t], axis=-1)
    for w, b in params[:-1]:
        x_in = x_in @ w + b
        x_in = jnp.tanh(x_in)
    w, b = params[-1]
    x_in = x_in @ w + b
    return jnp.clip(x_in, 0.0, 1.0)


def _forward_batch(
    params: tuple, x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray
) -> jnp.ndarray:
    return _forward_pass(params, x, y, t).flatten()


def _pde_residual_2d(
    params: tuple,
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
    fw_params,
    dt_eps=1e-4,
    dx_eps=1e-2,
    dy_eps=1e-2,
) -> jnp.ndarray:
    """2D PDE residual: dS/dt + d(f_w)/dx + d(f_w)/dy = 0"""
    S_pred = _forward_batch(params, x, y, t)

    fw_plus_x = fractional_flow(_forward_batch(params, x + dx_eps, y, t), fw_params)
    fw_minus_x = fractional_flow(_forward_batch(params, x - dx_eps, y, t), fw_params)

    fw_plus_y = fractional_flow(_forward_batch(params, x, y + dy_eps, t), fw_params)
    fw_minus_y = fractional_flow(_forward_batch(params, x, y - dy_eps, t), fw_params)

    S_plus_t = _forward_batch(params, x, y, t + dt_eps)

    dS_dt = (S_plus_t - S_pred) / dt_eps
    df_dx = (fw_plus_x - fw_minus_x) / (2 * dx_eps)
    df_dy = (fw_plus_y - fw_minus_y) / (2 * dy_eps)

    return dS_dt + df_dx + df_dy


def _pde_residual(
    params: tuple,
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
    fw_params,
    dims="2d",
    dt_eps=1e-4,
    dx_eps=1e-2,
) -> jnp.ndarray:
    """PDE residual. Supports 1D (y ignored) and 2D."""
    if dims == "1d":
        S_pred = _forward_batch(params, x, jnp.zeros_like(x), t)
        fw_plus = fractional_flow(
            _forward_batch(params, x + dx_eps, jnp.zeros_like(x), t), fw_params
        )
        fw_minus = fractional_flow(
            _forward_batch(params, x - dx_eps, jnp.zeros_like(x), t), fw_params
        )
        dS_dt = (
            _forward_batch(params, x, jnp.zeros_like(x), t + dt_eps) - S_pred
        ) / dt_eps
        df_dx = (fw_plus - fw_minus) / (2 * dx_eps)
        return dS_dt + df_dx
    else:
        return _pde_residual_2d(params, x, y, t, fw_params, dt_eps, dx_eps, dx_eps)


def train_pinn(
    train_data: Dict[str, jnp.ndarray],
    params,
    config: Optional[PINNConfig] = None,
    dims: str = "2d",
    verbose: bool = True,
) -> SaturationPINN:
    """Train PINN with physics-informed Adam optimizer.

    Args:
        train_data: Training data with x, y (2D), t, Sw
        params: Flow parameters
        config: PINN configuration
        dims: "1d" or "2d"
        verbose: Print progress
    """
    if config is None:
        config = PINNConfig()

    pinn = SaturationPINN(config, params)
    net_params = pinn.network.params

    x_data = train_data["x"]
    if dims == "2d":
        y_data = train_data["y"]
    else:
        y_data = jnp.zeros_like(x_data)
    t_data = train_data["t"]
    S_data = train_data["Sw"]

    if dims == "2d":
        nx_physics, ny_physics = 20, 10
        x_max, y_max = float(nx_physics), float(ny_physics)
        x_bc = jnp.zeros(10)
        y_bc = jnp.zeros(10)
        t_bc = jnp.linspace(0, 0.005, 10)
        S_bc = jnp.ones(10) * 0.8

        x_physics = jnp.linspace(0, x_max, nx_physics)
        y_physics = jnp.linspace(0, y_max, ny_physics)
        x_p, y_p, t_p = onp.meshgrid(x_physics, y_physics, t_bc, indexing="ij")
        x_physics = jnp.array(x_p.flatten())
        y_physics = jnp.array(y_p.flatten())
        t_physics = jnp.array(t_p.flatten())
    else:
        x_bc = jnp.zeros(10)
        y_bc = jnp.zeros(10)
        t_bc = jnp.linspace(0, 0.05, 10)
        S_bc = jnp.ones(10) * 0.8

        x_physics = jnp.linspace(0, 100, 100)
        y_physics = jnp.zeros_like(x_physics)
        t_physics = jnp.linspace(0, 0.05, 100)

    lr = config.learning_rate
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    pw = config.physics_weight
    bcw = config.bc_weight
    ew = config.entropy_weight

    m = [[jnp.zeros_like(w) for w in layer] for layer in net_params]
    v = [[jnp.zeros_like(w) for w in layer] for layer in net_params]

    def total_loss(net_params):
        S_pred = _forward_batch(net_params, x_data, y_data, t_data)
        loss_data = jnp.mean((S_pred - S_data) ** 2)

        residual = _pde_residual(
            net_params, x_physics, y_physics, t_physics, params, dims=dims
        )
        loss_physics = jnp.mean(residual**2)

        S_bc_pred = _forward_batch(net_params, x_bc, y_bc, t_bc)
        loss_bc = jnp.mean((S_bc_pred - S_bc) ** 2)

        dt_eps = 1e-4
        S_ph_pred = _forward_batch(net_params, x_physics, y_physics, t_physics)
        S_ph_plus = _forward_batch(net_params, x_physics, y_physics, t_physics + dt_eps)
        dS_dt = (S_ph_plus - S_ph_pred) / dt_eps
        loss_entropy = jnp.mean(jnp.maximum(-dS_dt, 0) ** 2)

        return loss_data + pw * loss_physics + bcw * loss_bc + ew * loss_entropy

    prev_loss = float("inf")

    for iteration in range(config.n_iterations):
        loss, grads = value_and_grad(total_loss)(net_params)

        m = [
            [beta1 * mw + (1 - beta1) * gw for mw, gw in zip(ml, gl)]
            for ml, gl in zip(m, grads)
        ]
        v = [
            [beta2 * vw + (1 - beta2) * gw**2 for vw, gw in zip(vl, gl)]
            for vl, gl in zip(v, grads)
        ]

        t_iter = iteration + 1
        m_hat = [[mw / (1 - beta1**t_iter) for mw in ml] for ml in m]
        v_hat = [[vw / (1 - beta2**t_iter) for vw in vl] for vl in v]

        net_params = tuple(
            tuple(
                pw_ - lr * mh / (jnp.sqrt(vh) + eps)
                for pw_, mh, vh in zip(pl, mhl, vhl)
            )
            for pl, mhl, vhl in zip(net_params, m_hat, v_hat)
        )

        if verbose and (iteration % 1000 == 0 or iteration == config.n_iterations - 1):
            delta = abs(loss - prev_loss)
            print(
                f"Iter {iteration:5d}: loss={float(loss):.6f} "
                f"(delta={float(delta):.2e})"
            )
            prev_loss = float(loss)

    pinn.network.params = net_params

    if verbose:
        print("Training completed!")

    return pinn


def train_pinn_inverse(
    train_data: Dict[str, jnp.ndarray],
    params_template: FlowParams,
    config: Optional[PINNConfig] = None,
    learn_params: list = ["M"],
    dims: str = "2d",
    verbose: bool = True,
) -> tuple[SaturationPINN, FlowParams]:
    """Inverse PINN - learn parameters from observed data (Zhang et al., Eq. 5).

    Loss = L_PDE + L_data

    Args:
        train_data: Training data with x, y (2D), t, Sw
        params_template: Initial flow parameters (M, N will be overridden)
        config: PINN configuration
        learn_params: Parameters to learn ["M"], ["N"], or ["M", "N"]
        dims: "1d" or "2d"
        verbose: Print progress

    Returns:
        Tuple of (trained_pinn, learned_params)
    """
    if config is None:
        config = PINNConfig()

    M_init = (
        params_template.kro0
        / params_template.mu_w
        * params_template.mu_o
        / params_template.krw0
    )
    N_init = params_template.N_sin_alpha

    if "M" in learn_params:
        M_log = jnp.log(M_init + 1e-10)
    else:
        M_log = jnp.log(M_init + 1e-10)

    if "N" in learn_params:
        N_param = N_init
    else:
        N_param = N_init

    pinn = SaturationPINN(config, params_template, dims)
    net_params = pinn.network.params

    x_data = train_data["x"]
    if dims == "2d":
        y_data = train_data["y"]
    else:
        y_data = jnp.zeros_like(x_data)
    t_data = train_data["t"]
    S_data = train_data["Sw"]

    if dims == "2d":
        nx_physics, ny_physics = 20, 10
        x_max, y_max = float(nx_physics), float(ny_physics)
        x_physics = jnp.linspace(0, x_max, nx_physics)
        y_physics = jnp.linspace(0, y_max, ny_physics)
        t_physics = jnp.linspace(0, 0.005, 10)
    else:
        x_physics = jnp.linspace(0, 100, 100)
        y_physics = jnp.zeros(100)
        t_physics = jnp.linspace(0, 0.05, 100)

    lr = config.learning_rate
    lr_param = 1e-3
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    pw = config.physics_weight

    m = [[jnp.zeros_like(w) for w in layer] for layer in net_params]
    v = [[jnp.zeros_like(w) for w in layer] for layer in net_params]

    m_M = 0.0
    v_M = 0.0
    m_N = 0.0
    v_N = 0.0

    def get_params_from_log(M_log_val, N_val):
        M_val = jnp.exp(M_log_val)
        kro0_eff = (
            M_val * params_template.mu_w / params_template.mu_o * params_template.krw0
        )
        return FlowParams(
            k=params_template.k,
            phi=params_template.phi,
            mu_w=params_template.mu_w,
            mu_o=params_template.mu_o,
            krw0=params_template.krw0,
            kro0=kro0_eff,
            n_w=params_template.n_w,
            n_o=params_template.n_o,
            Swc=params_template.Swc,
            Sor=params_template.Sor,
            N_sin_alpha=N_val,
            D_leading=params_template.D_leading,
            D_trailing=params_template.D_trailing,
            rho_o=params_template.rho_o,
            rho_w=params_template.rho_w,
        )

    def total_loss_inverse(net_params, M_log_val, N_val):
        current_p = get_params_from_log(M_log_val, N_val)

        S_pred = _forward_batch(net_params, x_data, y_data, t_data)
        loss_data = jnp.mean((S_pred - S_data) ** 2)

        x_p, y_p, t_p = onp.meshgrid(
            onp.array(x_physics),
            onp.array(y_physics),
            onp.array(t_physics),
            indexing="ij",
        )
        x_flat = jnp.array(x_p.flatten())
        y_flat = jnp.array(y_p.flatten())
        t_flat = jnp.array(t_p.flatten())

        residual = _pde_residual(
            net_params, x_flat, y_flat, t_flat, current_p, dims=dims
        )
        loss_physics = jnp.mean(residual**2)

        return loss_data + pw * loss_physics

    for iteration in range(config.n_iterations):
        loss_full, grads_full = value_and_grad(total_loss_inverse, argnums=(0, 1, 2))(
            net_params, M_log, N_param
        )

        if isinstance(loss_full, tuple):
            loss_val = loss_full[0]
            grads_net = grads_full[0]
            grad_M = grads_full[1]
            grad_N = grads_full[2]
        else:
            loss_val = loss_full
            grads_net = grads_full[0]
            grad_M = grads_full[1]
            grad_N = grads_full[2]

        m = [
            [beta1 * mw + (1 - beta1) * gw for mw, gw in zip(ml, gl)]
            for ml, gl in zip(m, grads_net)
        ]
        v = [
            [beta2 * vw + (1 - beta2) * gw**2 for vw, gw in zip(vl, gl)]
            for vl, gl in zip(v, grads_net)
        ]

        t_iter = iteration + 1
        m_hat = [[mw / (1 - beta1**t_iter) for mw in ml] for ml in m]
        v_hat = [[vw / (1 - beta2**t_iter) for vw in vl] for vl in v]

        net_params = tuple(
            tuple(
                pw_ - lr * mh / (jnp.sqrt(vh) + eps)
                for pw_, mh, vh in zip(pl, mhl, vhl)
            )
            for pl, mhl, vhl in zip(net_params, m_hat, v_hat)
        )

        if "M" in learn_params:
            m_M = beta1 * m_M + (1 - beta1) * grad_M
            v_M = beta2 * v_M + (1 - beta2) * grad_M**2
            m_M_hat = m_M / (1 - beta1**t_iter)
            v_M_hat = v_M / (1 - beta2**t_iter)
            M_log = M_log - lr_param * m_M_hat / (jnp.sqrt(v_M_hat) + eps)

        if "N" in learn_params:
            m_N = beta1 * m_N + (1 - beta1) * grad_N
            v_N = beta2 * v_N + (1 - beta2) * grad_N**2
            m_N_hat = m_N / (1 - beta1**t_iter)
            v_N_hat = v_N / (1 - beta2**t_iter)
            N_param = N_param - lr_param * m_N_hat / (jnp.sqrt(v_N_hat) + eps)

        if verbose and (iteration % 1000 == 0 or iteration == config.n_iterations - 1):
            M_learned = jnp.exp(M_log)
            print(
                f"Iter {iteration:5d}: loss={float(loss_val):.6f} "
                f"M={float(M_learned):.4f} N={float(N_param):.4f}"
            )

    pinn.network.params = net_params
    learned_params = get_params_from_log(M_log, N_param)

    if verbose:
        M_final = jnp.exp(M_log)
        print(
            f"Training completed! Learned: M={float(M_final):.6f}, N={float(N_param):.6f}"
        )

    return pinn, learned_params


def evaluate_pinn(
    pinn: SaturationPINN, x_test: jnp.ndarray, t_test: jnp.ndarray
) -> jnp.ndarray:
    """Evaluate trained PINN."""
    return pinn.predict(x_test, t_test)
