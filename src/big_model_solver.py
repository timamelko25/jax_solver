import jax.numpy as jnp
from typing import Optional, Dict, Any

from .benchmarks import load_big_model
from .properties import FlowParams
from .solver import IMPESSolver2D, IMPESSolver3D, SimulationConfig2D, SimulationConfig3D


def solve_big_model_2d(
    data_dir: str = "data/BIG_MODEL_12_09_1",
    k_layer: int = 21,
    config: Optional[SimulationConfig2D] = None,
) -> Dict[str, Any]:
    data = load_big_model(data_dir)

    nx_full, ny_full, nz = data["nx"], data["ny"], data["nz"]
    assert 0 <= k_layer < nz, f"k_layer={k_layer} out of range [0, {nz})"

    if config is None:
        config = SimulationConfig2D(
            nx=nx_full,
            ny=ny_full,
            Lx=float(nx_full),
            Ly=float(ny_full),
            dt=0.001,
            t_max=0.05,
            q_injection=1e-4,
            scheme="upwind",
            save_interval=10,
        )

    nx = min(config.nx, nx_full)
    ny = min(config.ny, ny_full)

    perm_x_3d = data["perm_x"]
    perm_y_3d = data["perm_y"]
    porosity_3d = data["porosity"]

    perm_x_2d = perm_x_3d[:nx, :ny, k_layer].T
    perm_y_2d = perm_y_3d[:nx, :ny, k_layer].T
    poro_2d = porosity_3d[:nx, :ny, k_layer].T

    mu_w = 1e-3
    mu_o = 1e-2
    k_avg = float(jnp.mean(perm_x_2d))
    phi_avg = float(jnp.mean(poro_2d))

    params = FlowParams(
        k=k_avg * 1e-15,
        phi=phi_avg,
        mu_w=mu_w,
        mu_o=mu_o,
        Swc=0.2,
        Sor=0.2,
    )

    solver = IMPESSolver2D(
        params=params,
        config=config,
        perm_x=perm_x_2d * 1e-15,
        perm_y=perm_y_2d * 1e-15,
    )

    Sw_init = jnp.full((config.ny, config.nx), params.Swc)
    time, Sw_history, p_history = solver.run(Sw_init)

    return {
        "time": time,
        "Sw_history": Sw_history,
        "p_history": p_history,
        "x": solver.x,
        "y": solver.y,
        "perm_x_2d": perm_x_2d,
        "perm_y_2d": perm_y_2d,
        "porosity_2d": poro_2d,
        "k_layer": k_layer,
        "params": params,
        "config": config,
    }


def plot_solution_2d(
    result: Dict[str, Any],
    save_path: str = "outputs/big_model_solution.png",
) -> None:
    import matplotlib.pyplot as plt

    Sw_history = result["Sw_history"]
    p_history = result["p_history"]
    x = result["x"]
    y = result["y"]
    k_layer = result["k_layer"]

    n_snapshots = min(4, Sw_history.shape[0])
    indices = jnp.linspace(0, Sw_history.shape[0] - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(2, n_snapshots, figsize=(4 * n_snapshots, 8))
    if n_snapshots == 1:
        axes = axes.reshape(2, 1)

    for i, idx in enumerate(indices):
        t = float(result["time"][idx])

        im1 = axes[0, i].pcolormesh(
            x, y, Sw_history[idx], shading="auto", cmap="Blues", vmin=0.2, vmax=0.8
        )
        axes[0, i].set_title(f"Sw (t={t:.3f})")
        axes[0, i].set_xlabel("x")
        axes[0, i].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0, i])

        im2 = axes[1, i].pcolormesh(
            x, y, p_history[idx] / 1e5, shading="auto", cmap="Reds"
        )
        axes[1, i].set_title(f"P (t={t:.3f}) [bar]")
        axes[1, i].set_xlabel("x")
        axes[1, i].set_ylabel("y")
        plt.colorbar(im2, ax=axes[1, i])

    fig.suptitle(f"BIG_MODEL 2D Slice (k={k_layer}) — IMPES Solution", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


def solve_big_model_3d(
    data_dir: str = "data/BIG_MODEL_12_09_1",
    config: Optional[SimulationConfig3D] = None,
) -> Dict[str, Any]:
    data = load_big_model(data_dir)

    nx, ny, nz = data["nx"], data["ny"], data["nz"]

    perm_x_3d = data["perm_x"]  # (nx, ny, nz)
    perm_y_3d = data["perm_y"]
    perm_z_3d = data["perm_z"]
    porosity_3d = data["porosity"]

    # data['perm_x'] has shape (nx, ny, nz) = (122, 183, 43)
    # For solver we need (nz, ny, nx) = (43, 183, 122)
    perm_x = perm_x_3d.T  # (43, 183, 122)
    perm_y = perm_y_3d.T
    perm_z = perm_z_3d.T
    porosity = porosity_3d.T

    if config is None:
        # nz=43, ny=183, nx=122 for solver (nz, ny, nx)
        config = SimulationConfig3D(
            nz=43,
            ny=183,
            nx=122,
            Lz=float(nz),
            Ly=float(ny),
            Lx=float(nx),
            dt=0.001,
            t_max=0.01,
            q_injection=1e-4,
            scheme="upwind",
            save_interval=10,
        )

    mu_w = 1e-3
    mu_o = 1e-2
    k_avg = float(jnp.mean(perm_x))
    phi_avg = float(jnp.mean(porosity))

    params = FlowParams(
        k=k_avg * 1e-15,
        phi=phi_avg,
        mu_w=mu_w,
        mu_o=mu_o,
        Swc=0.2,
        Sor=0.2,
    )

    solver = IMPESSolver3D(
        params=params,
        config=config,
        perm_x=perm_x * 1e-15,
        perm_y=perm_y * 1e-15,
        perm_z=perm_z * 1e-15,
    )

    Sw_init = jnp.full((nz, ny, nx), params.Swc)
    time, Sw_history, p_history = solver.run(Sw_init)

    return {
        "time": time,
        "Sw_history": Sw_history,
        "p_history": p_history,
        "x": solver.x,
        "y": solver.y,
        "z": solver.z,
        "perm_x": perm_x,
        "perm_y": perm_y,
        "perm_z": perm_z,
        "porosity": porosity,
        "params": params,
        "config": config,
    }


def plot_solution_3d(
    result: Dict[str, Any],
    save_path: str = "outputs/big_model_3d_solution.png",
    slice_idx: int = 21,
    axis: str = "z",
) -> None:
    import matplotlib.pyplot as plt

    Sw_history = result["Sw_history"]
    p_history = result["p_history"]
    n_snapshots = min(4, Sw_history.shape[0])
    indices = jnp.linspace(0, Sw_history.shape[0] - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(2, n_snapshots, figsize=(4 * n_snapshots, 8))
    if n_snapshots == 1:
        axes = axes.reshape(2, 1)

    for i, idx in enumerate(indices):
        t = float(result["time"][idx])

        if axis == "z":
            Sw_slice = Sw_history[idx, slice_idx, :, :]
            P_slice = p_history[idx, slice_idx, :, :] / 1e5
            x, y = result["x"], result["y"]
            xlabel, ylabel = "x", "y"
            title_suffix = f"z={slice_idx}"
        elif axis == "y":
            Sw_slice = Sw_history[idx, :, slice_idx, :]
            P_slice = p_history[idx, :, slice_idx, :] / 1e5
            x, y = result["x"], result["z"]
            xlabel, ylabel = "x", "z"
            title_suffix = f"y={slice_idx}"
        else:
            Sw_slice = Sw_history[idx, :, :, slice_idx]
            P_slice = p_history[idx, :, :, slice_idx] / 1e5
            x, y = result["y"], result["z"]
            xlabel, ylabel = "y", "z"
            title_suffix = f"x={slice_idx}"

        im1 = axes[0, i].pcolormesh(
            x, y, Sw_slice, shading="auto", cmap="Blues", vmin=0.2, vmax=0.8
        )
        axes[0, i].set_title(f"Sw (t={t:.3f}, {title_suffix})")
        axes[0, i].set_xlabel(xlabel)
        axes[0, i].set_ylabel(ylabel)
        plt.colorbar(im1, ax=axes[0, i])

        im2 = axes[1, i].pcolormesh(x, y, P_slice, shading="auto", cmap="Reds")
        axes[1, i].set_title(f"P (t={t:.3f}, {title_suffix}) [bar]")
        axes[1, i].set_xlabel(xlabel)
        axes[1, i].set_ylabel(ylabel)
        plt.colorbar(im2, ax=axes[1, i])

    fig.suptitle(f"BIG_MODEL 3D Solution (axis={axis}, slice={slice_idx})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")
