#!/usr/bin/env python3
"""Main entry point for SPEC_JAX - Two-phase flow solver in porous media.

Usage:
    python main.py --mode run --config configs/default.yaml
    python main.py --mode benchmark --dataset spe10
    python main.py --mode pinn --epochs 1000
    python main.py --mode plot --output results.png
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt

from src.properties import FlowParams, welge_construct
from src.saturation_solver import simulate_1d_buckley_leverett
from src.dataset import (
    generate_2d_heterogeneous,
)
from src.benchmarks import load_spe10_case1
from src.solver import SimulationConfig2D, IMPESSolver2D
from src.visualization import (
    plot_saturation_profile,
    plot_fractional_flow,
    plot_2d_saturation,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_simulation(config: dict):
    """Run two-phase flow simulation."""
    print("=" * 60)
    print("Running Two-Phase Flow Simulation")
    print("=" * 60)

    grid = config.get("grid", {})
    physics = config.get("physics", {})
    time_cfg = config.get("time", {})
    boundary = config.get("boundary", {})
    scheme = config.get("scheme", {})

    params = FlowParams(
        k=physics.get("permeability", 1e-12),
        phi=physics.get("porosity", 0.2),
        mu_w=physics.get("water_viscosity", 1e-3),
        mu_o=physics.get("oil_viscosity", 1e-2),
        krw0=physics.get("krw0", 0.3),
        kro0=physics.get("kro0", 0.8),
        n_w=physics.get("n_w", 2.0),
        n_o=physics.get("n_o", 2.0),
        Swc=physics.get("Swc", 0.2),
        Sor=physics.get("Sor", 0.2),
    )

    nx = grid.get("nx", 100)
    L = grid.get("L", 100.0)
    dx = L / nx

    dt = time_cfg.get("dt", 0.0005)
    t_max = time_cfg.get("t_max", 0.1)
    nt = int(t_max / dt)

    Sw_inj = boundary.get("Sw_inlet", 0.8)
    Sw_init = jnp.full(nx, params.Swc + params.Sor)

    u = 1e-4
    scheme_type = scheme.get("type", "upwind")

    print(f"Grid: {nx} cells, dx = {dx:.3f} m")
    print(f"Time: {nt} steps, dt = {dt:.2e} s")
    print(f"Scheme: {scheme_type}")
    print(f"Mobility ratio: {params.mu_o / params.mu_w:.2f}")

    Sw_history = simulate_1d_buckley_leverett(
        Sw_init, u, params, dt, dx, nt, scheme=scheme_type, Sw_inj=Sw_inj
    )

    x = jnp.linspace(0, L, nx)
    t_array = jnp.linspace(0, t_max, nt + 1)

    output_cfg = config.get("output", {})
    if output_cfg.get("plot", True):
        plot_saturation_profile(
            x,
            Sw_history,
            t_array,
            title="Two-Phase Flow Simulation",
            save_path=output_cfg.get("save_path", "outputs/") + "saturation.png",
        )
        print("Saved: saturation.png")

    print("Simulation completed successfully!")
    return x, Sw_history, t_array


def run_simulation_2d(args):
    """Run 2D two-phase flow simulation."""
    print("=" * 60)
    print("Running 2D Two-Phase Flow Simulation")
    print("=" * 60)

    if hasattr(args, "spe10") and args.spe10:
        print("Loading SPE10 data...")
        spe10_data = load_spe10_case1("data")
        nx = spe10_data["nx"]
        ny = spe10_data["ny"]
        Lx = float(nx)
        Ly = float(ny)
        perm_x = spe10_data["perm_x"].T
        perm_y = spe10_data["perm_y"].T
        print(f"Grid: {nx} x {ny}")
        print(
            f"Perm range: {float(jnp.min(perm_x)):.2f} - {float(jnp.max(perm_x)):.2f} mD"
        )
    else:
        nx = args.nx
        ny = args.ny
        Lx = args.Lx
        Ly = args.Ly
        perm_x = None
        perm_y = None
        print(f"Grid: {nx} x {ny}, Lx={Lx}, Ly={Ly}")

    config = SimulationConfig2D(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        dt=args.dt,
        t_max=args.t_max,
        p_inlet=args.p_inlet,
        p_outlet=args.p_outlet,
        q_injection=args.q_injection,
        save_interval=args.save_interval,
    )

    params = FlowParams()
    solver = IMPESSolver2D(params, config, perm_x, perm_y)

    Sw_init = jnp.full((ny, nx), params.Swc)
    Sw_init = Sw_init.at[:, :3].set(1.0 - params.Sor)

    print(f"Initializing: Sw={float(params.Swc):.2f} -> Sw_inj={1.0 - params.Sor:.2f}")
    print(f"Running {int(args.t_max / args.dt)} time steps...")

    time_arr, Sw_history, p_history = solver.run(Sw_init)

    print(f"Completed: {len(time_arr)} snapshots saved")
    print(
        f"Sw range: [{float(jnp.min(Sw_history)):.4f}, {float(jnp.max(Sw_history)):.4f}]"
    )

    output_dir = "outputs/"
    os.makedirs(output_dir, exist_ok=True)

    if len(Sw_history) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_grid, y_grid = onp.meshgrid(solver.x, solver.y)
        Sw_plot = Sw_history[-1]
        if Sw_plot.shape != x_grid.shape:
            Sw_plot = Sw_plot.T
        im = ax.pcolormesh(x_grid, y_grid, Sw_plot, cmap="viridis", shading="gouraud")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("2D Saturation (final)")
        plt.colorbar(im, label="Sw")
        plt.tight_layout()
        plt.savefig(output_dir + "saturation_2d_final.png", dpi=150)
        print("Saved: saturation_2d_final.png")
        plt.close()

    print("2D Simulation completed successfully!")


def run_benchmark(dataset: str, case: str = "1"):
    """Run benchmark on specified dataset."""
    print("=" * 60)
    print(f"Running Benchmark: {dataset} (Case {case})")
    print("=" * 60)

    if dataset == "spe10":
        if case == "1":
            print("Loading SPE10 Case 1 (2D cross-section) from data/perm_case1.dat...")
            data = load_spe10_case1("data")
            print(f"Grid: {data['nx']} x {data['ny']}")

            perm_stats = {
                "min": float(jnp.min(data["perm_x"])),
                "max": float(jnp.max(data["perm_x"])),
                "mean": float(jnp.mean(data["perm_x"])),
            }
            print(
                f"Permeability range: {perm_stats['min']:.2f} - {perm_stats['max']:.2f} mD"
            )
            print(
                f"Porosity range: {float(jnp.min(data['porosity'])):.4f} - {float(jnp.max(data['porosity'])):.4f}"
            )

            plot_2d_saturation(
                data["x"],
                data["y"],
                data["perm_x"] / jnp.max(data["perm_x"]),
                title="SPE10 Case 1 - Normalized Permeability",
                save_path="outputs/spe10_case1_perm.png",
            )
            print("Saved: outputs/spe10_case1_perm.png")
        else:
            print("Loading SPE10 Case 2 (3D full model)...")
            print("Case 2 not yet implemented - using synthetic data")
            data = generate_2d_heterogeneous(
                nx=60, ny=30, Lx=100.0, Ly=40.0, t_max=0.005, dt=0.0001
            )
            print(f"Grid: {len(data['x'])} x {len(data['y'])}")

            perm_stats = {
                "min": float(jnp.min(data["perm_x"])),
                "max": float(jnp.max(data["perm_x"])),
                "mean": float(jnp.mean(data["perm_x"])),
            }
            print(
                f"Permeability range: {perm_stats['min']:.2e} - {perm_stats['max']:.2e}"
            )

            plot_2d_saturation(
                data["x"],
                data["y"],
                data["Sw"][0],
                title="SPE10-like Heterogeneous Media",
                save_path="outputs/spe10_initial.png",
            )
            print("Saved: outputs/spe10_initial.png")

    elif dataset == "analytical":
        params = FlowParams()
        Sw_inj = 1.0 - params.Sor
        Sw_init = params.Swc + params.Sor

        S_shock, v_shock = welge_construct(Sw_inj, Sw_init, params)
        print("Welge Construction Results:")
        print(f"  Shock saturation: {S_shock:.4f}")
        print(f"  Shock velocity: {v_shock:.4e} m/s")

        Sw_range = jnp.linspace(params.Swc, 1.0 - params.Sor, 100)
        plot_fractional_flow(Sw_range, params, save_path="outputs/fractional_flow.png")
        print("Saved: fractional_flow.png")

    else:
        print(f"Unknown dataset: {dataset}")
        print("Available: spe10, analytical")

    print("Benchmark completed!")


def run_pinn(
    epochs: int = 1000, dims: str = "2d", inverse: bool = False, learn: str = "M"
):
    """Run PINN training."""
    print("=" * 60)
    if inverse:
        print(f"Running INVERSE PINN ({dims.upper()}) for {epochs} epochs")
    else:
        print(f"Running PINN ({dims.upper()}) for {epochs} epochs")
    print("=" * 60)

    print("Generating training data...")
    from src.pinn import create_training_data, train_pinn, PINNConfig

    data = create_training_data(n_points=1000, dims=dims)

    if dims == "2d":
        print(f"Training samples: {len(data['x'])} (2D with SPE10)")
    else:
        print(f"Training samples: {len(data['x'])} (1D)")

    config = PINNConfig(n_iterations=epochs)

    if inverse:
        learn_list = learn.split(",")
        print(f"Running Inverse PINN, learning: {learn_list}")
        from src.pinn import train_pinn_inverse

        params_template = FlowParams()
        pinn, learned_params = train_pinn_inverse(
            data, params_template, config, learn_list, dims
        )
        print(
            f"Learned params: kro0={learned_params.kro0}, N_sin_alpha={learned_params.N_sin_alpha}"
        )
    else:
        params = FlowParams()
        pinn = train_pinn(data, params, config, dims=dims)
    return pinn


def run_plot(output_path: str):
    """Plot existing results."""
    print("=" * 60)
    print("Plotting Results")
    print("=" * 60)

    print(f"Output path: {output_path}")
    print("Note: This mode requires pre-existing simulation results")


def main():
    parser = argparse.ArgumentParser(
        description="SPEC_JAX - Two-phase flow solver in porous media"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="run",
        choices=["run", "run-2d", "benchmark", "pinn", "plot", "demo"],
        help="Operation mode",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="spe10",
        choices=["spe10", "analytical"],
        help="Dataset for benchmark",
    )

    parser.add_argument(
        "--case",
        type=str,
        default="1",
        choices=["1", "2"],
        help="SPE10 case number (1: 2D cross-section, 2: 3D full)",
    )

    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of training epochs for PINN"
    )

    parser.add_argument(
        "--dims",
        type=str,
        default="2d",
        choices=["1d", "2d"],
        help="Dimensionality for PINN (default: 2d with SPE10)",
    )

    parser.add_argument(
        "--inverse",
        action="store_true",
        help="Use inverse PINN to learn parameters from data",
    )

    parser.add_argument(
        "--learn",
        type=str,
        default="M",
        help="Parameters to learn in inverse PINN: M, N, or M,N",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results.png",
        help="Output file path for plotting",
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=50,
        help="Grid cells in x-direction for 2D",
    )

    parser.add_argument(
        "--ny",
        type=int,
        default=30,
        help="Grid cells in y-direction for 2D",
    )

    parser.add_argument(
        "--Lx",
        type=float,
        default=50.0,
        help="Domain size in x-direction",
    )

    parser.add_argument(
        "--Ly",
        type=float,
        default=30.0,
        help="Domain size in y-direction",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Time step for 2D simulation",
    )

    parser.add_argument(
        "--t-max",
        type=float,
        default=0.005,
        help="Maximum simulation time",
    )

    parser.add_argument(
        "--p-inlet",
        type=float,
        default=1e5,
        help="Inlet pressure",
    )

    parser.add_argument(
        "--p-outlet",
        type=float,
        default=0.0,
        help="Outlet pressure",
    )

    parser.add_argument(
        "--q-injection",
        type=float,
        default=1e-4,
        help="Injection rate",
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save interval for history",
    )

    parser.add_argument(
        "--spe10",
        action="store_true",
        help="Use SPE10 data for 2D simulation",
    )

    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    if args.mode == "run":
        if os.path.exists(args.config):
            config = load_config(args.config)
            run_simulation(config)
        else:
            print(f"Config file not found: {args.config}")
            print("Running with default parameters...")
            config = {
                "grid": {"nx": 100, "L": 100.0},
                "physics": FlowParams().__dict__,
                "time": {"dt": 0.0005, "t_max": 0.05},
                "scheme": {"type": "upwind"},
                "output": {"plot": True, "save_path": "outputs/"},
            }
            run_simulation(config)

    elif args.mode == "benchmark":
        run_benchmark(args.dataset, args.case)

    elif args.mode == "run-2d":
        run_simulation_2d(args)

    elif args.mode == "pinn":
        run_pinn(args.epochs, args.dims, args.inverse, args.learn)

    elif args.mode == "plot":
        run_plot(args.output)

    elif args.mode == "demo":
        print("Running demonstration mode...")
        from examples.run_1d import run_all_demos

        run_all_demos()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
