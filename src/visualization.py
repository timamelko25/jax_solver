"""Visualization utilities for two-phase flow simulation results."""

import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as jnp
from typing import Optional, List


def plot_saturation_profile(
    x: jnp.ndarray,
    Sw_history: jnp.ndarray,
    time_array: Optional[jnp.ndarray] = None,
    title: str = "Water Saturation Profile",
    xlabel: str = "Position (m)",
    ylabel: str = "Water Saturation",
    save_path: Optional[str] = None,
    show_shock: bool = True,
    analytical_x: Optional[jnp.ndarray] = None,
    analytical_Sw: Optional[jnp.ndarray] = None,
) -> plt.Figure:
    """Plot 1D saturation profile over time.

    Args:
        x: Spatial coordinates
        Sw_history: Saturation history (n_times, n_space)
        time_array: Time at each snapshot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure
        show_shock: Show shock front marker
        analytical_x: Analytical solution x
        analytical_Sw: Analytical solution saturation

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_times = Sw_history.shape[0]
    if time_array is None:
        time_array = jnp.linspace(0, 1, n_times)

    n_curves = min(n_times, 8)
    idx = onp.linspace(0, n_times - 1, n_curves).astype(int)

    cmap = plt.cm.viridis
    colors = cmap(onp.linspace(0, 1, n_curves))

    for i, t_idx in enumerate(idx):
        ax.plot(
            x,
            Sw_history[t_idx],
            color=colors[i],
            linewidth=2,
            label=f"t = {float(time_array[t_idx]):.3f}",
        )

    if analytical_x is not None and analytical_Sw is not None:
        ax.plot(analytical_x, analytical_Sw, "r--", linewidth=2, label="Analytical")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    if show_shock:
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_fractional_flow(
    Sw_range: jnp.ndarray, params, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot fractional flow curve and its derivative.

    Args:
        Sw_range: Saturation range
        params: Flow parameters
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    from .properties import fractional_flow, df_dsaturation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    f_w = fractional_flow(Sw_range, params)
    df_dS = df_dsaturation(Sw_range, params)

    ax1.plot(Sw_range, f_w, "b-", linewidth=2)
    ax1.set_xlabel("Water Saturation", fontsize=12)
    ax1.set_ylabel("Fractional Flow $f_w$", fontsize=12)
    ax1.set_title("Fractional Flow Curve", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(Sw_range, df_dS, "r-", linewidth=2)
    ax2.set_xlabel("Water Saturation", fontsize=12)
    ax2.set_ylabel("$df_w/dS_w$", fontsize=12)
    ax2.set_title("Fractional Flow Derivative", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_2d_saturation(
    x: jnp.ndarray,
    y: jnp.ndarray,
    Sw: jnp.ndarray,
    title: str = "2D Water Saturation",
    save_path: Optional[str] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> plt.Figure:
    """Plot 2D saturation field.

    Args:
        x: X coordinates
        y: Y coordinates
        Sw: 2D saturation field
        title: Plot title
        save_path: Path to save figure
        vmin, vmax: Color scale limits

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    X, Y = onp.meshgrid(x, y)

    im = ax.pcolormesh(
        X, Y, Sw.T, cmap="viridis", vmin=vmin, vmax=vmax, shading="gouraud"
    )

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Water Saturation", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_mobility_ratio_comparison(
    datasets: dict, time_idx: int = -1, save_path: Optional[str] = None
) -> plt.Figure:
    """Compare saturation profiles for different mobility ratios.

    Args:
        datasets: Dictionary of datasets with mobility ratio results
        time_idx: Time index to plot
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in datasets.items():
        if "Sw_test" in data:
            Sw = data["Sw_test"]
        elif "Sw" in data:
            Sw = data["Sw"]
        else:
            continue

        if len(Sw.shape) == 2:
            ax.plot(data["x"], Sw[time_idx], linewidth=2, label=label)
        else:
            ax.plot(data["x"], Sw[time_idx], linewidth=2, label=label)

    ax.set_xlabel("Position (m)", fontsize=12)
    ax.set_ylabel("Water Saturation", fontsize=12)
    ax.set_title("Effect of Mobility Ratio on Displacement", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_convergence(
    errors: List[float], label: str = "L2 Error", save_path: Optional[str] = None
) -> plt.Figure:
    """Plot convergence history.

    Args:
        errors: List of error values
        label: Label for the error
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(errors, "b-", linewidth=2, marker="o", label=label)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("Convergence History", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pressure_field(
    x: jnp.ndarray,
    p_history: jnp.ndarray,
    time_array: Optional[jnp.ndarray] = None,
    title: str = "Pressure Field",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot pressure distribution over time.

    Args:
        x: Spatial coordinates
        p_history: Pressure history
        time_array: Time array
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_times = p_history.shape[0]
    if time_array is None:
        time_array = jnp.linspace(0, 1, n_times)

    n_curves = min(n_times, 5)
    idx = onp.linspace(0, n_times - 1, n_curves).astype(int)

    for t_idx in idx:
        ax.plot(
            x,
            p_history[t_idx],
            linewidth=2,
            label=f"t = {float(time_array[t_idx]):.3f}",
        )

    ax.set_xlabel("Position (m)", fontsize=12)
    ax.set_ylabel("Pressure (Pa)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
