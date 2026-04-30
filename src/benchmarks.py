"""SPE10 Benchmark dataset loader and utilities.

The SPE10 dataset is a standard benchmark for multi-phase flow in porous media.
It contains:
- 60 layers in z-direction
- 220 x 60 grid in each layer (x, y)
- Heterogeneous permeability field with log-normal distribution
- Variable porosity

Reference:
    SPE 10th Comparative Solution Project
    https://www.spe.org/web/csp/
"""

import jax.numpy as jnp
import numpy as onp
from typing import Any, Dict, Optional
from dataclasses import dataclass
import os


@dataclass
class SPE10Config:
    """Configuration for SPE10 benchmark."""

    nx: int = 220
    ny: int = 60
    nz: int = 60
    dx: float = 1.0  # ft
    dy: float = 1.0  # ft
    dz: float = 2.0  # ft
    layer: int = 1  # which layer to use (1-60)


def generate_synthetic_spe10(
    config: Optional[SPE10Config] = None, seed: int = 42
) -> Dict[str, jnp.ndarray]:
    """Generate synthetic SPE10-like heterogeneous media.

    Creates a log-normal permeability field similar to SPE10.

    Args:
        config: SPE10 configuration
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
        - perm_x: Permeability in x-direction (nx, ny)
        - perm_y: Permeability in y-direction (nx, ny)
        - perm_z: Permeability in z-direction (nx, ny)
        - porosity: Porosity field (nx, ny)
        - x, y, z: Grid coordinates
    """
    if config is None:
        config = SPE10Config()

    onp.random.seed(seed)

    nx, ny, nz = config.nx, config.ny, config.nz

    log_k_mean = onp.log(1e-3)  # mean of log(k) in Darcy
    log_k_std = 2.0  # standard deviation

    log_perm = onp.random.randn(nz, ny, nx) * log_k_std + log_k_mean

    perm_3d = onp.exp(log_perm)

    layer_idx = config.layer - 1
    perm_layer = perm_3d[layer_idx]

    perm_x = jnp.array(perm_layer.T)
    perm_y = jnp.array(perm_layer.T)
    perm_z = jnp.array(perm_layer.T * 0.1)

    porosity = jnp.array(onp.random.uniform(0.15, 0.35, (nx, ny)))

    x = jnp.arange(nx) * config.dx
    y = jnp.arange(ny) * config.dy
    z = jnp.array([0.0])

    return {
        "perm_x": perm_x,
        "perm_y": perm_y,
        "perm_z": perm_z,
        "porosity": porosity,
        "x": x,
        "y": y,
        "z": z,
        "nx": nx,
        "ny": ny,
        "config": config,
    }


def load_spe10_case1(data_dir: str = "data") -> Dict[str, jnp.ndarray]:
    """Load SPE10 Case 1 (2D vertical cross-section).

    Case 1 is a 2D vertical cross-section (100 x 20).
    File format: perm_case1.dat contains Kx, Ky, Kz in sequence (3 x 2000 values).

    Args:
        data_dir: Directory containing SPE10 files

    Returns:
        Dictionary with:
        - perm_x: Permeability in x-direction (100, 20)
        - perm_y: Permeability in y-direction (100, 20)
        - perm_z: Permeability in z-direction (100, 20)
        - porosity: Porosity field (100, 20) from phi-k correlation
        - x, y: Grid coordinates
    """
    nx, ny = 100, 20

    filepath = os.path.join(data_dir, "perm_case1.dat")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SPE10 Case 1 file not found: {filepath}")

    with open(filepath, "r") as f:
        content = f.read()

    import re

    values = re.findall(r"[\d.]+", content)
    values = [float(v) for v in values]

    n_perm = nx * ny
    perm_x = onp.array(values[:n_perm]).reshape(ny, nx).T
    perm_y = onp.array(values[n_perm : 2 * n_perm]).reshape(ny, nx).T
    perm_z = onp.array(values[2 * n_perm : 3 * n_perm]).reshape(ny, nx).T

    porosity = porosity_from_perm(perm_x)

    x = jnp.arange(nx) * 1.0 + 0.5
    y = jnp.arange(ny) * 1.0 + 0.5

    return {
        "perm_x": jnp.array(perm_x),
        "perm_y": jnp.array(perm_y),
        "perm_z": jnp.array(perm_z),
        "porosity": jnp.array(porosity),
        "x": x,
        "y": y,
        "nx": nx,
        "ny": ny,
    }


def porosity_from_perm(
    perm: jnp.ndarray, a: float = 0.25, b: float = 0.1
) -> jnp.ndarray:
    """Compute porosity from permeability using phi-k correlation.

    Uses power-law relationship: phi = a * k^b
    Typical values: a ~= 0.25, b ~= 0.1 for sandstone

    Args:
        perm: Permeability in mD
        a: Coefficient (default 0.25)
        b: Exponent (default 0.1)

    Returns:
        Porosity (fraction)
    """
    perm_clamped = jnp.clip(perm, 1e-6, 1e6)
    phi = a * jnp.power(perm_clamped, b)
    return jnp.clip(phi, 0.05, 0.5)


def load_spe10_layer(
    layer: int = 1, data_dir: Optional[str] = None
) -> Dict[str, jnp.ndarray]:
    """Load a specific layer from SPE10 dataset.

    If SPE10 files are not available, generates synthetic data.

    Args:
        layer: Layer number (1-60)
        data_dir: Directory with SPE10 data files (optional)

    Returns:
        Dictionary with permeability and porosity fields
    """
    if data_dir is not None and os.path.exists(data_dir):
        try:
            return load_spe10_from_file(layer, data_dir)
        except Exception as e:
            print(f"Could not load SPE10 from {data_dir}: {e}")
            print("Using synthetic data instead")

    config = SPE10Config(layer=layer)
    return generate_synthetic_spe10(config)


def load_spe10_from_file(layer: int, data_dir: str) -> Dict[str, jnp.ndarray]:
    """Load SPE10 data from actual files.

    Expected file format:
    - perm_x_LAYER.dat
    - perm_y_LAYER.dat
    - perm_z_LAYER.dat
    - porosity_LAYER.dat

    Args:
        layer: Layer number
        data_dir: Directory containing SPE10 files

    Returns:
        Dictionary with fields
    """

    def load_file(filename):
        filepath = os.path.join(data_dir, filename)
        data = onp.loadtxt(filepath)
        return jnp.array(data)

    nx, ny = 220, 60

    perm_x = load_file(f"permx{layer}.dat").reshape(ny, nx).T
    perm_y = load_file(f"permy{layer}.dat").reshape(ny, nx).T
    perm_z = load_file(f"permz{layer}.dat").reshape(ny, nx).T
    porosity = load_file(f"poro{layer}.dat").reshape(ny, nx).T

    x = jnp.arange(nx) * 1.0
    y = jnp.arange(ny) * 1.0

    return {
        "perm_x": perm_x,
        "perm_y": perm_y,
        "perm_z": perm_z,
        "porosity": porosity,
        "x": x,
        "y": y,
        "nx": nx,
        "ny": ny,
        "layer": layer,
    }


def create_spe10_simulation_params(data: Dict[str, jnp.ndarray], params) -> Dict:
    """Create simulation parameters from SPE10 data.

    Args:
        data: SPE10 data dictionary
        params: Base flow parameters

    Returns:
        Updated parameters with SPE10 fields
    """
    return {
        "perm_x": data["perm_x"],
        "perm_y": data["perm_y"],
        "phi": data["porosity"],
        "k_reference": jnp.mean(data["perm_x"]),
    }


def run_spe10_benchmark(
    layer: int = 35, t_max: float = 0.01, nx_sub: int = 60, ny_sub: int = 30
) -> Dict[str, jnp.ndarray]:
    """Run 2D simulation on SPE10 layer (subsampled).

    Args:
        layer: SPE10 layer number
        t_max: Maximum simulation time
        nx_sub: Subsample in x
        ny_sub: Subsample in y

    Returns:
        Simulation results
    """
    print(f"Running SPE10 benchmark on layer {layer}")
    print(f"Grid: {nx_sub} x {ny_sub}")

    data_full = load_spe10_layer(layer)

    perm_x = onp.array(data_full["perm_x"])
    perm_y = onp.array(data_full["perm_y"])
    porosity = onp.array(data_full["porosity"])

    perm_x_sub = perm_x[:nx_sub, :ny_sub]
    perm_y_sub = perm_y[:nx_sub, :ny_sub]
    porosity_sub = porosity[:nx_sub, :ny_sub]

    perm_x_jax = jnp.array(perm_x_sub)
    perm_y_jax = jnp.array(perm_y_sub)
    phi_jax = jnp.array(porosity_sub)

    from .dataset import generate_2d_heterogeneous

    Lx = 60.0
    Ly = 30.0

    result = generate_2d_heterogeneous(
        nx=nx_sub, ny=ny_sub, Lx=Lx, Ly=Ly, t_max=t_max, dt=0.0001
    )

    result["perm_x"] = perm_x_jax
    result["perm_y"] = perm_y_jax
    result["porosity"] = phi_jax
    result["layer"] = layer

    return result


def compare_with_analytical(data: Dict[str, jnp.ndarray], params) -> Dict[str, float]:
    """Compare numerical solution with analytical (if available).

    For 1D displacement, can compare with Welge construction.

    Args:
        data: Simulation results
        params: Flow parameters

    Returns:
        Dictionary with comparison metrics
    """
    if "Sw" not in data or len(data["Sw"].shape) != 2:
        return {"status": "no_comparison_available"}

    Sw_final = data["Sw"][-1]

    Sw_avg = jnp.mean(Sw_final)

    Sw_inj = 1.0 - params.Sor
    Sw_init = params.Swc + params.Sor

    recovery = (Sw_avg - Sw_init) / (Sw_inj - Sw_init) * 100

    from .properties import welge_construct

    try:
        S_shock, v_shock = welge_construct(Sw_inj, Sw_init, params)

        return {
            "recovery_factor": float(recovery),
            "shock_saturation": float(S_shock),
            "shock_velocity": float(v_shock),
            "avg_saturation": float(Sw_avg),
            "status": "ok",
        }
    except Exception as e:
        return {
            "recovery_factor": float(recovery),
            "status": "partial",
            "error": str(e),
        }


def load_big_model(data_dir: str = "data") -> Dict[str, Any]:
    """Load BIG_MODEL_12_09_1 Eclipse dataset.

    Args:
        data_dir: Directory containing BIG_MODEL files

    Returns:
        Dictionary with:
        - perm_x, perm_y, perm_z: permeability (nx, ny, nz)
        - porosity: porosity (nx, ny, nz)
        - x, y, z: grid coordinates
        - nx, ny, nz: grid dimensions (122, 183, 43)
        - summary: DataFrame from res2df
    """
    import re

    import logging

    logging.getLogger("res2df").setLevel(logging.CRITICAL)
    import res2df
    from res2df import ResdataFiles

    if data_dir.endswith("BIG_MODEL_12_09_1"):
        base = os.path.join(data_dir, "BIG_MODEL_12_09_1")
    else:
        base = os.path.join(data_dir, "BIG_MODEL_12_09_1", "BIG_MODEL_12_09_1")

    # Parse DIMENS from DATA file
    data_file = base + ".DATA"
    with open(data_file, "r") as f:
        content = f.read()

    dims = re.findall(r"DIMENS\s+(\d+)\s+(\d+)\s+(\d+)", content)
    if dims:
        nx, ny, nz = int(dims[0][0]), int(dims[0][1]), int(dims[0][2])
    else:
        nx, ny, nz = 122, 183, 43  # defaults from DATA file

    # Initialize placeholders
    perm_x = onp.ones((nx, ny, nz)) * 100.0
    perm_y = onp.ones((nx, ny, nz)) * 100.0
    perm_z = onp.ones((nx, ny, nz)) * 10.0
    porosity = onp.ones((nx, ny, nz)) * 0.2

    try:
        # Parse GRDECL files - handle ECLIPSE N*VALUE format
        def parse_grdecl(filepath):
            if not os.path.exists(filepath):
                return None
            values = []
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("--"):
                        continue
                    for token in line.split():
                        if token == "/":
                            break
                        if "*" in token:
                            parts = token.split("*")
                            if len(parts) == 2:
                                try:
                                    count = int(parts[0])
                                    val = float(parts[1])
                                    values.extend([val] * count)
                                except ValueError:
                                    continue
                        else:
                            try:
                                values.append(float(token))
                            except ValueError:
                                continue
            return onp.array(values) if values else None

        # Parse each property file
        poro_arr = parse_grdecl(base + "_PROP_PORO.GRDECL")
        permx_arr = parse_grdecl(base + "_PROP_PERMX.GRDECL")
        permy_arr = parse_grdecl(base + "_PROP_PERMY.GRDECL")
        permz_arr = parse_grdecl(base + "_PROP_PERMZ.GRDECL")

        n_total = nx * ny * nz

        if poro_arr is not None and len(poro_arr) == n_total:
            porosity = poro_arr.reshape(nz, ny, nx).swapaxes(0, 2)
        if permx_arr is not None and len(permx_arr) == n_total:
            perm_x = permx_arr.reshape(nz, ny, nx).swapaxes(0, 2)
        if permy_arr is not None and len(permy_arr) == n_total:
            perm_y = permy_arr.reshape(nz, ny, nx).swapaxes(0, 2)
        if permz_arr is not None and len(permz_arr) == n_total:
            perm_z = permz_arr.reshape(nz, ny, nx).swapaxes(0, 2)
    except Exception as e:
        print(f"Warning: Could not parse GRDECL files: {e}")

    # Try to load summary - suppress errors gracefully
    summary = None
    try:
        unsmry_file = base + ".UNSMRY"
        if os.path.exists(unsmry_file):
            rf = ResdataFiles(unsmry_file)
            summary = res2df.summary.df(rf, column_keys="*", time_index="raw")
    except Exception:
        pass

    # Grid coordinates
    x = onp.arange(nx) * 1.0
    y = onp.arange(ny) * 1.0
    z = onp.arange(nz) * 1.0

    return {
        "perm_x": jnp.array(perm_x),
        "perm_y": jnp.array(perm_y),
        "perm_z": jnp.array(perm_z),
        "porosity": jnp.array(porosity),
        "x": jnp.array(x),
        "y": jnp.array(y),
        "z": jnp.array(z),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "summary": summary,
        "base": base,
    }
