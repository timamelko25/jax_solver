"""VTK export module for ResInsight-compatible ASCII VTK files."""

import os
from typing import Optional

import jax.numpy as jnp
import numpy as onp


def write_vtk_header(f, filename: str) -> None:
    """Write VTK file header.

    Args:
        f: Open file handle
        filename: Name for the VTK file
    """
    f.write("# vtk DataFile Version 3.0\n")
    f.write(f"{filename}\n")
    f.write("ASCII\n")


def write_vtk_points(f, x: onp.ndarray, y: Optional[onp.ndarray] = None) -> None:
    if y is None:
        n_points = len(x)
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {n_points} 1 1\n")
        f.write("ORIGIN 0 0 0\n")
        f.write(f"SPACING {x[1] - x[0] if len(x) > 1 else 1.0} 1 1\n")
    else:
        nx, ny = len(x), len(y)
        n_points = nx * ny
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write(f"POINTS {n_points} float\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{x[i]} {y[j]} 0\n")


def write_vtk_cell_data_header(f, n_cells: int) -> None:
    f.write("CELL_DATA {}\n".format(n_cells))


def write_vtk_point_data_header(f, n_points: int) -> None:
    f.write("POINT_DATA {}\n".format(n_points))


def write_vtk_cells(f, nx: int, ny: int = 1) -> None:
    """Write VTK cells section for structured grid.

    Args:
        f: Open file handle
        nx: Number of cells in x direction
        ny: Number of cells in y direction
    """
    n_cells = nx * ny
    f.write(f"CELLS {n_cells} {n_cells * 5}\n")

    for j in range(ny):
        for i in range(nx):
            p0 = j * (nx + 1) + i
            p1 = p0 + 1
            p2 = p0 + (nx + 1)
            p3 = p2 + 1
            f.write(f"4 {p0} {p1} {p2} {p3}\n")


def write_vtk_cell_data(f, name: str, values: onp.ndarray) -> None:
    """Write cell data section.

    Args:
        f: Open file handle
        name: Scalar field name
        values: Array of cell values (n_cells,)
    """
    write_vtk_cell_data_header(f, len(values))
    write_vtk_scalar(f, name, values)


def write_vtk_point_data(f, name: str, values: onp.ndarray) -> None:
    """Write point data section.

    Args:
        f: Open file handle
        name: Scalar field name
        values: Array of point values (n_points,)
    """
    write_vtk_point_data_header(f, len(values))
    write_vtk_scalar(f, name, values)


def write_vtk_scalar(f, name: str, values: onp.ndarray) -> None:
    f.write("SCALARS {} float 1\n".format(name))
    f.write("LOOKUP_TABLE default\n")
    for v in values:
        f.write("{}\n".format(v))


def _ensure_dir(filepath: str) -> None:
    """Ensure output directory exists."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _to_numpy(arr) -> onp.ndarray:
    """Convert JAX array to numpy array if needed."""
    if isinstance(arr, jnp.ndarray):
        return onp.asarray(arr)
    return onp.asarray(arr)


def export_1d_saturation(
    saturation: onp.ndarray,
    x: Optional[onp.ndarray] = None,
    dx: float = 1.0,
    output_path: str = "outputs/vtk/saturation_1d.vtk",
    time_value: Optional[float] = None,
) -> None:
    """Export 1D saturation field to VTK file.

    Args:
        saturation: 1D array of saturation values (n_cells)
        x: Optional x coordinates (default: uniform spacing with dx)
        dx: Grid spacing if x not provided
        output_path: Output file path
        time_value: Optional time value for timestamping
    """
    sat = _to_numpy(saturation)
    n_cells = len(sat)
    n_points = n_cells + 1  # points = cells + 1 for finite volume

    # Generate x coordinates if not provided
    if x is None:
        x_coords = onp.arange(n_points) * dx
    else:
        x_coords = _to_numpy(x)
        if len(x_coords) != n_points:
            raise ValueError(
                f"x coordinates length ({len(x_coords)}) must match "
                f"n_cells + 1 ({n_points})"
            )

    _ensure_dir(output_path)

    with open(output_path, "w") as f:
        # Header
        filename = os.path.basename(output_path)
        write_vtk_header(f, filename)

        # Points (cell centers for 1D)
        write_vtk_points(f, x_coords)

        # Point data
        write_vtk_point_data_header(f, n_points)

        # Saturation at cell centers (use points data)
        write_vtk_scalar(f, "saturation", sat)


def export_2d_saturation(
    saturation: onp.ndarray,
    nx: int,
    ny: int,
    x: Optional[onp.ndarray] = None,
    y: Optional[onp.ndarray] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    output_path: str = "outputs/vtk/saturation_2d.vtk",
    time_value: Optional[float] = None,
) -> None:
    """Export 2D saturation field to VTK file.

    Args:
        saturation: 2D array of saturation values (ny x nx) in row-major order
        nx: Number of cells in x direction
        ny: Number of cells in y direction
        x: Optional x coordinates (default: uniform spacing with dx)
        y: Optional y coordinates (default: uniform spacing with dy)
        dx: Grid spacing in x if x not provided
        dy: Grid spacing in y if y not provided
        output_path: Output file path
        time_value: Optional time value for timestamping
    """
    sat = _to_numpy(saturation)

    # Ensure correct shape
    if sat.shape != (ny, nx):
        # Try to reshape if flat
        if sat.ndim == 1 and len(sat) == nx * ny:
            sat = sat.reshape((ny, nx))
        else:
            raise ValueError(
                f"Saturation shape {sat.shape} must match (ny, nx) = ({ny}, {nx})"
            )

    # Number of points = (nx+1) * (ny+1) for cell-centered grid
    npx, npy = nx + 1, ny + 1
    n_points = npx * npy

    # Generate coordinates if not provided
    if x is None:
        x_coords = onp.arange(npx) * dx
    else:
        x_coords = _to_numpy(x)
        if len(x_coords) != npx:
            raise ValueError(
                f"x coordinates length ({len(x_coords)}) must match nx + 1 ({npx})"
            )

    if y is None:
        y_coords = onp.arange(npy) * dy
    else:
        y_coords = _to_numpy(y)
        if len(y_coords) != npy:
            raise ValueError(
                f"y coordinates length ({len(y_coords)}) must match ny + 1 ({npy})"
            )

    _ensure_dir(output_path)

    with open(output_path, "w") as f:
        # Header
        filename = os.path.basename(output_path)
        write_vtk_header(f, filename)

        # Points
        write_vtk_points(f, x_coords, y_coords)

        # Point data
        write_vtk_point_data_header(f, n_points)

        # Saturation at cell centers expanded to vertices
        # For simplicity, use saturation values at points (average of adjacent cells)
        sat_points = onp.zeros((npy, npx))

        # Interior points: average of 4 adjacent cells
        for j in range(1, npy - 1):
            for i in range(1, npx - 1):
                sat_points[j, i] = (
                    sat[j - 1, i - 1] + sat[j - 1, i] + sat[j, i - 1] + sat[j, i]
                ) / 4.0

        # Boundary points: copy from nearest interior
        sat_points[0, :] = sat_points[1, :]
        sat_points[-1, :] = sat_points[-2, :]
        sat_points[:, 0] = sat_points[:, 1]
        sat_points[:, -1] = sat_points[:, -2]

        # Flatten in row-major order (y varies fastest for VTK)
        sat_flat = sat_points.flatten()
        write_vtk_scalar(f, "saturation", sat_flat)


def export_pressure(
    pressure: onp.ndarray,
    nx: int,
    ny: int,
    x: Optional[onp.ndarray] = None,
    y: Optional[onp.ndarray] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    output_path: str = "outputs/vtk/pressure.vtk",
    time_value: Optional[float] = None,
) -> None:
    """Export 2D pressure field to VTK file.

    Args:
        pressure: 2D array of pressure values (ny x nx) in row-major order
        nx: Number of cells in x direction
        ny: Number of cells in y direction
        x: Optional x coordinates (default: uniform spacing with dx)
        y: Optional y coordinates (default: uniform spacing with dy)
        dx: Grid spacing in x if x not provided
        dy: Grid spacing in y if y not provided
        output_path: Output file path
        time_value: Optional time value for timestamping
    """
    pres = _to_numpy(pressure)

    # Ensure correct shape
    if pres.shape != (ny, nx):
        # Try to reshape if flat
        if pres.ndim == 1 and len(pres) == nx * ny:
            pres = pres.reshape((ny, nx))
        else:
            raise ValueError(
                f"Pressure shape {pres.shape} must match (ny, nx) = ({ny}, {nx})"
            )

    # Number of points = (nx+1) * (ny+1) for cell-centered grid
    npx, npy = nx + 1, ny + 1
    n_points = npx * npy

    # Generate coordinates if not provided
    if x is None:
        x_coords = onp.arange(npx) * dx
    else:
        x_coords = _to_numpy(x)
        if len(x_coords) != npx:
            raise ValueError(
                f"x coordinates length ({len(x_coords)}) must match nx + 1 ({npx})"
            )

    if y is None:
        y_coords = onp.arange(npy) * dy
    else:
        y_coords = _to_numpy(y)
        if len(y_coords) != npy:
            raise ValueError(
                f"y coordinates length ({len(y_coords)}) must match ny + 1 ({npy})"
            )

    _ensure_dir(output_path)

    with open(output_path, "w") as f:
        # Header
        filename = os.path.basename(output_path)
        write_vtk_header(f, filename)

        # Points
        write_vtk_points(f, x_coords, y_coords)

        # Point data
        write_vtk_point_data_header(f, n_points)

        # Pressure at cell centers expanded to vertices
        pres_points = onp.zeros((npy, npx))

        # Interior points: average of 4 adjacent cells
        for j in range(1, npy - 1):
            for i in range(1, npx - 1):
                pres_points[j, i] = (
                    pres[j - 1, i - 1] + pres[j - 1, i] + pres[j, i - 1] + pres[j, i]
                ) / 4.0

        # Boundary points: copy from nearest interior
        pres_points[0, :] = pres_points[1, :]
        pres_points[-1, :] = pres_points[-2, :]
        pres_points[:, 0] = pres_points[:, 1]
        pres_points[:, -1] = pres_points[:, -2]

        # Flatten in row-major order
        pres_flat = pres_points.flatten()
        write_vtk_scalar(f, "pressure", pres_flat)


def export_1d_saturation_history(
    Sw_history: onp.ndarray,
    x: onp.ndarray,
    dx: float,
    output_dir: str,
    time_array: onp.ndarray,
    export_interval: int = 1,
    p_history: Optional[onp.ndarray] = None,
) -> None:
    """Export 1D saturation history to VTK files.

    Args:
        Sw_history: Saturation history array (n_times, n_cells)
        x: X coordinates (n_cells + 1)
        dx: Grid spacing
        output_dir: Output directory path
        time_array: Time values (n_times,)
        export_interval: Export every N time steps
        p_history: Optional pressure history (n_times, n_cells)
    """
    os.makedirs(output_dir, exist_ok=True)

    n_times = Sw_history.shape[0]
    n_cells = Sw_history.shape[1]
    n_points = n_cells + 1

    for t_idx in range(0, n_times, export_interval):
        sat_file = os.path.join(output_dir, f"saturation_t{t_idx:04d}.vtk")
        sat_slice = Sw_history[t_idx]

        sat_points_out = onp.zeros(n_points)
        sat_points_out[1:-1] = (sat_slice[:-1] + sat_slice[1:]) / 2.0
        sat_points_out[0] = sat_slice[0]
        sat_points_out[-1] = sat_slice[-1]

        with open(sat_file, "w") as f:
            filename = os.path.basename(sat_file)
            write_vtk_header(f, filename)
            write_vtk_points(f, x)
            write_vtk_point_data_header(f, n_points)
            write_vtk_scalar(f, "saturation", sat_points_out)

        # Export pressure if provided
        if p_history is not None:
            p_file = os.path.join(output_dir, f"pressure_t{t_idx:04d}.vtk")
            p_slice = p_history[t_idx]

            p_points_out = onp.zeros(n_points)
            p_points_out[1:-1] = (p_slice[:-1] + p_slice[1:]) / 2.0
            p_points_out[0] = p_slice[0]
            p_points_out[-1] = p_slice[-1]

            with open(p_file, "w") as f:
                filename = os.path.basename(p_file)
                write_vtk_header(f, filename)
                write_vtk_points(f, x)
                write_vtk_point_data_header(f, n_points)
                write_vtk_scalar(f, "pressure", p_points_out)


def export_2d_saturation_history(
    Sw_history: onp.ndarray,
    nx: int,
    ny: int,
    x: onp.ndarray,
    y: onp.ndarray,
    dx: float,
    dy: float,
    output_dir: str,
    time_array: onp.ndarray,
    export_interval: int = 1,
) -> None:
    """Export 2D saturation history to VTK files.

    Args:
        Sw_history: Saturation history array (n_times, ny, nx)
        nx: Number of cells in x
        ny: Number of cells in y
        x: X coordinates (nx + 1)
        y: Y coordinates (ny + 1)
        dx: Grid spacing in x
        dy: Grid spacing in y
        output_dir: Output directory path
        time_array: Time values (n_times,)
        export_interval: Export every N time steps
    """
    os.makedirs(output_dir, exist_ok=True)

    n_times, ny_in, nx_in = Sw_history.shape
    if nx_in != nx or ny_in != ny:
        raise ValueError(
            f"Sw_history shape {Sw_history.shape} must be (n_times, {ny}, {nx})"
        )

    npx, npy = nx + 1, ny + 1
    n_points = npx * npy

    for t_idx in range(0, n_times, export_interval):
        sat_file = os.path.join(output_dir, f"saturation_2d_t{t_idx:04d}.vtk")
        sat_slice = Sw_history[t_idx]  # (ny, nx)

        # Expand cell-centered to vertices
        sat_points_out = onp.zeros((npy, npx))

        # Interior: average of 4 adjacent cells
        for j in range(1, npy - 1):
            for i in range(1, npx - 1):
                sat_points_out[j, i] = (
                    sat_slice[j - 1, i - 1]
                    + sat_slice[j - 1, i]
                    + sat_slice[j, i - 1]
                    + sat_slice[j, i]
                ) / 4.0

        # Boundary: copy from nearest interior
        sat_points_out[0, :] = sat_points_out[1, :]
        sat_points_out[-1, :] = sat_points_out[-2, :]
        sat_points_out[:, 0] = sat_points_out[:, 1]
        sat_points_out[:, -1] = sat_points_out[:, -2]

        sat_flat = sat_points_out.flatten()

        with open(sat_file, "w") as f:
            filename = os.path.basename(sat_file)
            write_vtk_header(f, filename)
            write_vtk_points(f, x, y)
            write_vtk_point_data_header(f, n_points)
            write_vtk_scalar(f, "saturation", sat_flat)


def export_1d_pressure_history(
    p_history: onp.ndarray,
    x: onp.ndarray,
    dx: float,
    output_dir: str,
    time_array: onp.ndarray,
    export_interval: int = 1,
) -> None:
    """Export 1D pressure history to VTK files.

    Args:
        p_history: Pressure history array (n_times, n_cells)
        x: X coordinates (n_cells + 1)
        dx: Grid spacing
        output_dir: Output directory path
        time_array: Time values (n_times,)
        export_interval: Export every N time steps
    """
    os.makedirs(output_dir, exist_ok=True)

    n_times = p_history.shape[0]
    n_cells = p_history.shape[1]
    n_points = n_cells + 1

    for t_idx in range(0, n_times, export_interval):
        p_file = os.path.join(output_dir, f"pressure_t{t_idx:04d}.vtk")
        p_slice = p_history[t_idx]

        # Expand cell-centered to points
        p_points_out = onp.zeros(n_points)
        p_points_out[1:-1] = (p_slice[:-1] + p_slice[1:]) / 2.0
        p_points_out[0] = p_slice[0]
        p_points_out[-1] = p_slice[-1]

        with open(p_file, "w") as f:
            filename = os.path.basename(p_file)
            write_vtk_header(f, filename)
            write_vtk_points(f, x)
            write_vtk_point_data_header(f, n_points)
            write_vtk_scalar(f, "pressure", p_points_out)


def write_vtk_points_3d(f, x: onp.ndarray, y: onp.ndarray, z: onp.ndarray) -> None:
    """Write 3D VTK points for structured grid.

    Args:
        f: Open file handle
        x: X coordinates (nx,)
        y: Y coordinates (ny,)
        z: Z coordinates (nz,)
    """
    nx, ny, nz = len(x), len(y), len(z)
    n_points = nx * ny * nz
    f.write("DATASET STRUCTURED_GRID\n")
    f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
    f.write(f"POINTS {n_points} float\n")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                f.write(f"{x[i]} {y[j]} {z[k]}\n")


def export_3d_saturation(
    perm_x: onp.ndarray,
    perm_y: onp.ndarray,
    perm_z: onp.ndarray,
    saturation: onp.ndarray,
    filename: str,
    x: Optional[onp.ndarray] = None,
    y: Optional[onp.ndarray] = None,
    z: Optional[onp.ndarray] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
) -> None:
    """Export 3D saturation field to VTK file for ResInsight.

    Args:
        perm_x: Permeability in x-direction (nx, ny, nz)
        perm_y: Permeability in y-direction (nx, ny, nz)
        perm_z: Permeability in z-direction (nx, ny, nz)
        saturation: Saturation field (nx, ny, nz)
        filename: Output VTK filename
        x: Optional x coordinates
        y: Optional y coordinates
        z: Optional z coordinates
        dx: Grid spacing in x if coordinates not provided
        dy: Grid spacing in y if coordinates not provided
        dz: Grid spacing in z if coordinates not provided
    """
    sat = _to_numpy(saturation)
    perm_x = _to_numpy(perm_x)
    perm_y = _to_numpy(perm_y)
    perm_z = _to_numpy(perm_z)

    if sat.ndim != 3:
        raise ValueError(f"Saturation must be 3D array, got shape {sat.shape}")

    nx, ny, nz = sat.shape

    if x is None:
        x_coords = onp.arange(nx) * dx
    else:
        x_coords = _to_numpy(x)
        if len(x_coords) != nx:
            raise ValueError(
                f"x coordinates length ({len(x_coords)}) must match nx ({nx})"
            )

    if y is None:
        y_coords = onp.arange(ny) * dy
    else:
        y_coords = _to_numpy(y)
        if len(y_coords) != ny:
            raise ValueError(
                f"y coordinates length ({len(y_coords)}) must match ny ({ny})"
            )

    if z is None:
        z_coords = onp.arange(nz) * dz
    else:
        z_coords = _to_numpy(z)
        if len(z_coords) != nz:
            raise ValueError(
                f"z coordinates length ({len(z_coords)}) must match nz ({nz})"
            )

    _ensure_dir(filename)

    with open(filename, "w") as f:
        write_vtk_header(f, os.path.basename(filename))
        write_vtk_points_3d(f, x_coords, y_coords, z_coords)

        n_points = nx * ny * nz
        write_vtk_point_data_header(f, n_points)

        # Fortran order required for VTK structured grid
        sat_flat = sat.flatten(order="F")
        write_vtk_scalar(f, "saturation", sat_flat)

        if perm_x.shape == (nx, ny, nz):
            perm_x_flat = perm_x.flatten(order="F")
            write_vtk_scalar(f, "permeability_x", perm_x_flat)

        if perm_y.shape == (nx, ny, nz):
            perm_y_flat = perm_y.flatten(order="F")
            write_vtk_scalar(f, "permeability_y", perm_y_flat)

        if perm_z.shape == (nx, ny, nz):
            perm_z_flat = perm_z.flatten(order="F")
            write_vtk_scalar(f, "permeability_z", perm_z_flat)


def export_3d_pressure(
    perm_x: onp.ndarray,
    perm_y: onp.ndarray,
    perm_z: onp.ndarray,
    pressure: onp.ndarray,
    filename: str,
    x: Optional[onp.ndarray] = None,
    y: Optional[onp.ndarray] = None,
    z: Optional[onp.ndarray] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
) -> None:
    """Export 3D pressure field to VTK file for ResInsight.

    Args:
        perm_x: Permeability in x-direction (nx, ny, nz)
        perm_y: Permeability in y-direction (nx, ny, nz)
        perm_z: Permeability in z-direction (nx, ny, nz)
        pressure: Pressure field (nx, ny, nz)
        filename: Output VTK filename
        x: Optional x coordinates
        y: Optional y coordinates
        z: Optional z coordinates
        dx: Grid spacing in x if coordinates not provided
        dy: Grid spacing in y if coordinates not provided
        dz: Grid spacing in z if coordinates not provided
    """
    pres = _to_numpy(pressure)
    perm_x = _to_numpy(perm_x)
    perm_y = _to_numpy(perm_y)
    perm_z = _to_numpy(perm_z)

    # Get grid dimensions from pressure shape
    if pres.ndim != 3:
        raise ValueError(f"Pressure must be 3D array, got shape {pres.shape}")

    nx, ny, nz = pres.shape

    # Generate coordinates if not provided
    if x is None:
        x_coords = onp.arange(nx) * dx
    else:
        x_coords = _to_numpy(x)
        if len(x_coords) != nx:
            raise ValueError(
                f"x coordinates length ({len(x_coords)}) must match nx ({nx})"
            )

    if y is None:
        y_coords = onp.arange(ny) * dy
    else:
        y_coords = _to_numpy(y)
        if len(y_coords) != ny:
            raise ValueError(
                f"y coordinates length ({len(y_coords)}) must match ny ({ny})"
            )

    if z is None:
        z_coords = onp.arange(nz) * dz
    else:
        z_coords = _to_numpy(z)
        if len(z_coords) != nz:
            raise ValueError(
                f"z coordinates length ({len(z_coords)}) must match nz ({nz})"
            )

    _ensure_dir(filename)

    with open(filename, "w") as f:
        # Header
        write_vtk_header(f, os.path.basename(filename))

        # Points
        write_vtk_points_3d(f, x_coords, y_coords, z_coords)

        # Point data
        n_points = nx * ny * nz
        write_vtk_point_data_header(f, n_points)

        # Pressure (flatten in Fortran order: x fastest, then y, then z)
        pres_flat = pres.flatten(order="F")
        write_vtk_scalar(f, "pressure", pres_flat)

        # Permeability fields
        if perm_x.shape == (nx, ny, nz):
            perm_x_flat = perm_x.flatten(order="F")
            write_vtk_scalar(f, "permeability_x", perm_x_flat)

        if perm_y.shape == (nx, ny, nz):
            perm_y_flat = perm_y.flatten(order="F")
            write_vtk_scalar(f, "permeability_y", perm_y_flat)

        if perm_z.shape == (nx, ny, nz):
            perm_z_flat = perm_z.flatten(order="F")
            write_vtk_scalar(f, "permeability_z", perm_z_flat)


def export_3d_saturation_history(
    saturation_history: onp.ndarray,
    perm_x: onp.ndarray,
    perm_y: onp.ndarray,
    perm_z: onp.ndarray,
    output_dir: str,
    x: Optional[onp.ndarray] = None,
    y: Optional[onp.ndarray] = None,
    z: Optional[onp.ndarray] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    export_interval: int = 1,
    time_array: Optional[onp.ndarray] = None,
) -> None:
    """Export 3D saturation history to VTK files.

    Args:
        saturation_history: Saturation history (n_times, nx, ny, nz)
        perm_x: Permeability in x-direction (nx, ny, nz)
        perm_y: Permeability in y-direction (nx, ny, nz)
        perm_z: Permeability in z-direction (nx, ny, nz)
        output_dir: Output directory
        x: Optional x coordinates
        y: Optional y coordinates
        z: Optional z coordinates
        dx: Grid spacing in x
        dy: Grid spacing in y
        dz: Grid spacing in z
        export_interval: Export every N time steps
        time_array: Optional time values for naming
    """
    os.makedirs(output_dir, exist_ok=True)

    sat_history = _to_numpy(saturation_history)
    n_times = sat_history.shape[0]

    for t_idx in range(0, n_times, export_interval):
        if time_array is not None:
            time_val = time_array[t_idx]
            suffix = f"{t_idx:04d}_t{time_val:.3f}"
        else:
            suffix = f"{t_idx:04d}"

        filename = os.path.join(output_dir, f"saturation_3d_{suffix}.vtk")
        export_3d_saturation(
            perm_x,
            perm_y,
            perm_z,
            sat_history[t_idx],
            filename,
            x,
            y,
            z,
            dx,
            dy,
            dz,
        )
