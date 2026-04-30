"""ResInsight integration module for 3D reservoir visualization.

Exports BIG_MODEL data to ResInsight-compatible VTK format.
ResInsight is an open-source 3D visualization tool from Equinor.
"""

import os
import numpy as onp

from src.vtk_export import export_3d_saturation, export_3d_pressure


def load_big_model_data(data_dir: str) -> dict:
    """Load BIG_MODEL_12_09_1 Eclipse format data.

    Args:
        data_dir: Path to directory containing BIG_MODEL_12_09_1

    Returns:
        Dictionary with perm_x, perm_y, perm_z, porosity, grid dimensions
    """
    from .benchmarks import load_big_model

    data = load_big_model(data_dir)

    return {
        "perm_x": onp.array(data["perm_x"]),
        "perm_y": onp.array(data["perm_y"]),
        "perm_z": onp.array(data["perm_z"]),
        "porosity": onp.array(data["porosity"]),
        "x": onp.array(data["x"]),
        "y": onp.array(data["y"]),
        "z": onp.array(data["z"]),
        "nx": data["nx"],
        "ny": data["ny"],
        "nz": data["nz"],
    }


def export_for_resinsight(
    data_dir: str = "data",
    output_dir: str = "outputs/resinsight/",
    saturation: onp.ndarray = None,
    pressure: onp.ndarray = None,
) -> None:
    """Export BIG_MODEL data to ResInsight-compatible VTK files.

    Args:
        data_dir: Directory containing BIG_MODEL data
        output_dir: Output directory for VTK files
        saturation: Optional saturation field (nx, ny, nz) to export
        pressure: Optional pressure field (nx, ny, nz) to export
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading BIG_MODEL data...")
    data = load_big_model_data(data_dir)

    perm_x = data["perm_x"]
    perm_y = data["perm_y"]
    perm_z = data["perm_z"]
    x = data["x"]
    y = data["y"]
    z = data["z"]

    print(f"Grid dimensions: {data['nx']} x {data['ny']} x {data['nz']}")
    print(f"Permeability range: {perm_x.min():.2f} - {perm_x.max():.2f} mD")

    print("Exporting permeability and porosity...")
    base_vtk = os.path.join(output_dir, "big_model_properties.vtk")

    with open(base_vtk, "w") as f:
        from src.vtk_export import write_vtk_header, write_vtk_points_3d
        from src.vtk_export import write_vtk_point_data_header, write_vtk_scalar

        write_vtk_header(f, "big_model_properties.vtk")
        write_vtk_points_3d(f, x, y, z)

        n_points = data["nx"] * data["ny"] * data["nz"]
        write_vtk_point_data_header(f, n_points)

        write_vtk_scalar(f, "permeability_x", perm_x.flatten(order="F"))
        write_vtk_scalar(f, "permeability_y", perm_y.flatten(order="F"))
        write_vtk_scalar(f, "permeability_z", perm_z.flatten(order="F"))
        write_vtk_scalar(f, "porosity", data["porosity"].flatten(order="F"))

    print(f"Exported: {base_vtk}")

    if saturation is not None:
        sat_file = os.path.join(output_dir, "big_model_saturation.vtk")
        export_3d_saturation(perm_x, perm_y, perm_z, saturation, sat_file, x, y, z)
        print(f"Exported: {sat_file}")

    if pressure is not None:
        pres_file = os.path.join(output_dir, "big_model_pressure.vtk")
        export_3d_pressure(perm_x, perm_y, perm_z, pressure, pres_file, x, y, z)
        print(f"Exported: {pres_file}")

    create_resinsight_batch_script(output_dir, data["nx"], data["ny"], data["nz"])

    create_resinsight_readme(output_dir)

    print(f"\nAll files exported to: {output_dir}")
    print("Open ResInsight and load the .vtk files to visualize")


def create_resinsight_batch_script(output_dir: str, nx: int, ny: int, nz: int) -> None:
    """Create batch script for loading data into ResInsight."""
    script_path = os.path.join(output_dir, "load_in_resinsight.sh")

    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Batch script to load BIG_MODEL data into ResInsight\n")
        f.write("# Usage: resinsight --project-dir . &\n")
        f.write("\n")
        f.write("# ResInsight command line options:\n")
        f.write("#   resinsight [--project-dir <dir>] [<file>.vtk ...]\n")
        f.write("\n")
        f.write(f"# Grid dimensions: {nx} x {ny} x {nz}\n")
        f.write("\n")
        f.write("# To load all VTK files:\n")
        f.write("#   resinsight outputs/resinsight/*.vtk\n")
        f.write("\n")
        f.write("# Or open ResInsight GUI and:\n")
        f.write("#   1. File -> Open -> Select .vtk files\n")
        f.write("#   2. Project -> New -> Import Eclipse Files\n")
        f.write("#   3. Use ResInsight Python API for automation\n")

    os.chmod(script_path, 0o755)
    print(f"Created batch script: {script_path}")


def create_resinsight_readme(output_dir: str) -> None:
    """Create README with instructions for ResInsight."""
    readme_path = os.path.join(output_dir, "README.md")

    with open(readme_path, "w") as f:
        f.write("# ResInsight Visualization - BIG_MODEL\n\n")
        f.write("## Overview\n\n")
        f.write(
            "This directory contains VTK files exported from the BIG_MODEL_12_09_1\n"
        )
        f.write("SPE10 dataset for 3D visualization in ResInsight.\n\n")
        f.write("## Files\n\n")
        f.write("- `big_model_properties.vtk` - Permeability and porosity fields\n")
        f.write("- `big_model_saturation.vtk` - Saturation field (if exported)\n")
        f.write("- `big_model_pressure.vtk` - Pressure field (if exported)\n\n")
        f.write("## How to Open in ResInsight\n\n")
        f.write("### Method 1: Command Line\n\n")
        f.write("```bash\n")
        f.write("# Open all VTK files at once\n")
        f.write("resinsight outputs/resinsight/*.vtk\n\n")
        f.write("# Or with project directory\n")
        f.write("resinsight --project-dir outputs/resinsight/\n")
        f.write("```\n\n")
        f.write("### Method 2: GUI\n\n")
        f.write("1. Launch ResInsight\n")
        f.write("2. File -> Open -> Navigate to `outputs/resinsight/`\n")
        f.write("3. Select `.vtk` files and click Open\n")
        f.write("4. Files appear in Project Tree\n\n")
        f.write("### Method 3: Eclipse Files (Alternative)\n\n")
        f.write("ResInsight natively supports Eclipse format:\n\n")
        f.write("1. File -> Import Eclipse Files\n")
        f.write("2. Select `BIG_MODEL_12_09_1.INIT` and `BIG_MODEL_12_09_1.UNRST`\n")
        f.write("3. ResInsight will load all properties automatically\n\n")
        f.write("## Recommended Visualization Settings\n\n")
        f.write("### Permeability (log scale)\n\n")
        f.write("- Use log scale colormap for permeability\n")
        f.write("- Recommended range: 0.1 - 1000 mD\n")
        f.write("- Colormap: `Rainbow` or `Viridis`\n\n")
        f.write("### Saturation\n\n")
        f.write("- Range: 0.0 - 1.0 (or 0-1 fraction)\n")
        f.write("- Colormap: `Blue-White-Red` (water=blue, oil=red)\n")
        f.write("- Enable opacity for values < 0.1 (porosity threshold)\n\n")
        f.write("### Pressure\n\n")
        f.write("- Range: depends on simulation (typically 0-1e5 Pa)\n")
        f.write("- Colormap: `Rainbow` or `Coolwarm`\n\n")
        f.write("## ResInsight Python API\n\n")
        f.write("```python\n")
        f.write("import rips\n\n")
        f.write("connection = rips.Connection.open()\n")
        f.write("project = connection.project.create()\n\n")
        f.write("# Import VTK files\n")
        f.write("project.import_source_files(['big_model_properties.vtk'])\n")
        f.write("```\n\n")
        f.write("## Grid Information\n\n")
        f.write("- Dimensions: 122 (x) × 183 (y) × 43 (z)\n")
        f.write("- Grid spacing: 1 ft (x, y), 2 ft (z)\n")
        f.write("- Total cells: 961,698\n\n")

    print(f"Created README: {readme_path}")


def generate_synthetic_saturation(
    perm_x: onp.ndarray, porosity: onp.ndarray, injection_x: int = 10
) -> onp.ndarray:
    nx, ny, nz = perm_x.shape
    saturation = onp.zeros((nx, ny, nz))

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                dist = ((i - injection_x) ** 2 + (j - ny // 2) ** 2) ** 0.5
                sat_val = onp.exp(-dist / 20.0) * (perm_x[i, j, k] / perm_x.max())
                saturation[i, j, k] = min(max(sat_val, 0.2), 0.8)

    return saturation


def generate_synthetic_pressure(
    perm_x: onp.ndarray, injection_x: int = 10, pressure_inj: float = 1e5
) -> onp.ndarray:
    nx, ny, nz = perm_x.shape
    pressure = onp.zeros((nx, ny, nz))

    for i in range(nx):
        pressure[i, :, :] = pressure_inj * (1.0 - i / nx)

    return pressure


if __name__ == "__main__":
    export_for_resinsight()
