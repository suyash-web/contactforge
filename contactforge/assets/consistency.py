
from pathlib import Path

import numpy as np
import trimesh


def compare_inertia(mesh_path, mass, declared_inertia, tolerance=0.05):
    """
    Compare declared rigid-body inertia with mesh-derived inertia.

    Assumptions:
    - The mesh is a closed, consistently oriented solid.
    - Mesh coordinates are in meters.
    - Mass is distributed uniformly through the mesh.
    - Both tensors use the mesh coordinate axes.
    - Both tensors are measured about the center of mass.
    """
    path = Path(mesh_path)

    if not path.is_file():
        raise FileNotFoundError(f"Mesh not found: {path}")

    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Mass must be positive and finite.")

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative.")

    declared = np.asarray(declared_inertia, dtype=float)

    if declared.shape != (3, 3):
        raise ValueError("Declared inertia must be a 3x3 matrix.")

    if not np.all(np.isfinite(declared)):
        raise ValueError("Declared inertia contains invalid numbers.")

    if not np.allclose(declared, declared.T, rtol=0, atol=1e-12):
        raise ValueError("Declared inertia must be symmetric.")

    mesh = trimesh.load(path, process=False)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected a single triangle mesh.")

    mesh = mesh.copy()

    if not np.all(np.isfinite(mesh.vertices)):
        raise ValueError("Mesh contains invalid coordinates.")

    mesh.merge_vertices()

    if not mesh.is_watertight or not mesh.is_volume:
        raise ValueError(
            "A watertight, consistently oriented solid is required."
        )

    volume = float(mesh.volume)

    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Mesh volume must be positive.")

    # Trimesh initially uses unit density.
    # Inertia scales linearly with material density.
    density = mass / volume
    mesh.density = density

    expected = np.asarray(mesh.moment_inertia, dtype=float)

    difference = declared - expected

    reference_norm = np.linalg.norm(expected, ord="fro")

    if reference_norm <= 0 or not np.isfinite(reference_norm):
        raise ValueError("Unable to compute reference inertia.")

    relative_error = float(
        np.linalg.norm(difference, ord="fro") / reference_norm
    )

    return {
        "mesh": str(path),
        "mass_kg": float(mass),
        "volume_m3": volume,
        "estimated_density_kg_m3": float(density),
        "mesh_center_of_mass_m": mesh.center_mass.tolist(),
        "expected_inertia": expected.tolist(),
        "declared_inertia": declared.tolist(),
        "relative_error": relative_error,
        "tolerance": float(tolerance),
        "status": "consistent" if relative_error <= tolerance
                  else "mismatch",
        "assumption": "Uniform density; common COM and axes"
    }
