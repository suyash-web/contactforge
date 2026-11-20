
from pathlib import Path

import numpy as np
import trimesh


def inspect_mesh(file_path):
    """Inspect mesh geometry before simulation asset import."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Mesh not found: {path}")

    try:
        loaded = trimesh.load(path, process=False)
    except Exception as error:
        raise ValueError(
            f"Unable to load mesh: {error}"
        ) from error

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(
            "Expected one triangle mesh, not a scene or point cloud."
        )

    # Preserve the imported geometry and work on a separate copy.
    # STL files may contain duplicate vertices for adjacent triangles.
    # Merge coincident vertices before evaluating mesh connectivity.
    mesh = loaded.copy()
    original_vertex_count = len(mesh.vertices)

    if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
        if np.all(np.isfinite(mesh.vertices)):
            mesh.merge_vertices()

    issues = []

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        issues.append("Mesh contains no usable geometry.")
        return {
            "file": str(path),
            "status": "failed",
            "issues": issues
        }

    if not np.all(np.isfinite(mesh.vertices)):
        issues.append("Mesh contains non-finite coordinates.")
        return {
            "file": str(path),
            "status": "failed",
            "issues": issues
        }

    dimensions = mesh.extents

    if np.any(dimensions <= 0):
        issues.append("Mesh has a degenerate bounding box.")

    if not mesh.is_watertight:
        issues.append(
            "Mesh is not watertight; enclosed volume "
            "cannot be trusted."
        )

    if not mesh.is_winding_consistent:
        issues.append("Mesh faces have inconsistent winding.")

    if not mesh.is_volume:
        issues.append(
            "Mesh does not define a valid consistently "
            "oriented solid volume."
        )

    report = {
        "file": str(path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "valid_volume": bool(mesh.is_volume),
        "dimensions_m_assuming_meters": dimensions.tolist(),
        "original_vertices": original_vertex_count,
        "merged_vertices": int(len(mesh.vertices)),
        "issues": issues,
        "status": "passed" if not issues else "failed"
    }

    if mesh.is_volume:
        report["volume_mesh_units_cubed"] = float(mesh.volume)
        report["center_of_volume_mesh_units"] = (
            mesh.center_mass.tolist()
        )

    return report
