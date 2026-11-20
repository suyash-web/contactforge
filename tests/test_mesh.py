import numpy as np
import trimesh

from contactforge.assets.mesh import inspect_mesh


def test_closed_stl_is_watertight(tmp_path):
    mesh = trimesh.creation.box(
        extents=[0.2, 0.3, 0.4]
    )

    path = tmp_path / "closed_box.stl"
    mesh.export(path)

    report = inspect_mesh(path)

    if report["status"] != "passed":
        raise AssertionError(
            f"Closed box incorrectly rejected: {report['issues']}"
        )

    if not report["watertight"]:
        raise AssertionError(
            "Closed STL mesh should be watertight."
        )

    if not np.isclose(
        report["volume_mesh_units_cubed"],
        0.024,
        rtol=1e-5
    ):
        raise AssertionError("Incorrect box volume.")


def test_open_stl_is_rejected(tmp_path):
    mesh = trimesh.creation.box(
        extents=[0.2, 0.3, 0.4]
    )

    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[0] = False

    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()

    path = tmp_path / "open_box.stl"
    mesh.export(path)

    report = inspect_mesh(path)

    if report["status"] != "failed":
        raise AssertionError(
            "Open mesh should fail validation."
        )

    if report["watertight"]:
        raise AssertionError(
            "Open mesh should not be watertight."
        )


def test_missing_mesh(tmp_path):
    path = tmp_path / "missing.stl"

    try:
        inspect_mesh(path)
    except FileNotFoundError:
        return

    raise AssertionError(
        "Missing mesh should raise FileNotFoundError."
    )
