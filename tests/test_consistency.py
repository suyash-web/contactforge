
import numpy as np
import pytest

from contactforge.assets.consistency import compare_inertia
from contactforge.assets.inertia import box_inertia


MESH = "examples/meshes/valid_box.stl"
OPEN_MESH = "examples/meshes/open_box.stl"


def test_matching_geometry_and_inertia():
    mass = 2.0
    inertia = box_inertia(mass, 0.2, 0.3, 0.4)

    report = compare_inertia(MESH, mass, inertia)

    if report["status"] != "consistent":
        raise AssertionError(
            f"Valid inertia was rejected: {report['relative_error']}"
        )

    if report["relative_error"] > 1e-5:
        raise AssertionError("Unexpected geometric inertia error.")


def test_inconsistent_inertia_detected():
    mass = 2.0
    inertia = box_inertia(mass, 0.2, 0.3, 0.4) * 10

    report = compare_inertia(MESH, mass, inertia)

    if report["status"] != "mismatch":
        raise AssertionError(
            "Incorrect inertia was not detected."
        )

    if not np.isclose(report["relative_error"], 9.0, rtol=1e-5):
        raise AssertionError(
            f"Unexpected error: {report['relative_error']}"
        )


def test_open_mesh_rejected():
    inertia = box_inertia(2.0, 0.2, 0.3, 0.4)

    with pytest.raises(ValueError, match="watertight"):
        compare_inertia(OPEN_MESH, 2.0, inertia)
