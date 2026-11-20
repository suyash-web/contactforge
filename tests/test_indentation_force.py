
import math

import pytest

from experiments.indentation_force import run_indentation


SCENE = "examples/scenes/controlled_indentation.xml"
CONTROLS = "examples/configs/indentation_controls.json"


@pytest.fixture(scope="module")
def indentation_report():
    return run_indentation(
        SCENE,
        CONTROLS,
        timestep=0.001,
        duration=1.0,
        sample_interval=0.01,
    )


def test_indentation_produces_contact(indentation_report):
    measurements = indentation_report["measurements"]

    if len(measurements) != 101:
        pytest.fail("Expected 101 measurements.")

    contact_rows = [
        row for row in measurements
        if row["contact_points"] > 0
    ]

    if not contact_rows:
        pytest.fail("Indenter never contacted the specimen.")

    first_contact_time = contact_rows[0]["time_s"]

    if not 0.4 < first_contact_time < 0.5:
        pytest.fail(
            f"Unexpected contact time: {first_contact_time}"
        )


def test_final_indentation_force_is_consistent(
    indentation_report,
):
    final = indentation_report["measurements"][-1]

    compression_m = final["compression_mm"] / 1000
    stiffness = indentation_report["stiffness_N_per_m"]

    expected_spring_force = stiffness * compression_m
    actual_spring_force = final["spring_force_N"]
    actual_contact_force = final["contact_normal_force_N"]

    if compression_m <= 0:
        pytest.fail("Specimen has no final compression.")

    if actual_contact_force <= 0:
        pytest.fail("Expected positive contact force.")

    if not math.isclose(
        actual_spring_force,
        expected_spring_force,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        pytest.fail("Recorded spring force is inconsistent.")

    # Final state should be close to static force balance.
    # The 5% tolerance accommodates residual dynamics.
    if not math.isclose(
        actual_contact_force,
        expected_spring_force,
        rel_tol=0.05,
    ):
        pytest.fail(
            "Final contact force is inconsistent "
            "with spring restoring force."
        )


def test_initial_state_has_no_contact(indentation_report):
    first = indentation_report["measurements"][0]

    if first["contact_points"] != 0:
        pytest.fail("Unexpected initial contact.")

    if abs(first["contact_normal_force_N"]) > 1e-9:
        pytest.fail("Unexpected initial contact force.")

    if abs(first["compression_mm"]) > 1e-9:
        pytest.fail("Unexpected initial specimen compression.")
