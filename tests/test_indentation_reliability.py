
import math

import pytest

from experiments.indentation_reliability import (
    analyze,
    simulate,
)


SCENE = "examples/scenes/controlled_indentation.xml"

CONTROLS = {
    "stage_motor": [
        {"time_s": 0.0, "value": 0.0},
        {"time_s": 0.2, "value": 0.015},
        {"time_s": 0.4, "value": 0.04},
    ]
}


@pytest.fixture(scope="module")
def short_run():
    return simulate(
        SCENE,
        CONTROLS,
        timestep=0.001,
        duration=0.6,
        sample_interval=0.01,
    )


def test_force_measurements_are_finite(short_run):
    rows = short_run["measurements"]

    if len(rows) != 61:
        pytest.fail("Expected 61 observations.")

    fields = [
        "spring_up_N",
        "damping_up_N",
        "inertia_up_N",
        "contact_down_N",
        "force_balance_residual_N",
    ]

    for row in rows:
        for field in fields:
            if not math.isfinite(row[field]):
                pytest.fail(
                    f"Non-finite value in {field}."
                )


def test_indentation_generates_contact_force(short_run):
    rows = short_run["measurements"]

    initial = rows[0]

    if initial["contact_points"] != 0:
        pytest.fail("Unexpected initial contact.")

    if initial["contact_down_N"] != 0:
        pytest.fail("Unexpected initial contact force.")

    if not any(
        row["contact_down_N"] > 0
        for row in rows
    ):
        pytest.fail(
            "No positive indentation contact force detected."
        )


def test_reliability_report_compares_timesteps():
    report = analyze(
        SCENE,
        CONTROLS,
        timesteps=[0.002, 0.001],
        duration=0.6,
        sample_interval=0.01,
    )

    if report["reference_timestep_ms"] != 1.0:
        pytest.fail("Incorrect reference timestep.")

    if len(report["comparisons"]) != 2:
        pytest.fail("Expected two timestep summaries.")

    comparisons = report["comparisons"]

    reference = [
        item for item in comparisons
        if item["reference"]
    ]

    candidates = [
        item for item in comparisons
        if not item["reference"]
    ]

    if len(reference) != 1 or len(candidates) != 1:
        pytest.fail(
            "Expected one reference and one candidate."
        )

    if not math.isfinite(
        candidates[0]["peak_abs_contact_force_difference_N"]
    ):
        pytest.fail("Invalid force convergence metric.")
