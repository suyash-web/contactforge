
import pytest
import numpy as np
from contactforge.simulation.runner import run_scene
from contactforge.diagnostics.convergence import (
    analyze_scene,
    compare_runs,
)

SCENE = "examples/scenes/falling_box.xml"


def test_falling_box_convergence():
    report = analyze_scene(
        SCENE,
        timesteps=[0.002, 0.001, 0.0005],
        duration=0.5,
        sample_interval=0.01,
    )

    if report["reference_timestep_ms"] != 0.5:
        pytest.fail("Incorrect numerical reference.")

    if len(report["comparisons"]) != 2:
        pytest.fail("Expected two comparisons.")

    if report["tracked_bodies"] != ["falling_box"]:
        pytest.fail("Unexpected tracked bodies.")

    for comparison in report["comparisons"]:
        if comparison["max_trajectory_error_mm"] < 0:
            pytest.fail("Trajectory error cannot be negative.")


def test_rejects_duplicate_timesteps():
    with pytest.raises(ValueError, match="Duplicate"):
        analyze_scene(
            SCENE,
            timesteps=[0.001, 0.001],
        )


def test_rejects_single_timestep():
    with pytest.raises(ValueError, match="At least two"):
        analyze_scene(
            SCENE,
            timesteps=[0.001],
        )


def test_rejects_body_mismatch():
    candidate = {
        "body_names": ["box_a"],
        "observations": [],
    }

    reference = {
        "body_names": ["box_b"],
        "observations": [],
    }

    with pytest.raises(ValueError, match="body names"):
        compare_runs(candidate, reference)


def make_pose_run(quaternion):
    return {
        "body_names": ["test_body"],
        "observations": [
            {
                "time_s": 0.0,
                "positions_m": [[0.0, 0.0, 0.1]],
                "orientations_wxyz": [quaternion],
                "contact_count": 0,
                "contact_pairs": [],
            }
        ],
    }


def test_quaternion_sign_equivalence():
    candidate = make_pose_run([1.0, 0.0, 0.0, 0.0])
    reference = make_pose_run([-1.0, 0.0, 0.0, 0.0])

    result = compare_runs(candidate, reference)

    if result["max_orientation_error_degrees"] > 1e-6:
        pytest.fail(
            "Equivalent quaternions produced nonzero error."
        )


def test_detects_ninety_degree_rotation():
    import math

    angle = math.pi / 2.0

    candidate = make_pose_run([
        math.cos(angle / 2.0),
        0.0,
        0.0,
        math.sin(angle / 2.0),
    ])

    reference = make_pose_run([1.0, 0.0, 0.0, 0.0])

    result = compare_runs(candidate, reference)

    if not math.isclose(
        result["max_orientation_error_degrees"],
        90.0,
        abs_tol=1e-6,
    ):
        pytest.fail(
            "Expected a 90-degree orientation difference."
        )

    if result["max_trajectory_error_mm"] != 0.0:
        pytest.fail(
            "Position error should remain zero."
        )

def test_tilted_box_initial_orientation():
    result = run_scene(
        "examples/scenes/tilted_box.xml",
        timestep=0.001,
        duration=0.1,
        sample_interval=0.01,
    )

    initial = result["observations"][0]

    actual = np.array(
        initial["orientations_wxyz"][0]
    )

    expected = np.array([
        0.984807753,
        0.0,
        0.173648178,
        0.0,
    ])

    if not np.allclose(actual, expected, atol=1e-6):
        pytest.fail(
            "Scene runner did not preserve initial orientation."
        )


def test_initial_freejoint_velocity_is_applied():
    result = run_scene(
        "examples/scenes/tilted_box.xml",
        timestep=0.001,
        duration=0.1,
        sample_interval=0.01,
        initial_freejoint_velocities={
            "box_freejoint": [0.05, 0, 0, 0, 0.5, 0]
        },
    )

    actual = result["initial_freejoint_velocities"]

    if actual["box_freejoint"] != [0.05, 0, 0, 0, 0.5, 0]:
        pytest.fail("Initial freejoint velocity was not recorded.")


def test_rejects_invalid_freejoint_velocity():
    with pytest.raises(
        ValueError,
        match="requires six velocities",
    ):
        run_scene(
            "examples/scenes/tilted_box.xml",
            timestep=0.001,
            duration=0.1,
            sample_interval=0.01,
            initial_freejoint_velocities={
                "box_freejoint": [0.05, 0, 0]
            },
        )


def test_requested_velocity_matches_applied_state():
    requested = [0.05, 0.0, 0.0, 0.0, 0.5, 0.0]

    result = run_scene(
        "examples/scenes/tilted_box.xml",
        timestep=0.001,
        duration=0.1,
        sample_interval=0.01,
        initial_freejoint_velocities={
            "box_freejoint": requested
        },
    )

    applied = result["applied_initial_velocities"][
        "box_freejoint"
    ]

    if not np.allclose(applied, requested, atol=1e-12):
        pytest.fail(
            f"Requested velocity {requested}, "
            f"but applied velocity was {applied}"
        )


def test_initial_motion_changes_trajectory():
    common = {
        "scene_path": "examples/scenes/tilted_box.xml",
        "timestep": 0.001,
        "duration": 0.1,
        "sample_interval": 0.01,
    }

    stationary = run_scene(**common)

    moving = run_scene(
        **common,
        initial_freejoint_velocities={
            "box_freejoint": [0.05, 0, 0, 0, 0.5, 0]
        },
    )

    stationary_final = np.asarray(
        stationary["observations"][-1]["positions_m"]
    )

    moving_final = np.asarray(
        moving["observations"][-1]["positions_m"]
    )

    difference = np.linalg.norm(
        moving_final - stationary_final
    )

    if difference <= 1e-5:
        pytest.fail(
            "Applying initial motion did not change the trajectory."
        )


def test_rejects_unknown_initial_velocity_joint():
    with pytest.raises(ValueError, match="Unknown joint"):
        run_scene(
            "examples/scenes/tilted_box.xml",
            timestep=0.001,
            duration=0.1,
            sample_interval=0.01,
            initial_freejoint_velocities={
                "nonexistent_joint": [0, 0, 0, 0, 0, 1]
            },
        )


def test_stage_control_schedule_is_applied():
    result = run_scene(
        "examples/scenes/actuated_stage.xml",
        timestep=0.001,
        duration=0.8,
        sample_interval=0.01,
        control_schedule={
            "stage_motor": [
                {"time_s": 0.0, "value": 1.0},
                {"time_s": 0.2, "value": -1.0},
                {"time_s": 0.5, "value": 0.0},
            ]
        },
    )

    expected = {
        0.0: 1.0,
        0.2: -1.0,
        0.5: 0.0,
        0.8: 0.0,
    }

    for target_time, expected_control in expected.items():
        observations = [
            item for item in result["observations"]
            if abs(item["time_s"] - target_time) < 1e-8
        ]

        if len(observations) != 1:
            pytest.fail(
                f"Missing observation at {target_time} seconds."
            )

        actual = observations[0]["controls"]["stage_motor"]

        if not np.isclose(actual, expected_control):
            pytest.fail(
                f"At {target_time}s: expected "
                f"{expected_control}, got {actual}"
            )


def test_stage_actuation_changes_trajectory():
    common = {
        "scene_path": "examples/scenes/actuated_stage.xml",
        "timestep": 0.001,
        "duration": 0.3,
        "sample_interval": 0.01,
    }

    uncontrolled = run_scene(**common)

    controlled = run_scene(
        **common,
        control_schedule={
            "stage_motor": [
                {"time_s": 0.0, "value": 1.0}
            ]
        },
    )

    p0 = np.asarray(
        uncontrolled["observations"][-1]["positions_m"]
    )

    p1 = np.asarray(
        controlled["observations"][-1]["positions_m"]
    )

    if np.linalg.norm(p1 - p0) < 0.001:
        pytest.fail(
            "Motor command did not change stage position."
        )


def test_rejects_unaligned_control_event():
    with pytest.raises(ValueError, match="align with timestep"):
        run_scene(
            "examples/scenes/actuated_stage.xml",
            timestep=0.002,
            duration=0.1,
            control_schedule={
                "stage_motor": [
                    {"time_s": 0.003, "value": 1.0}
                ]
            },
        )
