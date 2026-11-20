
import csv
import json

import pytest

from contactforge.diagnostics.report import build_report


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def create_test_results(directory):
    trajectory = [
        {
            "bodies": 8,
            "timestep_ms": 1.0,
            "reference_timestep_ms": 0.125,
            "mean_trajectory_error_mm": 0.4,
            "maximum_trajectory_error_mm": 3.2,
            "final_position_error_mm": 0.5,
        },
        {
            "bodies": 16,
            "timestep_ms": 2.0,
            "reference_timestep_ms": 0.125,
            "mean_trajectory_error_mm": 2.9,
            "maximum_trajectory_error_mm": 13.9,
            "final_position_error_mm": 1.1,
        },
    ]

    contact = [
        {
            "bodies": 8,
            "timestep_ms": 1.0,
            "contact_count_mismatch_percent": 54.6,
            "same_pairs_different_counts_percent": 53.4,
            "different_pairs_percent": 1.2,
        },
        {
            "bodies": 16,
            "timestep_ms": 2.0,
            "contact_count_mismatch_percent": 78.5,
            "same_pairs_different_counts_percent": 72.9,
            "different_pairs_percent": 5.6,
        },
    ]

    write_csv(
        directory / "stack_trajectory.csv",
        trajectory,
    )

    write_csv(
        directory / "contact_pair_diagnostics.csv",
        contact,
    )

    selection = {
        "criteria": {
            "max_stiffness_error_percent": 0.02,
            "max_contact_overlap_mm": 1.0,
        },
        "recommended": {
            "timestep_ms": 0.5,
            "contact_time_constant_ms": 2.5,
        },
        "accepted_count": 4,
        "tested": 9,
    }

    (directory / "selected_configuration.json").write_text(
        json.dumps(selection),
        encoding="utf-8",
    )


def test_report_aggregates_results(tmp_path):
    create_test_results(tmp_path)

    report = build_report(tmp_path)

    if report["reference_timestep_ms"] != 0.125:
        pytest.fail("Incorrect reference timestep.")

    stack = report["contact_stack_analysis"]

    if len(stack["configurations"]) != 2:
        pytest.fail("Expected two configurations.")

    worst = stack["worst_trajectory_configuration"]

    if worst["bodies"] != 16:
        pytest.fail("Incorrect worst trajectory configuration.")

    if worst["maximum_trajectory_error_mm"] != 13.9:
        pytest.fail("Incorrect maximum trajectory error.")


def test_report_rejects_inconsistent_reference(tmp_path):
    create_test_results(tmp_path)

    path = tmp_path / "stack_trajectory.csv"

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    rows[1]["reference_timestep_ms"] = 0.25
    write_csv(path, rows)

    with pytest.raises(
        ValueError,
        match="inconsistent reference timesteps",
    ):
        build_report(tmp_path)


def test_report_rejects_missing_contact_configuration(tmp_path):
    create_test_results(tmp_path)

    path = tmp_path / "contact_pair_diagnostics.csv"

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    write_csv(path, rows[:1])

    with pytest.raises(
        ValueError,
        match="Missing contact results",
    ):
        build_report(tmp_path)


def test_report_rejects_duplicate_trajectory_configuration(tmp_path):
    create_test_results(tmp_path)

    path = tmp_path / "stack_trajectory.csv"

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    rows.append(dict(rows[0]))
    write_csv(path, rows)

    with pytest.raises(
        ValueError,
        match="Duplicate trajectory configuration",
    ):
        build_report(tmp_path)
