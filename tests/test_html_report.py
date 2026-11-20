"""Milestone 41: offline reporting regression tests."""
import json

import pytest

from contactforge.reporting.html_report import generate_report, line_chart
from contactforge.__main__ import MODULE_COMMANDS, build_parser


def fixture_reports(tmp_path):
    stage = {
        "scene": "stage.xml",
        "reference_timestep_ms": 0.25,
        "duration_s": 0.8,
        "sample_interval_s": 0.01,
        "comparisons": [{
            "timestep_ms": 1.0,
            "mean_trajectory_error_mm": 0.075,
            "max_trajectory_error_mm": 0.146,
            "max_orientation_error_degrees": 0.0,
            "contact_pair_mismatch_percent": 0.0,
        }],
        "limitations": ["Numerical reference is not truth."],
    }
    reliability = {
        "scene": "indentation.xml",
        "reference_timestep_ms": 0.25,
        "duration_s": 1.0,
        "sample_interval_s": 0.01,
        "model_stiffness_N_per_m": 1500,
        "comparisons": [
            {
                "timestep_ms": 1.0,
                "reference": False,
                "peak_abs_contact_force_difference_N": 0.1286,
                "quasistatic_stiffness_estimate_N_per_m": 1499.9,
                "quasistatic_stiffness_error_percent": 0.0067,
                "max_abs_force_balance_residual_N": 0.0,
            },
            {
                "timestep_ms": 0.25,
                "reference": True,
                "quasistatic_stiffness_estimate_N_per_m": 1499.9,
                "quasistatic_stiffness_error_percent": 0.0067,
                "max_abs_force_balance_residual_N": 0.0,
            },
        ],
        "limitations": ["Contact onset can dominate sample error."],
    }
    force = {
        "scene": "indentation.xml",
        "duration_s": 1.0,
        "sample_interval_s": 0.01,
        "timestep_s": 0.0005,
        "measurements": [
            {"time_s": 0.0, "contact_normal_force_N": 0.0, "compression_mm": 0.0},
            {"time_s": 0.5, "contact_normal_force_N": 10.0, "compression_mm": 6.0},
            {"time_s": 1.0, "contact_normal_force_N": 12.0, "compression_mm": 8.0},
        ],
    }
    paths = []
    for name, data in (("stage", stage), ("reliability", reliability), ("force", force)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        paths.append(path)
    return paths


def test_html_report_is_self_contained(tmp_path):
    stage, reliability, force = fixture_reports(tmp_path)
    destination = tmp_path / "nested" / "report.html"
    assert generate_report(stage, reliability, force, destination) == destination
    document = destination.read_text()
    assert "<!doctype html>" in document
    assert document.count("<svg") == 5
    assert "1499.90" in document
    assert "0.1286" in document
    assert "Contact onset can dominate sample error." in document
    assert "<script" not in document
    assert "https://" not in document


def test_missing_input_field_is_rejected(tmp_path):
    stage, reliability, force = fixture_reports(tmp_path)
    stage.write_text(json.dumps({"scene": "stage.xml"}))
    with pytest.raises(ValueError, match="missing fields"):
        generate_report(stage, reliability, force, tmp_path / "out.html")


def test_svg_rejects_nonfinite_data():
    with pytest.raises(ValueError, match="finite"):
        line_chart([(0.0, float("nan"))], "time", "force")


def test_report_command_registered():
    assert MODULE_COMMANDS["report"]["module"] == "contactforge.reporting.html_report"
    assert build_parser().parse_args(["report"]).command == "report"
