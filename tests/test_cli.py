
import json
import subprocess
import sys
import pytest
from pathlib import Path
from types import SimpleNamespace
from contactforge.__main__ import (
    MODULE_COMMANDS,
    build_parser,
    main,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "contactforge", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )


def test_invalid_model_generates_report(tmp_path):
    report_path = tmp_path / "validation.json"

    result = run_cli(
        "inspect",
        "examples/test_robot.xml",
        "--output",
        str(report_path)
    )

    if result.returncode != 1:
        raise AssertionError(
            f"Expected validation failure.\n{result.stderr}"
        )

    if not report_path.exists():
        raise AssertionError("JSON report was not created.")

    report = json.loads(report_path.read_text())

    if report["failed"] != 1:
        raise AssertionError("Expected one invalid body.")

    if report["skipped"] != 1:
        raise AssertionError("Expected one skipped body.")


def test_missing_model_returns_error():
    result = run_cli(
        "inspect",
        "examples/nonexistent.xml"
    )

    if result.returncode != 2:
        raise AssertionError(
            "Missing model should return exit code 2."
        )

    if "MJCF file not found" not in result.stderr:
        raise AssertionError(
            "Expected a descriptive missing-file error."
        )

def test_original_asset_commands_preserved():
    parser = build_parser()

    mjcf = parser.parse_args([
        "inspect",
        "example.xml",
    ])

    mesh = parser.parse_args([
        "inspect-mesh",
        "example.stl",
    ])

    if mjcf.command != "inspect":
        pytest.fail("MJCF inspection command missing.")

    if mesh.command != "inspect-mesh":
        pytest.fail("Mesh inspection command missing.")


def test_new_commands_registered():
    required = {
        "simulate",
        "analyze",
        "indentation",
        "reliability",
        "plot",
    }

    if not required.issubset(MODULE_COMMANDS):
        pytest.fail("Required commands are missing.")


def test_new_command_forwards_arguments(monkeypatch):
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "contactforge.__main__.subprocess.run",
        fake_run,
    )

    result = main([
        "simulate",
        "scene.xml",
        "--timestep",
        "0.0005",
    ])

    expected = [
        sys.executable,
        "-m",
        "contactforge.simulation.runner",
        "scene.xml",
        "--timestep",
        "0.0005",
    ]

    if captured["command"] != expected:
        pytest.fail("Arguments were forwarded incorrectly.")

    if result != 0:
        pytest.fail("Expected successful exit code.")


def test_subprocess_failure_is_propagated(monkeypatch):
    def fake_run(command, check):
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(
        "contactforge.__main__.subprocess.run",
        fake_run,
    )

    result = main([
        "analyze",
        "missing.xml",
    ])

    if result != 7:
        pytest.fail("Backend failure code was not preserved.")


def test_original_inspect_dispatch(monkeypatch):
    called = {}

    def fake_inspect(args):
        called["model"] = args.model
        return 0

    monkeypatch.setattr(
        "contactforge.__main__.inspect_command",
        fake_inspect,
    )

    result = main([
        "inspect",
        "model.xml",
    ])

    if result != 0 or called.get("model") != "model.xml":
        pytest.fail("Original inspection dispatch is broken.")
