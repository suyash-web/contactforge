
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "contactforge", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )


def test_valid_mesh_cli():
    result = run_cli(
        "inspect-mesh",
        "examples/meshes/valid_box.stl"
    )

    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)

    if "PASSED" not in result.stdout:
        raise AssertionError("Expected successful mesh validation.")


def test_invalid_mesh_cli(tmp_path):
    output = tmp_path / "open_box.json"

    result = run_cli(
        "inspect-mesh",
        "examples/meshes/open_box.stl",
        "--output",
        str(output)
    )

    if result.returncode != 1:
        raise AssertionError(
            f"Expected validation failure:\n{result.stdout}"
        )

    if not output.exists():
        raise AssertionError("JSON report was not created.")

    report = json.loads(output.read_text())

    if report["watertight"]:
        raise AssertionError("Open mesh reported as watertight.")

    if report["status"] != "failed":
        raise AssertionError("Expected failed status.")


def test_missing_mesh_cli():
    result = run_cli(
        "inspect-mesh",
        "examples/meshes/missing.stl"
    )

    if result.returncode != 2:
        raise AssertionError(
            "Missing mesh should return exit code 2."
        )
