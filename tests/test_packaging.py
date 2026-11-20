"""Release packaging and installed-command smoke tests.

Run after ``python -m pip install -e '.[dev]'``.
"""

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_and_entry_point():
    with (ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)

    assert project["project"]["name"] == "contactforge"
    assert project["project"]["version"] == "1.0.0"
    assert project["project"]["scripts"]["contactforge"] == (
        "contactforge.cli:main"
    )
    assert "experiments" in project["tool"]["setuptools"]["packages"]


def test_experiment_modules_are_importable():
    for module in (
        "experiments.indentation_force",
        "experiments.indentation_reliability",
        "experiments.plot_indentation",
    ):
        assert importlib.util.find_spec(module) is not None, module


def test_installed_package_works_outside_project(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "contactforge", "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "inspect-mesh" in result.stdout
    assert "report" in result.stdout


def test_installed_experiment_command_works_outside_project(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "contactforge", "reliability", "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--controls" in result.stdout


def test_console_entry_point_matches_existing_cli():
    from contactforge.cli import main
    from contactforge.__main__ import main as original_main

    assert main.__module__ == "contactforge.cli"
    assert callable(original_main)
