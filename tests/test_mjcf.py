import xml.etree.ElementTree as ET

import numpy as np
import pytest

from contactforge.assets.mjcf import (
    inspect_mjcf,
    parse_inertia
)


def test_mjcf_inspection():
    report = inspect_mjcf("examples/test_robot.xml")

    assert report["total_bodies"] == 3
    assert report["passed"] == 1
    assert report["failed"] == 1
    assert report["skipped"] == 1


def test_full_inertia_parsing():
    xml = ET.fromstring(
        '<inertial mass="2" pos="0 0 0" '
        'fullinertia="0.1 0.2 0.3 0.01 0.02 0.03"/>'
    )

    mass, tensor, center = parse_inertia(xml)

    expected = np.array([
        [0.1, 0.01, 0.02],
        [0.01, 0.2, 0.03],
        [0.02, 0.03, 0.3]
    ])

    assert mass == 2
    assert np.allclose(tensor, expected)
    assert center == [0, 0, 0]


def test_missing_inertia_components():
    xml = ET.fromstring('<inertial mass="1"/>')

    with pytest.raises(ValueError, match="Missing diaginertia"):
        parse_inertia(xml)


def test_rejects_nonfinite_values():
    xml = ET.fromstring(
        '<inertial mass="nan" diaginertia="0.1 0.2 0.3"/>'
    )

    with pytest.raises(ValueError, match="NaN or infinity"):
        parse_inertia(xml)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        inspect_mjcf("examples/does_not_exist.xml")
