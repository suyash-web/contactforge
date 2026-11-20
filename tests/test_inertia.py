
import numpy as np
import pytest

from contactforge.assets.inertia import (
    box_inertia,
    validate_inertia
)


def test_valid_box():
    I = box_inertia(2.0, 0.2, 0.3, 0.4)
    result = validate_inertia(2.0, I)

    assert result.valid
    assert not result.errors


def test_negative_mass():
    I = np.diag([0.1, 0.2, 0.25])
    result = validate_inertia(-1.0, I)

    assert not result.valid
    assert any("Mass" in e for e in result.errors)


def test_negative_inertia():
    I = np.diag([-0.1, 0.2, 0.3])
    result = validate_inertia(1.0, I)

    assert not result.valid
    assert any("positive definite" in e for e in result.errors)


def test_triangle_inequality():
    I = np.diag([0.1, 0.1, 1.0])
    result = validate_inertia(1.0, I)

    assert not result.valid
    assert any("triangle inequality" in e for e in result.errors)


def test_asymmetric_tensor():
    I = np.array([
        [0.2, 0.1, 0],
        [0, 0.2, 0],
        [0, 0, 0.2]
    ])
    result = validate_inertia(1.0, I)

    assert not result.valid
    assert any("symmetric" in e for e in result.errors)


@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_invalid_box_mass(value):
    with pytest.raises(ValueError):
        box_inertia(value, 0.2, 0.3, 0.4)
