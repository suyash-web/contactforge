
from dataclasses import dataclass
import numpy as np


@dataclass
class InertiaValidation:
    valid: bool
    errors: list[str]
    principal_moments: list[float]


def box_inertia(mass, width, depth, height):
    """Inertia of a uniform solid box about its center of mass."""
    dimensions = np.array([width, depth, height], dtype=float)

    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Mass must be positive and finite.")

    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
        raise ValueError("Box dimensions must be positive and finite.")

    ixx = mass * (depth**2 + height**2) / 12
    iyy = mass * (width**2 + height**2) / 12
    izz = mass * (width**2 + depth**2) / 12

    return np.diag([ixx, iyy, izz])


def validate_inertia(mass, tensor, tolerance=1e-10):
    """Validate a rigid body's mass and COM inertia tensor."""
    errors = []
    I = np.asarray(tensor, dtype=float)

    if not np.isfinite(mass) or mass <= 0:
        errors.append("Mass must be positive and finite.")

    if I.shape != (3, 3):
        return InertiaValidation(
            False, errors + ["Inertia tensor must be 3x3."], []
        )

    if not np.all(np.isfinite(I)):
        return InertiaValidation(
            False, errors + ["Inertia contains NaN or infinity."], []
        )

    if not np.allclose(I, I.T, atol=tolerance, rtol=0):
        errors.append("Inertia tensor is not symmetric.")
        return InertiaValidation(False, errors, [])

    moments = np.linalg.eigvalsh(I)
    scale = max(float(np.max(np.abs(moments))), 1e-12)
    eps = tolerance * scale

    if np.any(moments <= eps):
        errors.append("Inertia is not positive definite.")

    if (
        moments[0] + moments[1]
        < moments[2] - eps
    ):
        errors.append("Principal moments violate triangle inequality.")

    return InertiaValidation(
        valid=len(errors) == 0,
        errors=errors,
        principal_moments=moments.tolist()
    )
