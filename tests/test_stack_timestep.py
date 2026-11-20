
import math

import pytest

from experiments.stack_stability import measure_stack


@pytest.mark.parametrize(
    "timestep",
    [0.002, 0.001, 0.0005, 0.00025],
)
def test_stack_uses_requested_timestep(timestep):
    result = measure_stack(
        body_count=1,
        initial_lateral_speed=0.0,
        timestep=timestep,
    )

    actual = result["actual_timestep_s"]
    expected_steps = round(2.5 / timestep)

    if not math.isclose(
        actual,
        timestep,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        pytest.fail(
            f"Requested {timestep}, got {actual}"
        )

    if result["simulation_steps"] != expected_steps:
        pytest.fail(
            f"Expected {expected_steps} steps, "
            f"got {result['simulation_steps']}"
        )
