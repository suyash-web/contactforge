
import mujoco
import numpy as np
import pytest


GRAVITY = 9.81
INITIAL_HEIGHT = 1.0
DURATION = 0.2

TIMESTEPS = [0.01, 0.005, 0.002, 0.001, 0.0005]


def run_freefall(timestep):
    xml = f"""
    <mujoco model="freefall_test">
      <option gravity="0 0 -{GRAVITY}"
              timestep="{timestep}"
              integrator="Euler"/>
      <worldbody>
        <body pos="0 0 {INITIAL_HEIGHT}">
          <freejoint/>
          <geom type="box"
                size="0.05 0.05 0.05"
                mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    steps = round(DURATION / timestep)

    for _ in range(steps):
        mujoco.mj_step(model, data)

    actual = float(data.qpos[2])
    expected = INITIAL_HEIGHT - 0.5 * GRAVITY * data.time**2

    return abs(actual - expected), data.time


@pytest.mark.parametrize("timestep", TIMESTEPS)
def test_freefall_error_matches_theory(timestep):
    error, actual_time = run_freefall(timestep)

    predicted_error = (
        0.5 * GRAVITY * actual_time * timestep
    )

    assert np.isclose(
        error,
        predicted_error,
        rtol=1e-7,
        atol=1e-10
    ), (
        f"Unexpected integration error at dt={timestep}: "
        f"measured={error}, predicted={predicted_error}"
    )


def test_first_order_convergence():
    errors = []

    for timestep in TIMESTEPS:
        error, _ = run_freefall(timestep)
        errors.append(error)

    slope = np.polyfit(
        np.log(TIMESTEPS),
        np.log(errors),
        1
    )[0]

    assert abs(slope - 1.0) < 0.05, (
        f"Expected first-order convergence, got {slope}"
    )
