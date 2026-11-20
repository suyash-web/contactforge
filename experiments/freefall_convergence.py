
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


GRAVITY = 9.81
INITIAL_HEIGHT = 1.0
DURATION = 0.2

TIMESTEPS = [0.01, 0.005, 0.002, 0.001, 0.0005]


def simulate_freefall(timestep):
    xml = f"""
    <mujoco model="freefall_validation">
      <option
        gravity="0 0 -{GRAVITY}"
        timestep="{timestep}"
        integrator="Euler"
      />
      <worldbody>
        <body name="falling_box" pos="0 0 {INITIAL_HEIGHT}">
          <freejoint/>
          <geom type="box" size="0.05 0.05 0.05" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    steps = round(DURATION / timestep)

    for _ in range(steps):
        mujoco.mj_step(model, data)

    simulated_height = float(data.qpos[2])
    analytical_height = INITIAL_HEIGHT - 0.5 * GRAVITY * data.time**2

    error = abs(simulated_height - analytical_height)

    return error


def main():
    errors = []

    print(f"{'Timestep (s)':<16} {'Error (m)':<16}")

    for timestep in TIMESTEPS:
        error = simulate_freefall(timestep)
        errors.append(error)

        print(f"{timestep:<16.5f} {error:<16.8f}")

    errors = np.array(errors)
    timesteps = np.array(TIMESTEPS)

    # Estimate how error changes with timestep
    slope = np.polyfit(
        np.log(timesteps),
        np.log(errors),
        1
    )[0]

    print(f"\nObserved convergence order: {slope:.4f}")

    assert abs(slope - 1.0) < 0.05, (
        f"Unexpected convergence order: {slope}"
    )

    # Plot the measured numerical errors
    plt.figure(figsize=(7, 5))
    plt.loglog(
        timesteps,
        errors,
        "o-",
        label="MuJoCo position error"
    )

    plt.loglog(
        timesteps,
        0.5 * GRAVITY * DURATION * timesteps,
        "--",
        label="Predicted Euler error"
    )

    plt.xlabel("Simulation timestep (s)")
    plt.ylabel("Absolute position error (m)")
    plt.title("MuJoCo Free-Fall Numerical Convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    output = Path("results/freefall_convergence.png")
    plt.savefig(output, dpi=200, bbox_inches="tight")

    print(f"Plot saved to: {output}")
    plt.show()


if __name__ == "__main__":
    main()
