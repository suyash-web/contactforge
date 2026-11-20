
import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import mujoco
import numpy as np


MASS = 0.2
RADIUS = 0.05
INITIAL_HEIGHT = 0.30
DURATION = 1.0

# TIMESTEPS = [0.002, 0.001, 0.0005]
TIMESTEPS = [0.002, 0.001, 0.0005, 0.00025, 0.000125]

# MuJoCo's positive solref values specify
# contact time constant and damping ratio.
CONTACT_SETTINGS = {
    "soft": (0.04, 1.0),
    "default_like": (0.02, 1.0),
    "stiff": (0.01, 1.0),
}


def create_model(timestep, time_constant, damping_ratio):
    xml = f"""
    <mujoco model="contact_sensitivity">
      <option timestep="{timestep}"
              gravity="0 0 -9.81"
              integrator="Euler"
              iterations="50"/>

      <worldbody>
        <geom name="ground"
              type="plane"
              size="1 1 0.1"
              solref="{time_constant} {damping_ratio}"/>

        <body name="sphere" pos="0 0 {INITIAL_HEIGHT}">
          <freejoint/>

          <geom name="sphere_geom"
                type="sphere"
                size="{RADIUS}"
                mass="{MASS}"
                solref="{time_constant} {damping_ratio}"/>
        </body>
      </worldbody>
    </mujoco>
    """

    return mujoco.MjModel.from_xml_string(xml)


def simulate(timestep, time_constant, damping_ratio):
    model = create_model(
        timestep,
        time_constant,
        damping_ratio
    )
    data = mujoco.MjData(model)

    peak_force = 0.0
    max_penetration = 0.0
    first_contact = None

    force_history = []
    velocity_history = []

    steps = round(DURATION / timestep)

    start = perf_counter()

    for _ in range(steps):
        mujoco.mj_step(model, data)

        total_normal_force = 0.0

        for index in range(data.ncon):
            contact = data.contact[index]

            if first_contact is None:
                first_contact = float(data.time)

            depth = max(0.0, -float(contact.dist))
            max_penetration = max(max_penetration, depth)

            wrench = np.zeros(6)
            mujoco.mj_contactForce(
                model, data, index, wrench
            )

            total_normal_force += max(
                0.0, float(wrench[0])
            )

        peak_force = max(peak_force, total_normal_force)

        force_history.append(total_normal_force)
        velocity_history.append(float(data.qvel[2]))

    elapsed = perf_counter() - start

    # Average over the final 10% of the simulation.
    tail_length = max(1, len(force_history) // 10)

    mean_rest_force = float(
        np.mean(force_history[-tail_length:])
    )

    max_rest_speed = float(
        np.max(np.abs(velocity_history[-tail_length:]))
    )

    return {
        "timestep_s": timestep,
        "contact_time_constant_s": time_constant,
        "damping_ratio": damping_ratio,
        "first_contact_s": first_contact,
        "peak_force_n": peak_force,
        "max_penetration_mm": max_penetration * 1000,
        "rest_force_n": mean_rest_force,
        "max_rest_speed_m_s": max_rest_speed,
        "final_height_m": float(data.qpos[2]),
        "wall_time_ms": elapsed * 1000,
        "sim_steps_per_second": steps / elapsed,
    }


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    records = []

    for setting, parameters in CONTACT_SETTINGS.items():
        time_constant, damping_ratio = parameters

        for timestep in TIMESTEPS:
            result = simulate(
                timestep,
                time_constant,
                damping_ratio
            )

            result["setting"] = setting
            records.append(result)

            print(
                f"{setting:13} "
                f"dt={timestep:.4f}  "
                f"penetration={result['max_penetration_mm']:.3f} mm  "
                f"peak={result['peak_force_n']:.2f} N  "
                f"runtime={result['wall_time_ms']:.2f} ms"
            )

    csv_path = output_dir / "contact_sweep.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    fig, ax = plt.subplots(figsize=(8, 5))

    for setting in CONTACT_SETTINGS:
        subset = [
            record for record in records
            if record["setting"] == setting
        ]

        ax.plot(
            [r["timestep_s"] * 1000 for r in subset],
            [r["max_penetration_mm"] for r in subset],
            marker="o",
            label=setting
        )

    ax.set_xlabel("Integration timestep (ms)")
    ax.set_ylabel("Maximum contact penetration (mm)")
    ax.set_title("Contact penetration sensitivity")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_dir / "contact_penetration_sweep.png",
        dpi=200
    )
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print("Saved: results/contact_penetration_sweep.png")


if __name__ == "__main__":
    main()
