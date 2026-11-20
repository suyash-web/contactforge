
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import mujoco
import numpy as np


MASS = 0.2
RADIUS = 0.05
INITIAL_HEIGHT = 0.30
GRAVITY = 9.81
TIMESTEP = 0.001
DURATION = 1.5


def create_model():
    xml = f"""
    <mujoco model="contactforge_impact">
      <option timestep="{TIMESTEP}"
              gravity="0 0 -{GRAVITY}"
              integrator="Euler"/>

      <worldbody>
        <geom name="ground"
              type="plane"
              size="1 1 0.1"
              friction="0.8 0.005 0.0001"/>

        <body name="sphere" pos="0 0 {INITIAL_HEIGHT}">
          <freejoint/>
          <geom name="sphere_geom"
                type="sphere"
                size="{RADIUS}"
                mass="{MASS}"
                friction="0.8 0.005 0.0001"/>
        </body>
      </worldbody>
    </mujoco>
    """

    return mujoco.MjModel.from_xml_string(xml)


def run_experiment():
    model = create_model()
    data = mujoco.MjData(model)

    samples = []
    first_contact_time = None
    peak_force = 0.0
    max_penetration = 0.0

    steps = round(DURATION / TIMESTEP)

    for _ in range(steps):
        mujoco.mj_step(model, data)

        height = float(data.qpos[2])
        velocity = float(data.qvel[2])

        normal_force = 0.0
        penetration = 0.0

        for index in range(data.ncon):
            contact = data.contact[index]

            penetration = max(
                penetration,
                max(0.0, -float(contact.dist))
            )

            wrench = np.zeros(6)
            mujoco.mj_contactForce(
                model, data, index, wrench
            )

            normal_force += max(0.0, float(wrench[0]))

        if data.ncon > 0 and first_contact_time is None:
            first_contact_time = float(data.time)

        peak_force = max(peak_force, normal_force)
        max_penetration = max(
            max_penetration, penetration
        )

        samples.append({
            "time_s": float(data.time),
            "height_m": height,
            "velocity_m_s": velocity,
            "normal_force_n": normal_force,
            "penetration_m": penetration
        })

    if first_contact_time is None:
        raise RuntimeError(
            "No contact occurred during the simulation."
        )

    return samples, {
        "first_contact_s": first_contact_time,
        "peak_normal_force_n": peak_force,
        "max_penetration_m": max_penetration,
        "final_height_m": samples[-1]["height_m"],
        "final_velocity_m_s": samples[-1]["velocity_m_s"]
    }


def main():
    samples, metrics = run_experiment()

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "impact_baseline.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(samples[0].keys())
        )
        writer.writeheader()
        writer.writerows(samples)

    times = [row["time_s"] for row in samples]
    heights = [row["height_m"] for row in samples]
    forces = [row["normal_force_n"] for row in samples]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7),
                             sharex=True)

    axes[0].plot(times, heights)
    axes[0].axhline(
        RADIUS,
        linestyle="--",
        label="Nominal ground contact height"
    )
    axes[0].set_ylabel("Sphere center height (m)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, forces)
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Normal contact force (N)")
    axes[1].grid(alpha=0.3)

    fig.suptitle("ContactForge: Rigid Sphere Impact")
    fig.tight_layout()

    plot_path = output_dir / "impact_baseline.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print("\nContactForge — Impact baseline")
    print("-" * 45)

    for name, value in metrics.items():
        print(f"{name}: {value:.8f}")

    print(f"\nCSV saved: {csv_path}")
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
