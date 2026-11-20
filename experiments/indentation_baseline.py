
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


TIMESTEP = 0.0005
DURATION = 2.5
MAX_TRAVEL = 0.055


def create_model():
    xml = f"""
    <mujoco model="contactforge_indentation">
      <option timestep="{TIMESTEP}"
              gravity="0 0 -9.81"
              integrator="implicitfast"
              iterations="80"/>

      <worldbody>
        <!-- Stationary spherical indenter -->
        <geom name="indenter"
              type="sphere"
              pos="0 0 0.20"
              size="0.03"
              solref="0.01 1"/>

        <!-- Actuated sample stage -->
        <body name="stage" pos="0 0 0.12">
          <joint name="stage_z"
                 type="slide"
                 axis="0 0 1"
                 limited="true"
                 range="-0.005 0.065"/>

          <geom name="sample"
                type="box"
                size="0.08 0.08 0.01"
                mass="0.5"
                solref="0.01 1"/>
        </body>
      </worldbody>

      <actuator>
        <position name="stage_servo"
                  joint="stage_z"
                  kp="15000"
                  kv="200"
                  ctrllimited="true"
                  ctrlrange="0 {MAX_TRAVEL}"
                  forcelimited="true"
                  forcerange="-500 500"/>
      </actuator>
    </mujoco>
    """

    return mujoco.MjModel.from_xml_string(xml)


def commanded_position(time):
    # Hold, approach, hold contact, retract.
    if time < 0.25:
        return 0.0
    if time < 1.25:
        return MAX_TRAVEL * (time - 0.25)
    if time < 1.65:
        return MAX_TRAVEL
    if time < 2.25:
        return MAX_TRAVEL * (2.25 - time) / 0.60
    return 0.0


def run():
    model = create_model()
    data = mujoco.MjData(model)

    samples = []
    steps = round(DURATION / TIMESTEP)

    for _ in range(steps):
        target = commanded_position(data.time)
        data.ctrl[0] = target

        mujoco.mj_step(model, data)

        contact_force = 0.0
        max_overlap = 0.0

        for index in range(data.ncon):
            contact = data.contact[index]
            max_overlap = max(
                max_overlap,
                max(0.0, -float(contact.dist))
            )

            wrench = np.zeros(6)
            mujoco.mj_contactForce(
                model, data, index, wrench
            )
            contact_force += abs(float(wrench[0]))

        samples.append({
            "time_s": float(data.time),
            "target_m": float(target),
            "stage_displacement_m": float(data.qpos[0]),
            "stage_velocity_m_s": float(data.qvel[0]),
            "contact_force_n": contact_force,
            "contact_overlap_m": max_overlap
        })

    return samples


def main():
    samples = run()

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "indentation_baseline.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=samples[0].keys()
        )
        writer.writeheader()
        writer.writerows(samples)

    time = [s["time_s"] for s in samples]
    target = [s["target_m"] * 1000 for s in samples]
    actual = [
        s["stage_displacement_m"] * 1000
        for s in samples
    ]
    force = [s["contact_force_n"] for s in samples]

    fig, axes = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True
    )

    axes[0].plot(time, target, "--", label="Commanded")
    axes[0].plot(time, actual, label="Actual")
    axes[0].set_ylabel("Stage travel (mm)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(time, force)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Contact normal force (N)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_dir / "indentation_baseline.png",
        dpi=200
    )
    plt.close(fig)

    print("ContactForge — Indentation baseline")
    print(f"Samples: {len(samples)}")
    print(f"Peak contact force: {max(force):.3f} N")
    print(f"Maximum stage travel: {max(actual):.3f} mm")
    print(f"CSV: {csv_path}")
    print("Plot: results/indentation_baseline.png")


if __name__ == "__main__":
    main()
