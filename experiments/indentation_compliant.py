
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from experiments.indentation_baseline import commanded_position


TIMESTEP = 0.0005
DURATION = 2.5
MAX_TRAVEL = 0.055

SAMPLE_STIFFNESS = 12000.0
SAMPLE_DAMPING = 30.0


def create_model(
        stiffness=SAMPLE_STIFFNESS,
        damping=SAMPLE_DAMPING,
        timestep=TIMESTEP,
        contact_time_constant=0.005,
    ):
    xml = f"""
    <mujoco model="compliant_indentation">
      <option timestep="{timestep}"
              gravity="0 0 -9.81"
              integrator="implicitfast"
              iterations="80"/>

      <worldbody>
        <geom name="indenter"
              type="sphere"
              pos="0 0 0.20"
              size="0.03"
              solref="{contact_time_constant} 1"/>

        <body name="stage" pos="0 0 0.12">
          <joint name="stage_z"
                 type="slide"
                 axis="0 0 1"
                 limited="true"
                 range="-0.005 0.065"/>

          <inertial pos="0 0 0"
                    mass="0.5"
                    diaginertia="0.001 0.001 0.001"/>

          <body name="sample_surface">
            <joint name="compression_z"
                   type="slide"
                   axis="0 0 1"
                   limited="true"
                   range="-0.025 0.003"
                   stiffness="{stiffness}"
                   damping="{damping}"
                   springref="0"/>

            <geom name="sample"
                  type="box"
                  size="0.08 0.08 0.01"
                  mass="0.05"
                  solref="0.005 1"/>
          </body>
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


def run(
        stiffness=SAMPLE_STIFFNESS,
        damping=SAMPLE_DAMPING,
        timestep=TIMESTEP,
        contact_time_constant=0.005,
    ):
    if stiffness <= 0 or damping < 0:
        raise ValueError("Invalid stiffness or damping.")

    if timestep <= 0 or contact_time_constant <= 0:
        raise ValueError("Timestep and contact time constant must be positive.")

    model = create_model(
        stiffness=stiffness,
        damping=damping,
        timestep=timestep,
        contact_time_constant=contact_time_constant,
    )
    data = mujoco.MjData(model)

    indenter_id = model.geom("indenter").id
    sample_id = model.geom("sample").id

    stage_joint_id = model.joint("stage_z").id
    compression_joint_id = model.joint("compression_z").id

    stage_qpos = model.jnt_qposadr[stage_joint_id]
    compression_qpos = model.jnt_qposadr[compression_joint_id]
    compression_dof = model.jnt_dofadr[compression_joint_id]

    samples = []
    steps = round(DURATION / timestep)

    for _ in range(steps):
        target = commanded_position(data.time)
        data.ctrl[0] = target

        mujoco.mj_step(model, data)

        contact_force = 0.0
        max_overlap = 0.0

        for index in range(data.ncon):
            contact = data.contact[index]

            pair = {int(contact.geom1), int(contact.geom2)}
            if pair != {indenter_id, sample_id}:
                continue

            max_overlap = max(
                max_overlap,
                max(0.0, -float(contact.dist))
            )

            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            contact_force += abs(float(wrench[0]))

        stage_position = float(data.qpos[stage_qpos])
        relative_position = float(data.qpos[compression_qpos])
        relative_velocity = float(data.qvel[compression_dof])

        compression = max(0.0, -relative_position)

        # Positive force corresponds to compression resistance.
        spring_damper_force = (
            -stiffness * relative_position
            -damping * relative_velocity
        )

        samples.append({
            "time_s": float(data.time),
            "target_m": float(target),
            "stage_displacement_m": stage_position,
            "sample_compression_m": compression,
            "contact_overlap_m": max_overlap,
            "contact_force_n": contact_force,
            "spring_damper_force_n": spring_damper_force,
            "sample_surface_displacement_m": (
                stage_position + relative_position
            ),
            "compression_velocity_m_s": -relative_velocity
        })

    return samples


def main():
    samples = run()

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "indentation_compliant.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(samples[0].keys())
        )
        writer.writeheader()
        writer.writerows(samples)

    time = np.array([s["time_s"] for s in samples])
    target = np.array([s["target_m"] for s in samples])
    stage = np.array([
        s["stage_displacement_m"] for s in samples
    ])
    compression = np.array([
        s["sample_compression_m"] for s in samples
    ])
    overlap = np.array([
        s["contact_overlap_m"] for s in samples
    ])
    force = np.array([
        s["contact_force_n"] for s in samples
    ])

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))

    axes[0].plot(time, target * 1000, "--", label="Commanded")
    axes[0].plot(time, stage * 1000, label="Actual")
    axes[0].set_ylabel("Stage travel (mm)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        time, compression * 1000, label="Sample compression"
    )
    axes[1].plot(
        time, overlap * 1000, label="Contact overlap"
    )
    axes[1].set_ylabel("Displacement (mm)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(time, force)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Contact force (N)")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()

    plot_path = output_dir / "indentation_compliant.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print("\nContactForge — Compliant indentation")
    print("-" * 48)
    print(f"Peak force: {np.max(force):.3f} N")
    print(f"Maximum stage travel: {np.max(stage)*1000:.3f} mm")
    print(
        f"Maximum sample compression: "
        f"{np.max(compression)*1000:.3f} mm"
    )
    print(
        f"Maximum numerical contact overlap: "
        f"{np.max(overlap)*1000:.3f} mm"
    )
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
