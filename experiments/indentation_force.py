
import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from contactforge.simulation.runner import (
    prepare_control_events,
    validate_settings,
)


def measure_contact_force(
    model,
    data,
    geom_a_name,
    geom_b_name,
):
    geom_a = model.geom(geom_a_name).id
    geom_b = model.geom(geom_b_name).id

    total_normal_force = 0.0
    matching_contacts = 0

    for index in range(data.ncon):
        contact = data.contact[index]

        pair = {
            int(contact.geom1),
            int(contact.geom2),
        }

        if pair != {geom_a, geom_b}:
            continue

        # MuJoCo returns six contact-frame force/torque components.
        force = np.zeros(6, dtype=float)

        mujoco.mj_contactForce(
            model,
            data,
            index,
            force,
        )

        # First component: normal force in the contact frame.
        total_normal_force += max(0.0, float(force[0]))
        matching_contacts += 1

    return total_normal_force, matching_contacts


def run_indentation(
    scene_path,
    controls_path,
    timestep=0.001,
    duration=1.0,
    sample_interval=0.01,
):
    steps, stride = validate_settings(
        timestep,
        duration,
        sample_interval,
    )

    model = mujoco.MjModel.from_xml_path(
        str(Path(scene_path).resolve())
    )
    model.opt.timestep = timestep

    data = mujoco.MjData(model)

    with open(controls_path, encoding="utf-8") as file:
        control_schedule = json.load(file)

    scheduled_events, _ = prepare_control_events(
        model,
        control_schedule,
        timestep,
        duration,
    )

    stage_joint_id = model.joint("stage_slide").id
    sample_joint_id = model.joint("sample_compression").id
    motor_id = model.actuator("stage_motor").id

    stage_qpos_index = int(
        model.jnt_qposadr[stage_joint_id]
    )

    sample_qpos_index = int(
        model.jnt_qposadr[sample_joint_id]
    )

    sample_qvel_index = int(
        model.jnt_dofadr[sample_joint_id]
    )

    stiffness = float(
        model.jnt_stiffness[sample_joint_id]
    )

    damping = float(
        model.dof_damping[sample_qvel_index]
    )

    spring_reference = float(
        model.qpos_spring[sample_qpos_index]
    )

    # Apply initial commands before stepping.
    for actuator_id, value in scheduled_events.get(0, []):
        data.ctrl[actuator_id] = value

    mujoco.mj_forward(model, data)

    measurements = []

    def record():
        stage_displacement = float(
            data.qpos[stage_qpos_index]
        )

        sample_joint_position = float(
            data.qpos[sample_qpos_index]
        )

        sample_joint_velocity = float(
            data.qvel[sample_qvel_index]
        )

        compression = max(
            0.0,
            spring_reference - sample_joint_position,
        )

        spring_force = stiffness * compression

        contact_force, matching_contacts = (
            measure_contact_force(
                model,
                data,
                "indenter_tip",
                "sample_collision",
            )
        )

        measurements.append({
            "time_s": float(data.time),
            "motor_target_m": float(
                data.ctrl[motor_id]
            ),
            "stage_displacement_mm": (
                stage_displacement * 1000
            ),
            "compression_mm": (
                compression * 1000
            ),
            "sample_joint_velocity_m_s": (
                sample_joint_velocity
            ),
            "spring_force_N": spring_force,
            "contact_normal_force_N": contact_force,
            "contact_points": matching_contacts,
        })

    record()

    for step in range(1, steps + 1):
        mujoco.mj_step(model, data)

        # Update controls for the next integration interval.
        if step < steps:
            for actuator_id, value in (
                scheduled_events.get(step, [])
            ):
                data.ctrl[actuator_id] = value

        if step % stride == 0:
            record()

    return {
        "scene": str(Path(scene_path).resolve()),
        "timestep_s": timestep,
        "duration_s": duration,
        "sample_interval_s": sample_interval,
        "stiffness_N_per_m": stiffness,
        "damping_Ns_per_m": damping,
        "measurements": measurements,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure force during controlled indentation."
    )

    parser.add_argument("scene")
    parser.add_argument("--controls", required=True)

    parser.add_argument(
        "--timestep",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/indentation_force.json"),
    )

    args = parser.parse_args()

    report = run_indentation(
        args.scene,
        args.controls,
        timestep=args.timestep,
        duration=args.duration,
        sample_interval=args.sample_interval,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = args.output.with_suffix(".csv")

    measurements = report["measurements"]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(measurements[0]),
        )

        writer.writeheader()
        writer.writerows(measurements)

    contact_measurements = [
        row
        for row in measurements
        if row["contact_points"] > 0
    ]

    print("\nContactForge — Indentation force analysis")
    print("-" * 55)

    print(
        f"Spring stiffness: "
        f"{report['stiffness_N_per_m']:.1f} N/m"
    )

    print(f"Samples recorded: {len(measurements)}")
    print(f"Samples with contact: {len(contact_measurements)}")

    final = measurements[-1]

    print(
        f"Final stage displacement: "
        f"{final['stage_displacement_mm']:.4f} mm"
    )

    print(
        f"Final compression: "
        f"{final['compression_mm']:.4f} mm"
    )

    print(
        f"Final spring force: "
        f"{final['spring_force_N']:.4f} N"
    )

    print(
        f"Final contact force: "
        f"{final['contact_normal_force_N']:.4f} N"
    )

    print(f"\nJSON: {args.output}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
