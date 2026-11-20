
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from experiments.stack_scaling import create_model


BODY_COUNTS = [1, 4, 8, 16]
DURATION = 2.5
SETTLING_WINDOW = 0.5
SAMPLE_EVERY = 10

OUTPUT = Path("results/stack_stability.json")



def measure_stack(
    body_count,
    initial_lateral_speed=0.0,
    timestep=0.001,
):
    if not math.isfinite(initial_lateral_speed) or initial_lateral_speed < 0:
        raise ValueError(
            "Initial lateral speed must be nonnegative and finite."
        )

    if not math.isfinite(timestep) or timestep <= 0:
        raise ValueError(
            "Timestep must be positive and finite."
        )

    model = create_model(body_count, timestep=timestep)
    data = mujoco.MjData(model)

    if not math.isclose(
        model.opt.timestep,
        timestep,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"Requested timestep {timestep}, "
            f"but model uses {model.opt.timestep}"
        )

    # Apply the disturbance to the bottom box.
    bottom_body = model.body("box_0").id
    bottom_joint = int(model.body_jntadr[bottom_body])

    if bottom_joint < 0:
        raise ValueError("Bottom box does not have a joint.")

    velocity_address = int(model.jnt_dofadr[bottom_joint])

    data.qvel[velocity_address] = initial_lateral_speed
    mujoco.mj_forward(model, data)

    body_ids = [
        model.body(f"box_{index}").id
        for index in range(body_count)
    ]

    initial_positions = np.array([
        [0.0, 0.0, 0.15 + index * 0.102]
        for index in range(body_count)
    ])

    steps = round(DURATION / model.opt.timestep)
    settling_start = DURATION - SETTLING_WINDOW

    late_speeds = []
    late_contacts = []

    for step in range(steps):
        mujoco.mj_step(model, data)

        if data.time < settling_start:
            continue

        if step % SAMPLE_EVERY != 0:
            continue

        speeds = []

        for index in range(body_count):
            body_id = body_ids[index]
            joint_id = int(model.body_jntadr[body_id])

            if joint_id < 0:
                raise ValueError(f"Body box_{index} has no joint.")

            dof_start = int(model.jnt_dofadr[joint_id])
            linear_velocity = data.qvel[dof_start:dof_start + 3]
            speeds.append(float(np.linalg.norm(linear_velocity)))

        late_speeds.append(max(speeds))
        late_contacts.append(int(data.ncon))

    body_results = []

    for index, body_id in enumerate(body_ids):
        position = np.array(data.xpos[body_id], dtype=float)

        horizontal_drift = float(np.linalg.norm(
            position[:2] - initial_positions[index, :2]
        ))

        rotation = np.array(
            data.xmat[body_id], dtype=float
        ).reshape(3, 3)

        # Local +Z axis expressed in world coordinates.
        vertical_alignment = float(
            np.clip(rotation[2, 2], -1.0, 1.0)
        )

        tilt_degrees = math.degrees(
            math.acos(vertical_alignment)
        )

        body_results.append({
            "body": f"box_{index}",
            "final_position_m": position.tolist(),
            "horizontal_drift_mm": horizontal_drift * 1000,
            "tilt_degrees": tilt_degrees,
        })

    return {
        "body_count": body_count,
        "duration_s": float(data.time),
        "max_horizontal_drift_mm": max(
            item["horizontal_drift_mm"]
            for item in body_results
        ),
        "max_tilt_degrees": max(
            item["tilt_degrees"]
            for item in body_results
        ),
        "mean_late_max_linear_speed_m_s": float(
            np.mean(late_speeds)
        ),
        "peak_late_max_linear_speed_m_s": max(late_speeds),
        "mean_late_contacts": float(np.mean(late_contacts)),
        "final_contacts": int(data.ncon),
        "bodies": body_results,
        "actual_timestep_s": float(model.opt.timestep),
        "simulation_steps": steps,
    }


def main():
    results = []

    print("\nContactForge — Stack stability diagnostics")
    print("-" * 75)

    for body_count in BODY_COUNTS:
        result = measure_stack(body_count)
        results.append(result)

        print(
            f"Boxes={body_count:2d} | "
            f"drift={result['max_horizontal_drift_mm']:.3f} mm | "
            f"tilt={result['max_tilt_degrees']:.3f} deg | "
            f"late speed={result['mean_late_max_linear_speed_m_s']:.6f} m/s"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8"
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
