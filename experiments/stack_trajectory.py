
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from experiments.stack_scaling import create_model


BODY_COUNTS = [8, 16]
TIMESTEPS = [0.002, 0.001, 0.0005, 0.00025, 0.000125]

DURATION = 2.5
SAMPLE_INTERVAL = 0.01
INITIAL_SPEED = 0.05

OUTPUT_DIR = Path("results")

def active_contact_pairs(data):
    pairs = set()

    for index in range(data.ncon):
        contact = data.contact[index]

        geom_a = int(contact.geom1)
        geom_b = int(contact.geom2)

        pairs.add(tuple(sorted((geom_a, geom_b))))

    return frozenset(pairs)

def simulate_trajectory(body_count, timestep):
    model = create_model(body_count, timestep=timestep)
    data = mujoco.MjData(model)

    actual_dt = float(model.opt.timestep)

    if not np.isclose(actual_dt, timestep, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Model timestep differs from requested value.")

    total_steps = round(DURATION / actual_dt)
    sample_stride = round(SAMPLE_INTERVAL / actual_dt)

    if not np.isclose(sample_stride * actual_dt, SAMPLE_INTERVAL):
        raise ValueError("Sampling interval is not aligned with timestep.")

    if total_steps % sample_stride != 0:
        raise ValueError("Final time is not aligned with sampling interval.")

    body_ids = np.array([
        model.body(f"box_{index}").id
        for index in range(body_count)
    ])

    bottom_body_id = int(body_ids[0])
    joint_id = int(model.body_jntadr[bottom_body_id])

    if joint_id < 0:
        raise ValueError("Bottom box is missing its free joint.")

    dof_start = int(model.jnt_dofadr[joint_id])
    data.qvel[dof_start] = INITIAL_SPEED

    mujoco.mj_forward(model, data)

    # times = [0.0]
    # positions = [data.xpos[body_ids].copy()]
    # contacts = [int(data.ncon)]

    times = [0.0]
    positions = [data.xpos[body_ids].copy()]
    contacts = [int(data.ncon)]
    contact_pairs = [active_contact_pairs(data)]

    for step in range(1, total_steps + 1):
        mujoco.mj_step(model, data)

        if step % sample_stride == 0:
            times.append(step * actual_dt)
            positions.append(data.xpos[body_ids].copy())
            contacts.append(int(data.ncon))
            contact_pairs.append(active_contact_pairs(data)) # modified

    return {
        "timestep_s": actual_dt,
        "steps": total_steps,
        "times": np.array(times),
        "positions": np.stack(positions),
        "contacts": np.array(contacts),
        "contact_pairs": contact_pairs,
    }

def compare_trajectories(candidate, reference):
    if candidate["positions"].shape != reference["positions"].shape:
        raise ValueError("Trajectory dimensions do not match.")

    if not np.allclose(
        candidate["times"],
        reference["times"],
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("Trajectory timestamps do not match.")

    # Shape: (time samples, bodies, XYZ coordinates).
    position_difference = (
        candidate["positions"] - reference["positions"]
    )

    # RMS Euclidean position difference across all bodies at each time.
    rms_error_mm = np.sqrt(
        np.mean(
            np.sum(position_difference ** 2, axis=2),
            axis=1,
        )
    ) * 1000

    contact_difference = np.abs(
        candidate["contacts"] - reference["contacts"]
    )

    return {
        "times": candidate["times"],
        "rms_error_mm": rms_error_mm,
        "contact_difference": contact_difference,
        "mean_error_mm": float(np.mean(rms_error_mm)),
        "max_error_mm": float(np.max(rms_error_mm)),
        "final_error_mm": float(rms_error_mm[-1]),
        "time_of_max_error_s": float(
            candidate["times"][np.argmax(rms_error_mm)]
        ),
        "contact_mismatch_fraction": float(
            np.mean(contact_difference != 0)
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    reference_dt = min(TIMESTEPS)

    fig, axes = plt.subplots(
        len(BODY_COUNTS),
        1,
        figsize=(10, 7),
        sharex=True,
    )

    for axis, body_count in zip(axes, BODY_COUNTS):
        reference = simulate_trajectory(body_count, reference_dt)

        print(f"\n{body_count}-box stack")
        print("-" * 76)

        for timestep in TIMESTEPS:
            if timestep == reference_dt:
                continue

            candidate = simulate_trajectory(body_count, timestep)
            comparison = compare_trajectories(candidate, reference)

            record = {
                "bodies": body_count,
                "timestep_ms": timestep * 1000,
                "reference_timestep_ms": reference_dt * 1000,
                "mean_trajectory_error_mm":
                    comparison["mean_error_mm"],
                "maximum_trajectory_error_mm":
                    comparison["max_error_mm"],
                "final_position_error_mm":
                    comparison["final_error_mm"],
                "time_of_max_error_s":
                    comparison["time_of_max_error_s"],
                "contact_mismatch_fraction":
                    comparison["contact_mismatch_fraction"],
            }

            records.append(record)

            axis.plot(
                comparison["times"],
                comparison["rms_error_mm"],
                label=f"{timestep * 1000:g} ms",
            )

            print(
                f"dt={timestep * 1000:5.3f} ms | "
                f"mean={comparison['mean_error_mm']:8.4f} mm | "
                f"max={comparison['max_error_mm']:8.4f} mm | "
                f"peak time={comparison['time_of_max_error_s']:.2f} s | "
                f"contact mismatch="
                f"{comparison['contact_mismatch_fraction'] * 100:.1f}%"
            )

        axis.set_title(f"{body_count}-box stack")
        axis.set_ylabel("RMS position difference (mm)")
        axis.grid(alpha=0.3)
        axis.legend(title="Timestep")

    axes[-1].set_xlabel("Simulation time (s)")

    fig.suptitle(
        "ContactForge: Contact-Rich Trajectory Sensitivity",
        fontsize=14,
    )
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "stack_trajectory.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    csv_path = OUTPUT_DIR / "stack_trajectory.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    json_path = OUTPUT_DIR / "stack_trajectory.json"
    json_path.write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nPlot: {plot_path}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
