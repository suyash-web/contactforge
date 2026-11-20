import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from contactforge.simulation.runner import prepare_control_events, validate_settings
from experiments.indentation_force import measure_contact_force


def simulate(scene_path, control_schedule, timestep, duration=1.0, sample_interval=0.01):
    steps, stride = validate_settings(timestep, duration, sample_interval)
    model = mujoco.MjModel.from_xml_path(str(Path(scene_path).resolve()))
    model.opt.timestep = timestep
    data = mujoco.MjData(model)

    events, _ = prepare_control_events(model, control_schedule, timestep, duration)
    stage_joint = model.joint("stage_slide").id
    sample_joint = model.joint("sample_compression").id
    motor_id = model.actuator("stage_motor").id
    sample_body_id = model.body("sample").id

    stage_qpos = int(model.jnt_qposadr[stage_joint])
    sample_qpos = int(model.jnt_qposadr[sample_joint])
    stage_dof = int(model.jnt_dofadr[stage_joint])
    sample_dof = int(model.jnt_dofadr[sample_joint])

    mass = float(model.body_mass[sample_body_id])
    stiffness = float(model.jnt_stiffness[sample_joint])
    damping = float(model.dof_damping[sample_dof])
    springref = float(model.qpos_spring[sample_qpos])

    # This force balance assumes the two slide joints both point upward.
    if not np.allclose(model.jnt_axis[stage_joint], [0, 0, 1], atol=1e-10):
        raise ValueError("Stage slide joint must point along +Z.")
    if not np.allclose(model.jnt_axis[sample_joint], [0, 0, 1], atol=1e-10):
        raise ValueError("Sample slide joint must point along +Z.")
    if not np.allclose(model.opt.gravity, [0, 0, 0], atol=1e-12):
        raise ValueError("This baseline balance requires zero gravity.")
    if mass <= 0:
        raise ValueError("Sample mass must be positive.")

    for actuator_id, value in events.get(0, []):
        data.ctrl[actuator_id] = value

    rows = []

    def record():
        # Recompute forces and accelerations for this exact sampled state.
        # This matters: mj_step can leave acceleration/contact forces
        # associated with the beginning rather than end of a step.
        mujoco.mj_forward(model, data)
        q = float(data.qpos[sample_qpos])
        v = float(data.qvel[sample_dof])
        a_stage = float(data.qacc[stage_dof])
        a_relative = float(data.qacc[sample_dof])
        compression = springref - q
        spring_up = stiffness * compression
        damping_up = -damping * v
        contact_down, contact_points = measure_contact_force(
            model, data, "indenter_tip", "sample_collision"
        )
        inertia_up = mass * (a_stage + a_relative)
        # Newton's second law, +Z positive:
        # inertia_up = spring_up + damping_up - contact_down
        residual = inertia_up - (spring_up + damping_up - contact_down)
        rows.append({
            "time_s": float(data.time),
            "motor_target_m": float(data.ctrl[motor_id]),
            "stage_displacement_mm": float(data.qpos[stage_qpos]) * 1000,
            "compression_mm": compression * 1000,
            "sample_relative_velocity_m_s": v,
            "stage_acceleration_m_s2": a_stage,
            "sample_relative_acceleration_m_s2": a_relative,
            "spring_up_N": spring_up,
            "damping_up_N": damping_up,
            "inertia_up_N": inertia_up,
            "contact_down_N": contact_down,
            "force_balance_residual_N": residual,
            "contact_points": contact_points,
        })

    record()
    for step in range(1, steps + 1):
        mujoco.mj_step(model, data)
        if step < steps:
            for actuator_id, value in events.get(step, []):
                data.ctrl[actuator_id] = value
        if step % stride == 0:
            record()

    return {
        "timestep_ms": timestep * 1000,
        "sample_mass_kg": mass,
        "known_stiffness_N_per_m": stiffness,
        "damping_Ns_per_m": damping,
        "measurements": rows,
    }


def summarize_run(run):
    rows = run["measurements"]
    contact_rows = [r for r in rows if r["contact_points"] > 0]
    residuals = [abs(r["force_balance_residual_N"]) for r in rows]

    # Quasi-static sampling: require actual contact, enough compression,
    # and small relative speed and absolute specimen acceleration.
    quiet = [
        r for r in contact_rows
        if r["compression_mm"] >= 1.0
        and abs(r["sample_relative_velocity_m_s"]) <= 0.005
        and abs(r["stage_acceleration_m_s2"] + r["sample_relative_acceleration_m_s2"]) <= 0.5
    ]
    estimate = None
    if len(quiet) >= 3:
        x = np.asarray([r["compression_mm"] / 1000 for r in quiet])
        y = np.asarray([r["contact_down_N"] for r in quiet])
        estimate = float(np.dot(x, y) / np.dot(x, x))

    return {
        "timestep_ms": run["timestep_ms"],
        "contact_samples": len(contact_rows),
        "max_abs_force_balance_residual_N": max(residuals),
        "mean_abs_force_balance_residual_N": float(np.mean(residuals)),
        "final_contact_force_N": rows[-1]["contact_down_N"],
        "final_compression_mm": rows[-1]["compression_mm"],
        "quasistatic_samples": len(quiet),
        "quasistatic_stiffness_estimate_N_per_m": estimate,
        "quasistatic_stiffness_error_percent": (
            100 * abs(estimate - run["known_stiffness_N_per_m"])
            / run["known_stiffness_N_per_m"]
            if estimate is not None else None
        ),
    }


def analyze(scene_path, control_schedule, timesteps=None, duration=1.0, sample_interval=0.01):
    if timesteps is None:
        timesteps = [0.002, 0.001, 0.0005, 0.00025]
    timesteps = list(timesteps)
    if len(timesteps) < 2:
        raise ValueError("At least two timesteps are required.")
    if len(set(timesteps)) != len(timesteps):
        raise ValueError("Duplicate timesteps are not allowed.")

    runs = {
        dt: simulate(scene_path, control_schedule, dt, duration, sample_interval)
        for dt in sorted(timesteps)
    }
    reference_dt = min(timesteps)
    reference = runs[reference_dt]
    ref_rows = reference["measurements"]

    comparisons = []
    for dt in sorted(timesteps, reverse=True):
        run = runs[dt]
        summary = summarize_run(run)
        if dt == reference_dt:
            summary["reference"] = True
        else:
            rows = run["measurements"]
            if len(rows) != len(ref_rows) or not np.allclose(
                [r["time_s"] for r in rows],
                [r["time_s"] for r in ref_rows],
                rtol=0, atol=1e-9,
            ):
                raise ValueError("Sample timestamps differ between timesteps.")
            force_errors = np.asarray([
                a["contact_down_N"] - b["contact_down_N"]
                for a, b in zip(rows, ref_rows)
            ])
            compression_errors = np.asarray([
                a["compression_mm"] - b["compression_mm"]
                for a, b in zip(rows, ref_rows)
            ])
            summary.update({
                "reference": False,
                "mean_abs_contact_force_difference_N": float(np.mean(np.abs(force_errors))),
                "peak_abs_contact_force_difference_N": float(np.max(np.abs(force_errors))),
                "peak_abs_compression_difference_mm": float(np.max(np.abs(compression_errors))),
            })
        comparisons.append(summary)

    return {
        "scene": str(Path(scene_path).resolve()),
        "reference_timestep_ms": reference_dt * 1000,
        "duration_s": duration,
        "sample_interval_s": sample_interval,
        "control_schedule": control_schedule,
        "model_stiffness_N_per_m": reference["known_stiffness_N_per_m"],
        "comparisons": comparisons,
        "limitations": [
            "The finest timestep is a numerical reference, not experimental ground truth.",
            "Force balance assumes +Z collinear slide joints, zero gravity, and one isolated sample contact.",
            "Stiffness is estimated only if at least 3 quasi-static samples meet fixed thresholds.",
            "The modeled spring is known; its recovery does not validate real material properties.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Indentation force-balance and timestep diagnostics.")
    parser.add_argument("scene")
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--timesteps", nargs="+", type=float, default=[0.002, 0.001, 0.0005, 0.00025])
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("results/indentation_reliability.json"))
    args = parser.parse_args()

    try:
        controls = json.loads(args.controls.read_text(encoding="utf-8"))
        result = analyze(args.scene, controls, args.timesteps, args.duration, args.sample_interval)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        parser.exit(2, f"ContactForge: {error}\n")

    print("\nContactForge — Indentation reliability")
    print("-" * 72)
    print(f"Reference timestep: {result['reference_timestep_ms']:.3f} ms")
    print(f"Known stiffness: {result['model_stiffness_N_per_m']:.1f} N/m")
    for row in result["comparisons"]:
        estimate = row["quasistatic_stiffness_estimate_N_per_m"]
        estimate_text = f"{estimate:.1f} N/m" if estimate is not None else "insufficient quiet samples"
        force_text = (
            "reference" if row["reference"]
            else f"{row['peak_abs_contact_force_difference_N']:.4f} N"
        )
        print(
            f"dt={row['timestep_ms']:.3f} ms | "
            f"peak force difference={force_text} | "
            f"peak balance residual={row['max_abs_force_balance_residual_N']:.6f} N | "
            f"estimated k={estimate_text}"
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
