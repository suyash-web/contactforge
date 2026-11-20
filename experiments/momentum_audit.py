
import json
from pathlib import Path

import mujoco

from contactforge.simulation.runner import prepare_control_events
from experiments.indentation_force import measure_contact_force


SCENE = "examples/scenes/controlled_indentation.xml"
CONTROLS = "examples/configs/indentation_controls.json"


def audit(timestep, start_time=0.4, end_time=0.46):
    model = mujoco.MjModel.from_xml_path(
        str(Path(SCENE).resolve())
    )
    model.opt.timestep = timestep
    data = mujoco.MjData(model)

    controls = json.loads(
        Path(CONTROLS).read_text(encoding="utf-8")
    )

    events, _ = prepare_control_events(
        model, controls, timestep, 0.6
    )

    stage_id = model.joint("stage_slide").id
    sample_id = model.joint("sample_compression").id

    stage_dof = int(model.jnt_dofadr[stage_id])
    sample_dof = int(model.jnt_dofadr[sample_id])
    sample_pos = int(model.jnt_qposadr[sample_id])

    mass = float(model.body_mass[model.body("sample").id])
    stiffness = float(model.jnt_stiffness[sample_id])
    damping = float(model.dof_damping[sample_dof])
    springref = float(model.qpos_spring[sample_pos])

    for actuator_id, value in events.get(0, []):
        data.ctrl[actuator_id] = value

    def snapshot():
        mujoco.mj_forward(model, data)

        relative_position = float(data.qpos[sample_pos])
        relative_velocity = float(data.qvel[sample_dof])
        stage_velocity = float(data.qvel[stage_dof])

        spring_force = stiffness * (
            springref - relative_position
        )
        damping_force = -damping * relative_velocity

        contact_force, _ = measure_contact_force(
            model, data, "indenter_tip", "sample_collision"
        )

        # Upward is positive.
        net_force = (
            spring_force + damping_force - contact_force
        )

        # Absolute velocity of the specimen.
        specimen_velocity = (
            stage_velocity + relative_velocity
        )

        return {
            "time": float(data.time),
            "velocity": specimen_velocity,
            "net_force": net_force,
        }

    samples = [snapshot()]

    steps = round(0.6 / timestep)

    for step in range(1, steps + 1):
        mujoco.mj_step(model, data)

        if step < steps:
            for actuator_id, value in events.get(step, []):
                data.ctrl[actuator_id] = value

        samples.append(snapshot())

    first_index = round(start_time / timestep)
    last_index = round(end_time / timestep)

    window = samples[first_index:last_index + 1]

    momentum_change = mass * (
        window[-1]["velocity"] -
        window[0]["velocity"]
    )

    # Trapezoidal integration of net force.
    impulse = 0.0

    for a, b in zip(window[:-1], window[1:]):
        impulse += (
            0.5 *
            (a["net_force"] + b["net_force"]) *
            timestep
        )

    residual = momentum_change - impulse

    print(f"\nTimestep: {timestep * 1000:.3f} ms")
    print(f"Momentum change: {momentum_change:+.8f} N s")
    print(f"Net-force impulse: {impulse:+.8f} N s")
    print(f"Balance residual: {residual:+.8f} N s")

    return residual


if __name__ == "__main__":
    for dt in [0.002, 0.001, 0.0005, 0.00025]:
        audit(dt)
