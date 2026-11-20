
import argparse
import json
import math
from pathlib import Path

import mujoco


def validate_settings(timestep, duration, sample_interval):
    values = {
        "timestep": timestep,
        "duration": duration,
        "sample_interval": sample_interval,
    }

    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"{name} must be positive and finite."
            )

    stride = round(sample_interval / timestep)
    steps = round(duration / timestep)

    if stride < 1:
        raise ValueError(
            "Sample interval cannot be smaller than timestep."
        )

    if not math.isclose(
        stride * timestep,
        sample_interval,
        rel_tol=0,
        abs_tol=1e-10,
    ):
        raise ValueError(
            "Sample interval must align with timestep."
        )

    if not math.isclose(
        steps * timestep,
        duration,
        rel_tol=0,
        abs_tol=1e-10,
    ):
        raise ValueError(
            "Duration must align with timestep."
        )

    if steps % stride != 0:
        raise ValueError(
            "Duration must align with sample interval."
        )

    return steps, stride


def get_contact_pairs(model, data):
    pairs = set()

    for index in range(data.ncon):
        contact = data.contact[index]

        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)

        geom1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id
        )
        geom2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id
        )

        name1 = geom1 or f"geom_id:{geom1_id}"
        name2 = geom2 or f"geom_id:{geom2_id}"

        pairs.add(tuple(sorted((name1, name2))))

    return [list(pair) for pair in sorted(pairs)]


def prepare_control_events(
    model,
    control_schedule,
    timestep,
    duration,
):
    if control_schedule is None:
        control_schedule = {}

    if not isinstance(control_schedule, dict):
        raise ValueError(
            "Control schedule must be a dictionary."
        )

    scheduled_events = {}

    for actuator_name, events in control_schedule.items():
        try:
            actuator_id = model.actuator(actuator_name).id
        except KeyError as error:
            raise ValueError(
                f"Unknown actuator: {actuator_name}"
            ) from error

        if not isinstance(events, list) or not events:
            raise ValueError(
                f"Actuator {actuator_name} needs control events."
            )

        previous_time = -1.0

        for event in events:
            if (
                not isinstance(event, dict)
                or "time_s" not in event
                or "value" not in event
            ):
                raise ValueError(
                    f"Invalid control event for {actuator_name}."
                )

            event_time = event["time_s"]
            value = event["value"]

            if not all(
                isinstance(x, (int, float))
                and math.isfinite(x)
                for x in (event_time, value)
            ):
                raise ValueError(
                    f"Invalid control event for {actuator_name}."
                )

            if event_time < 0 or event_time >= duration:
                raise ValueError(
                    "Control event time is outside simulation duration."
                )

            if event_time <= previous_time:
                raise ValueError(
                    "Control events must have increasing times."
                )

            previous_time = event_time
            event_step = round(event_time / timestep)

            if not math.isclose(
                event_step * timestep,
                event_time,
                rel_tol=0,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    "Control event time must align with timestep."
                )

            if model.actuator_ctrllimited[actuator_id]:
                low, high = model.actuator_ctrlrange[actuator_id]

                if not low <= value <= high:
                    raise ValueError(
                        f"Control for {actuator_name} outside range."
                    )

            scheduled_events.setdefault(
                event_step, []
            ).append((actuator_id, float(value)))

    return scheduled_events, control_schedule


def run_scene(
    scene_path,
    timestep=0.001,
    duration=2.5,
    sample_interval=0.01,
    initial_freejoint_velocities=None,
    control_schedule=None,
):
    steps, stride = validate_settings(
        timestep, duration, sample_interval
    )

    path = Path(scene_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    # Load the physics model and create its changing state.
    model = mujoco.MjModel.from_xml_path(
        str(path.resolve())
    )
    model.opt.timestep = timestep
    data = mujoco.MjData(model)

    if not math.isclose(
        float(model.opt.timestep),
        timestep,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Compiled model timestep mismatch."
        )

    # Apply the requested initial joint velocities.
    if initial_freejoint_velocities is None:
        initial_freejoint_velocities = {}

    if not isinstance(
        initial_freejoint_velocities, dict
    ):
        raise ValueError(
            "Initial joint velocities must be a dictionary."
        )

    for joint_name, velocities in (
        initial_freejoint_velocities.items()
    ):
        try:
            joint_id = model.joint(joint_name).id
        except KeyError as error:
            raise ValueError(
                f"Unknown joint: {joint_name}"
            ) from error

        if (
            model.jnt_type[joint_id]
            != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ValueError(
                f"Joint {joint_name} is not a free joint."
            )

        if (
            not isinstance(velocities, (list, tuple))
            or len(velocities) != 6
        ):
            raise ValueError(
                f"Free joint {joint_name} requires six velocities."
            )

        if not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            for value in velocities
        ):
            raise ValueError(
                f"Invalid velocity for joint {joint_name}."
            )

        dof_start = int(
            model.jnt_dofadr[joint_id]
        )

        data.qvel[dof_start:dof_start + 6] = velocities

    # Validate the actuator commands and their times.
    scheduled_events, control_schedule = (
        prepare_control_events(
            model,
            control_schedule,
            timestep,
            duration,
        )
    )

    # Apply time-zero commands before recording the initial state.
    for actuator_id, value in scheduled_events.get(
        0, []
    ):
        data.ctrl[actuator_id] = value

    mujoco.mj_forward(model, data)

    # Verify the velocities actually present in MuJoCo.
    applied_initial_velocities = {}

    for joint_name in initial_freejoint_velocities:
        joint_id = model.joint(joint_name).id
        dof_start = int(model.jnt_dofadr[joint_id])

        applied_initial_velocities[joint_name] = (
            data.qvel[dof_start:dof_start + 6].tolist()
        )

    # Find the jointed bodies whose poses we will track.
    body_ids = [
        body_id
        for body_id in range(1, model.nbody)
        if model.body_jntnum[body_id] > 0
    ]

    if not body_ids:
        raise ValueError(
            "Scene has no jointed bodies to track."
        )

    body_names = [
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        ) or f"body_id:{body_id}"
        for body_id in body_ids
    ]

    actuator_names = [
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        ) or f"actuator_id:{actuator_id}"
        for actuator_id in range(model.nu)
    ]

    observations = []

    # Record position, orientation, contacts, and motor controls.
    def record():
        observations.append({
            "time_s": float(data.time),
            "positions_m": [
                data.xpos[body_id].tolist()
                for body_id in body_ids
            ],
            "orientations_wxyz": [
                data.xquat[body_id].tolist()
                for body_id in body_ids
            ],
            "contact_count": int(data.ncon),
            "contact_pairs": get_contact_pairs(model, data),
            "controls": {
                name: float(data.ctrl[actuator_id])
                for actuator_id, name
                in enumerate(actuator_names)
            },
        })

    record()

    # Execute the simulation, changing controls at scheduled times.
    for step in range(1, steps + 1):
        # Integrate using the command for the current interval.
        mujoco.mj_step(model, data)

        # Update the command for the NEXT interval.
        # This makes observations at event boundaries report
        # the command active from that time onward.
        if step < steps:
            for actuator_id, value in scheduled_events.get(
                step, []
            ):
                data.ctrl[actuator_id] = value

        if step % stride == 0:
            record()

    return {
        "scene": str(path.resolve()),
        "timestep_s": float(model.opt.timestep),
        "duration_s": duration,
        "sample_interval_s": sample_interval,
        "simulation_steps": steps,
        "body_names": body_names,
        "observations": observations,
        "initial_freejoint_velocities": (
            initial_freejoint_velocities
        ),
        "applied_initial_velocities": (
            applied_initial_velocities
        ),
        "control_schedule": control_schedule,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a MuJoCo scene and record numerical diagnostics."
        )
    )

    parser.add_argument("scene")
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--initial-velocities",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/scene_run.json"),
    )

    args = parser.parse_args()

    try:
        initial_velocities = None
        controls = None

        if args.initial_velocities is not None:
            initial_velocities = json.loads(
                args.initial_velocities.read_text(
                    encoding="utf-8"
                )
            )

        if args.controls is not None:
            controls = json.loads(
                args.controls.read_text(
                    encoding="utf-8"
                )
            )

        report = run_scene(
            args.scene,
            timestep=args.timestep,
            duration=args.duration,
            sample_interval=args.sample_interval,
            initial_freejoint_velocities=initial_velocities,
            control_schedule=controls,
        )

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

    except (
        OSError,
        ValueError,
        RuntimeError,
        TypeError,
    ) as error:
        parser.exit(
            2, f"ContactForge: {error}\n"
        )

    print("\nContactForge — Scene runner")
    print("-" * 50)
    print(f"Scene: {args.scene}")
    print(
        f"Timestep: {report['timestep_s'] * 1000:.3f} ms"
    )
    print(
        f"Simulation steps: {report['simulation_steps']}"
    )
    print(
        f"Tracked bodies: {len(report['body_names'])}"
    )
    print(
        f"Observations: {len(report['observations'])}"
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
