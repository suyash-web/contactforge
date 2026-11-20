import argparse
import json
from pathlib import Path

import numpy as np

from contactforge.simulation.runner import run_scene


DEFAULT_TIMESTEPS = [0.002, 0.001, 0.0005, 0.00025]


def compare_runs(candidate, reference):
    if candidate["body_names"] != reference["body_names"]:
        raise ValueError("Tracked body names or ordering differ.")

    candidate_obs = candidate["observations"]
    reference_obs = reference["observations"]

    if len(candidate_obs) != len(reference_obs):
        raise ValueError("Observation counts differ.")

    times_a = np.array([item["time_s"] for item in candidate_obs])
    times_b = np.array([item["time_s"] for item in reference_obs])

    if not np.allclose(times_a, times_b, rtol=0, atol=1e-9):
        raise ValueError("Observation timestamps differ.")

    positions_a = np.asarray(
        [item["positions_m"] for item in candidate_obs], dtype=float
    )
    positions_b = np.asarray(
        [item["positions_m"] for item in reference_obs], dtype=float
    )

    if positions_a.shape != positions_b.shape:
        raise ValueError("Position array shapes differ.")
    if not np.all(np.isfinite(positions_a)):
        raise ValueError("Candidate contains non-finite positions.")
    if not np.all(np.isfinite(positions_b)):
        raise ValueError("Reference contains non-finite positions.")

    difference = positions_a - positions_b
    rms_error_mm = np.sqrt(
        np.mean(np.sum(difference ** 2, axis=2), axis=1)
    ) * 1000

    count_mismatches = []
    pair_mismatches = []
    same_pairs_different_counts = []

    for a, b in zip(candidate_obs, reference_obs):
        pairs_a = {tuple(pair) for pair in a["contact_pairs"]}
        pairs_b = {tuple(pair) for pair in b["contact_pairs"]}
        count_differs = a["contact_count"] != b["contact_count"]
        pairs_differ = pairs_a != pairs_b

        count_mismatches.append(count_differs)
        pair_mismatches.append(pairs_differ)
        same_pairs_different_counts.append(
            count_differs and not pairs_differ
        )

        # Control schedules must result in the same recorded commands at
        # corresponding observation times across numerical timesteps.
        if a.get("controls", {}) != b.get("controls", {}):
            raise ValueError(
                "Applied actuator controls differ between runs."
            )

    orientations_a = np.asarray(
        [item["orientations_wxyz"] for item in candidate_obs], dtype=float
    )
    orientations_b = np.asarray(
        [item["orientations_wxyz"] for item in reference_obs], dtype=float
    )
    expected_shape = positions_a.shape[:2] + (4,)

    if (
        orientations_a.shape != expected_shape
        or orientations_b.shape != expected_shape
    ):
        raise ValueError("Orientation array shapes differ.")

    if (
        not np.all(np.isfinite(orientations_a))
        or not np.all(np.isfinite(orientations_b))
    ):
        raise ValueError("Non-finite orientation values.")

    norms_a = np.linalg.norm(orientations_a, axis=2)
    norms_b = np.linalg.norm(orientations_b, axis=2)
    if np.any(norms_a < 1e-12) or np.any(norms_b < 1e-12):
        raise ValueError("Invalid zero-length quaternion.")

    orientations_a = orientations_a / norms_a[..., None]
    orientations_b = orientations_b / norms_b[..., None]
    dot_products = np.sum(orientations_a * orientations_b, axis=2)
    dot_products = np.clip(np.abs(dot_products), 0.0, 1.0)
    angular_error_degrees = np.degrees(2.0 * np.arccos(dot_products))
    rms_orientation_error_degrees = np.sqrt(
        np.mean(angular_error_degrees ** 2, axis=1)
    )

    return {
        "mean_trajectory_error_mm": float(np.mean(rms_error_mm)),
        "max_trajectory_error_mm": float(np.max(rms_error_mm)),
        "final_position_error_mm": float(rms_error_mm[-1]),
        "time_of_max_error_s": float(times_a[np.argmax(rms_error_mm)]),
        "contact_count_mismatch_percent": float(
            100 * np.mean(count_mismatches)
        ),
        "contact_pair_mismatch_percent": float(
            100 * np.mean(pair_mismatches)
        ),
        "same_pairs_different_counts_percent": float(
            100 * np.mean(same_pairs_different_counts)
        ),
        "mean_orientation_error_degrees": float(
            np.mean(rms_orientation_error_degrees)
        ),
        "max_orientation_error_degrees": float(
            np.max(rms_orientation_error_degrees)
        ),
        "final_orientation_error_degrees": float(
            rms_orientation_error_degrees[-1]
        ),
    }


def analyze_scene(
    scene_path,
    timesteps=None,
    duration=1.0,
    sample_interval=0.01,
    initial_freejoint_velocities=None,
    control_schedule=None,
):
    if timesteps is None:
        timesteps = DEFAULT_TIMESTEPS

    timesteps = list(timesteps)
    if len(timesteps) < 2:
        raise ValueError("At least two timesteps are required.")
    if len(set(timesteps)) != len(timesteps):
        raise ValueError("Duplicate timesteps are not allowed.")

    runs = {}
    for timestep in sorted(timesteps):
        runs[timestep] = run_scene(
            scene_path,
            timestep=timestep,
            duration=duration,
            sample_interval=sample_interval,
            initial_freejoint_velocities=initial_freejoint_velocities,
            control_schedule=control_schedule,
        )

    reference_dt = min(timesteps)
    reference = runs[reference_dt]
    comparisons = []

    for timestep in sorted(timesteps, reverse=True):
        if timestep == reference_dt:
            continue

        metrics = compare_runs(runs[timestep], reference)
        comparisons.append({
            "timestep_ms": timestep * 1000,
            **metrics,
        })

    return {
        "scene": str(Path(scene_path).resolve()),
        "reference_timestep_ms": reference_dt * 1000,
        "duration_s": duration,
        "sample_interval_s": sample_interval,
        "tracked_bodies": reference["body_names"],
        "comparisons": comparisons,
        "limitations": [
            "The finest timestep is a numerical reference, not ground truth.",
            "Only body-origin positions are compared; joint state and velocities are not yet compared.",
            "Orientation is compared independently using quaternion angular distance.",
            "Control commands use piecewise-constant schedules aligned with every timestep.",
            "Contact topology is sampled at fixed time intervals.",
            "Matching contact pairs do not imply matching contact forces.",
        ],
        "initial_freejoint_velocities": (
            {} if initial_freejoint_velocities is None
            else initial_freejoint_velocities
        ),
        "control_schedule": (
            {} if control_schedule is None else control_schedule
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze timestep sensitivity of an MJCF scene."
    )
    parser.add_argument("scene")
    parser.add_argument(
        "--timesteps", type=float, nargs="+", default=DEFAULT_TIMESTEPS
    )
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/scene_convergence.json"),
    )
    parser.add_argument(
        "--initial-velocities", type=Path, default=None,
        help="JSON file mapping free-joint names to six velocities.",
    )
    parser.add_argument(
        "--controls", type=Path, default=None,
        help="JSON file defining time-dependent actuator commands.",
    )

    args = parser.parse_args()
    try:
        initial_velocities = None
        control_schedule = None

        if args.initial_velocities is not None:
            initial_velocities = json.loads(
                args.initial_velocities.read_text(encoding="utf-8")
            )
        if args.controls is not None:
            control_schedule = json.loads(
                args.controls.read_text(encoding="utf-8")
            )

        report = analyze_scene(
            args.scene,
            timesteps=args.timesteps,
            duration=args.duration,
            sample_interval=args.sample_interval,
            initial_freejoint_velocities=initial_velocities,
            control_schedule=control_schedule,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        parser.exit(2, f"ContactForge: {error}\n")

    print("\nContactForge — MJCF timestep sensitivity")
    print("-" * 72)
    print(f"Scene: {args.scene}")
    print(f"Reference: {report['reference_timestep_ms']:.3f} ms")
    print(f"Tracked bodies: {len(report['tracked_bodies'])}")

    for item in report["comparisons"]:
        print(
            f"dt={item['timestep_ms']:.3f} ms | "
            f"mean error={item['mean_trajectory_error_mm']:.5f} mm | "
            f"peak error={item['max_trajectory_error_mm']:.5f} mm | "
            f"pair mismatch={item['contact_pair_mismatch_percent']:.2f}%"
        )
        print(
            f"             mean rotation="
            f"{item['mean_orientation_error_degrees']:.5f} deg | "
            f"peak rotation="
            f"{item['max_orientation_error_degrees']:.5f} deg"
        )

    print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
