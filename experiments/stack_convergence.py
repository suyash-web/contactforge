
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.stack_stability import measure_stack


BODY_COUNTS = [8, 16]
TIMESTEPS = [0.002, 0.001, 0.0005, 0.00025, 0.000125]
INITIAL_SPEED = 0.05

OUTPUT_DIR = Path("results")


def positions_from_report(report):
    return np.array(
        [
            body["final_position_m"]
            for body in report["bodies"]
        ],
        dtype=float,
    )


def rms_position_difference(report, reference):
    positions = positions_from_report(report)
    reference_positions = positions_from_report(reference)

    if positions.shape != reference_positions.shape:
        raise ValueError("Incompatible stack configurations.")

    difference = positions - reference_positions

    return float(
        np.sqrt(np.mean(np.sum(difference ** 2, axis=1)))
        * 1000
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("\nContactForge — Disturbed stack timestep convergence")
    print("-" * 85)

    for body_count in BODY_COUNTS:
        measurements = {}

        for timestep in TIMESTEPS:
            measurements[timestep] = measure_stack(
                body_count,
                initial_lateral_speed=INITIAL_SPEED,
                timestep=timestep,
            )
            measurement = measurements[timestep]

            actual_dt = measurement["actual_timestep_s"]
            actual_steps = measurement["simulation_steps"]

            expected_steps = round(2.5 / timestep)

            if not np.isclose(actual_dt, timestep, rtol=1e-10, atol=1e-12):
                raise RuntimeError(
                    f"Requested dt={timestep}, but simulation used {actual_dt}"
                )

            if actual_steps != expected_steps:
                raise RuntimeError(
                    f"Expected {expected_steps} steps, got {actual_steps}"
                )

            print(
                f"  Verified: dt={actual_dt * 1000:.3f} ms, "
                f"steps={actual_steps}"
            )

        reference = measurements[min(TIMESTEPS)]

        print(f"\n{body_count}-box stack:")

        for timestep in TIMESTEPS:
            measurement = measurements[timestep]

            position_error = rms_position_difference(
                measurement,
                reference,
            )

            result = {
                "bodies": body_count,
                "timestep_ms": timestep * 1000,
                "initial_speed_m_s": INITIAL_SPEED,
                "rms_final_position_difference_mm": position_error,
                "max_horizontal_drift_mm": measurement[
                    "max_horizontal_drift_mm"
                ],
                "max_tilt_degrees": measurement[
                    "max_tilt_degrees"
                ],
                "mean_late_speed_m_s": measurement[
                    "mean_late_max_linear_speed_m_s"
                ],
                "final_contacts": measurement[
                    "final_contacts"
                ],
                "reference_timestep_ms": min(TIMESTEPS) * 1000,
            }

            results.append(result)

            print(
                f"dt={result['timestep_ms']:.3f} ms | "
                f"RMS position difference={position_error:.5f} mm | "
                f"drift={result['max_horizontal_drift_mm']:.4f} mm | "
                f"tilt={result['max_tilt_degrees']:.4f} deg"
            )

    csv_path = OUTPUT_DIR / "stack_convergence.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    json_path = OUTPUT_DIR / "stack_convergence.json"
    json_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for body_count in BODY_COUNTS:
        subset = [
            result for result in results
            if result["bodies"] == body_count
            and result["timestep_ms"] > min(TIMESTEPS) * 1000
        ]

        ax.plot(
            [result["timestep_ms"] for result in subset],
            [
                result["rms_final_position_difference_mm"]
                for result in subset
            ],
            marker="o",
            label=f"{body_count} boxes",
        )

    ax.set_xlabel("Integration timestep (ms)")
    ax.set_ylabel("RMS final position difference (mm)")
    ax.set_title("ContactForge: Disturbed Stack Timestep Sensitivity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "stack_convergence.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Plot: {plot_path}")
    print(
        f"\nReference: {min(TIMESTEPS) * 1000:.3f} ms simulation. "
        "Zero reference difference is true by construction."
    )


if __name__ == "__main__":
    main()
