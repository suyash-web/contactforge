
import csv
import json
from pathlib import Path

import numpy as np

from experiments.indentation_compliant import run


STIFFNESSES = [6000.0, 12000.0, 24000.0]
SAMPLE_MASS = 0.05
GRAVITY = 9.81
DAMPING = 30.0

OUTPUT_DIR = Path("results")
LOADING_START = 1.05
LOADING_END = 1.23


def estimate_stiffness(samples):
    def column(name):
        return np.array(
            [float(row[name]) for row in samples],
            dtype=float
        )

    time = column("time_s")
    stage = column("stage_displacement_m")
    compression = column("sample_compression_m")
    compression_velocity = column("compression_velocity_m_s")
    contact_force = column("contact_force_n")

    if not np.all(np.isfinite(time)):
        raise ValueError("Non-finite timestamps.")

    if np.any(np.diff(time) <= 0):
        raise ValueError("Timestamps must increase.")

    # Surface acceleration is distinct from stage acceleration.
    stage_velocity = np.gradient(stage, time, edge_order=2)
    surface_velocity = stage_velocity - compression_velocity
    surface_acceleration = np.gradient(
        surface_velocity, time, edge_order=2
    )

    corrected_force = (
        contact_force
        + SAMPLE_MASS * GRAVITY
        + SAMPLE_MASS * surface_acceleration
        - DAMPING * compression_velocity
    )

    selected = (
        (time >= LOADING_START)
        & (time <= LOADING_END)
        & (compression >= 0.001)
        & (contact_force >= 5.0)
    )

    x = compression[selected]
    y = corrected_force[selected]

    if len(x) < 10 or np.ptp(x) < 1e-5:
        raise ValueError("Insufficient loading data for estimation.")

    stiffness, intercept = np.polyfit(x, y, 1)

    predicted = stiffness * x + intercept
    total_variation = np.sum((y - np.mean(y)) ** 2)

    if total_variation <= 0:
        raise ValueError("Insufficient force variation.")

    r_squared = 1 - (
        np.sum((y - predicted) ** 2) / total_variation
    )

    return {
        "estimated_stiffness_n_m": float(stiffness),
        "intercept_n": float(intercept),
        "r_squared": float(r_squared),
        "samples_used": int(len(x))
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for configured_stiffness in STIFFNESSES:
        samples = run(
            stiffness=configured_stiffness,
            damping=DAMPING
        )

        estimate = estimate_stiffness(samples)
        recovered = estimate["estimated_stiffness_n_m"]

        error = (
            abs(recovered - configured_stiffness)
            / configured_stiffness * 100
        )

        result = {
            "configured_stiffness_n_m": configured_stiffness,
            **estimate,
            "relative_error_percent": error,
            "peak_contact_force_n": max(
                row["contact_force_n"] for row in samples
            ),
            "max_compression_mm": max(
                row["sample_compression_m"] for row in samples
            ) * 1000,
            "max_contact_overlap_mm": max(
                row["contact_overlap_m"] for row in samples
            ) * 1000
        }

        results.append(result)

        print(
            f"k={configured_stiffness:8.1f} N/m | "
            f"recovered={recovered:10.2f} N/m | "
            f"error={error:.4f}% | "
            f"R²={estimate['r_squared']:.8f}"
        )

    csv_path = OUTPUT_DIR / "material_recovery.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    json_path = OUTPUT_DIR / "material_recovery.json"
    json_path.write_text(
        json.dumps(results, indent=2) + "\n"
    )

    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
