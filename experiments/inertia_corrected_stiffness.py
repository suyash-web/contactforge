
import csv
from pathlib import Path

import numpy as np


DATA_FILE = Path("results/indentation_compliant.csv")

SAMPLE_MASS = 0.05
GRAVITY = 9.81
DAMPING = 30.0
REFERENCE_STIFFNESS = 12000.0

LOADING_START = 1.05
LOADING_END = 1.23
MIN_COMPRESSION = 0.001
MIN_FORCE = 5.0


def load_measurements(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if len(rows) < 5:
        raise ValueError("Not enough samples for acceleration estimation.")

    columns = [
        "time_s",
        "stage_displacement_m",
        "sample_compression_m",
        "compression_velocity_m_s",
        "contact_force_n",
    ]

    data = {}

    for column in columns:
        data[column] = np.array(
            [float(row[column]) for row in rows],
            dtype=float
        )

        if not np.all(np.isfinite(data[column])):
            raise ValueError(f"Invalid numeric values in {column}")

    if np.any(np.diff(data["time_s"]) <= 0):
        raise ValueError("Simulation timestamps must be increasing.")

    return data


def fit_stiffness(compression, force):
    slope, intercept = np.polyfit(compression, force, 1)

    predictions = slope * compression + intercept

    residual_sum = np.sum((force - predictions) ** 2)
    total_sum = np.sum((force - np.mean(force)) ** 2)

    r_squared = (
        1 - residual_sum / total_sum
        if total_sum > 0 else float("nan")
    )

    return float(slope), float(intercept), float(r_squared)


def main():
    data = load_measurements(DATA_FILE)

    time = data["time_s"]
    stage = data["stage_displacement_m"]
    compression = data["sample_compression_m"]
    compression_velocity = data["compression_velocity_m_s"]
    contact_force = data["contact_force_n"]

    # Recover sample-surface velocity from the two DOFs.
    stage_velocity = np.gradient(stage, time, edge_order=2)
    surface_velocity = stage_velocity - compression_velocity

    # Differentiate velocity to estimate acceleration.
    surface_acceleration = np.gradient(
        surface_velocity, time, edge_order=2
    )

    # Previous quasi-static correction.
    force_without_inertia = (
        contact_force
        + SAMPLE_MASS * GRAVITY
        - DAMPING * compression_velocity
    )

    # Full vertical force balance.
    force_with_inertia = (
        force_without_inertia
        + SAMPLE_MASS * surface_acceleration
    )

    selection = (
        (time >= LOADING_START)
        & (time <= LOADING_END)
        & (compression >= MIN_COMPRESSION)
        & (contact_force >= MIN_FORCE)
    )

    x = compression[selection]

    if len(x) < 10 or np.ptp(x) < 1e-5:
        raise ValueError("Insufficient loading data.")

    old_fit = fit_stiffness(
        x, force_without_inertia[selection]
    )

    corrected_fit = fit_stiffness(
        x, force_with_inertia[selection]
    )

    print("\nContactForge — Inertia-corrected stiffness")
    print("-" * 55)
    print(f"Loading samples: {len(x)}")
    print(f"Reference stiffness: {REFERENCE_STIFFNESS:.2f} N/m")

    for label, fit in [
        ("Quasi-static", old_fit),
        ("Inertia-corrected", corrected_fit),
    ]:
        slope, intercept, r_squared = fit
        error = abs(slope - REFERENCE_STIFFNESS)
        relative_error = 100 * error / REFERENCE_STIFFNESS

        print(f"\n{label}")
        print(f"  Estimated stiffness: {slope:.4f} N/m")
        print(f"  Relative error: {relative_error:.6f}%")
        print(f"  Intercept: {intercept:.6f} N")
        print(f"  R-squared: {r_squared:.8f}")

    print(
        "\nNote: Surface acceleration is estimated by numerical "
        "differentiation and may contain discretization noise."
    )


if __name__ == "__main__":
    main()
