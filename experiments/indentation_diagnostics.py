
import csv
from pathlib import Path

import numpy as np


CSV_PATH = Path("results/indentation_baseline.csv")

STAGE_MASS = 0.5
GRAVITY = 9.81
KP = 15000.0
KV = 200.0
FORCE_LIMIT = 500.0

# Avoid approach/retraction transients.
HOLD_START = 1.35
HOLD_END = 1.55


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)

    with CSV_PATH.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("The CSV contains no samples.")

    hold = [
        row for row in rows
        if HOLD_START <= float(row["time_s"]) <= HOLD_END
    ]

    if not hold:
        raise ValueError("No samples found in the hold interval.")

    positions = np.array([
        float(row["stage_displacement_m"]) for row in hold
    ])
    targets = np.array([
        float(row["target_m"]) for row in hold
    ])
    velocities = np.array([
        float(row["stage_velocity_m_s"]) for row in hold
    ])
    measured_forces = np.array([
        float(row["contact_force_n"]) for row in hold
    ])

    # Reconstruct the PD servo's approximate generalized force.
    servo_forces = KP * (targets - positions) - KV * velocities
    servo_forces = np.clip(
        servo_forces, -FORCE_LIMIT, FORCE_LIMIT
    )

    # Static vertical force balance: servo - gravity - contact = 0.
    predicted_forces = servo_forces - STAGE_MASS * GRAVITY

    print("\nContactForge — Hold-phase diagnostics")
    print("-" * 48)
    print(f"Samples examined: {len(hold)}")
    print(f"Mean stage position: {np.mean(positions)*1000:.4f} mm")
    print(f"Mean tracking error: {np.mean(targets-positions)*1000:.4f} mm")
    print(f"Maximum absolute stage speed: {np.max(np.abs(velocities)):.6f} m/s")
    print(f"Mean measured contact force: {np.mean(measured_forces):.4f} N")
    print(f"Mean force-balance estimate: {np.mean(predicted_forces):.4f} N")
    print(f"Mean force discrepancy: {np.mean(measured_forces-predicted_forces):.4f} N")


if __name__ == "__main__":
    main()
