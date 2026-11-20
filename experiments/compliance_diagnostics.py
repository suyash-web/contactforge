
import csv
from pathlib import Path

import numpy as np


CSV_PATH = Path("results/indentation_compliant.csv")

SAMPLE_MASS = 0.05
GRAVITY = 9.81
STIFFNESS = 12000.0
MAX_COMPRESSION = 0.025

HOLD_START = 1.35
HOLD_END = 1.55


def load_hold_samples():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)

    with CSV_PATH.open(newline="") as file:
        rows = list(csv.DictReader(file))

    hold = [
        row for row in rows
        if HOLD_START <= float(row["time_s"]) <= HOLD_END
    ]

    if not hold:
        raise ValueError("No samples found in hold interval.")

    return hold


def main():
    hold = load_hold_samples()

    compression = np.array([
        float(row["sample_compression_m"])
        for row in hold
    ])

    contact_force = np.array([
        float(row["contact_force_n"])
        for row in hold
    ])

    spring_damper_force = np.array([
        float(row["spring_damper_force_n"])
        for row in hold
    ])

    overlap = np.array([
        float(row["contact_overlap_m"])
        for row in hold
    ])

    compression_velocity = np.array([
        float(row["compression_velocity_m_s"])
        for row in hold
    ])

    # Quasi-static vertical equilibrium of the sample surface.
    expected_contact_force = (
        spring_damper_force - SAMPLE_MASS * GRAVITY
    )

    residual = contact_force - expected_contact_force

    print("\nContactForge — Compliance diagnostics")
    print("-" * 50)

    print(f"Hold samples: {len(hold)}")
    print(
        f"Mean compression: "
        f"{np.mean(compression)*1000:.4f} mm"
    )
    print(
        f"Mean contact overlap: "
        f"{np.mean(overlap)*1000:.4f} mm"
    )
    print(
        f"Mean spring-damper force: "
        f"{np.mean(spring_damper_force):.4f} N"
    )
    print(
        f"Mean measured contact force: "
        f"{np.mean(contact_force):.4f} N"
    )
    print(
        f"Mean force-balance estimate: "
        f"{np.mean(expected_contact_force):.4f} N"
    )
    print(
        f"Mean force discrepancy: "
        f"{np.mean(residual):.6f} N"
    )
    print(
        f"Maximum absolute discrepancy: "
        f"{np.max(np.abs(residual)):.6f} N"
    )
    print(
        f"Maximum compression speed: "
        f"{np.max(np.abs(compression_velocity)):.8f} m/s"
    )

    if np.max(compression) >= MAX_COMPRESSION - 0.0001:
        print("\nWARNING: Compression joint is near its limit.")
    else:
        print("\nCompression joint has not reached its limit.")

    print(
        "\nNote: Force-balance estimation assumes "
        "negligible sample acceleration and no "
        "additional joint-limit reaction."
    )


if __name__ == "__main__":
    main()
