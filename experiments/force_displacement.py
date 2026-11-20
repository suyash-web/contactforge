
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_FILE = Path("results/indentation_compliant.csv")
OUTPUT_DIR = Path("results")

SAMPLE_MASS = 0.05
GRAVITY = 9.81
DAMPING = 30.0
REFERENCE_STIFFNESS = 12000.0

# Focus on the loading phase after contact has been established.
LOADING_START = 1.05
LOADING_END = 1.23

MIN_COMPRESSION = 0.001
MIN_FORCE = 5.0


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with INPUT_FILE.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("No simulation data found.")

    columns = [
        "time_s",
        "sample_compression_m",
        "compression_velocity_m_s",
        "contact_force_n"
    ]

    data = {
        name: np.array(
            [float(row[name]) for row in rows],
            dtype=float
        )
        for name in columns
    }

    for name, values in data.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid values in {name}")

    return data


def main():
    data = load_data()

    time = data["time_s"]
    compression = data["sample_compression_m"]
    velocity = data["compression_velocity_m_s"]
    contact_force = data["contact_force_n"]

    # Remove the modeled effects of gravity and damping.
    # This does not correct for sample-surface acceleration.
    corrected_force = (
        contact_force + SAMPLE_MASS * GRAVITY
        - DAMPING * velocity
    )

    loading = (
        (time >= LOADING_START)
        & (time <= LOADING_END)
        & (compression >= MIN_COMPRESSION)
        & (contact_force >= MIN_FORCE)
    )

    x = compression[loading]
    y = corrected_force[loading]

    if len(x) < 10 or np.ptp(x) < 1e-5:
        raise ValueError(
            "Insufficient compression variation "
            "for stiffness estimation."
        )

    # Fit the measured force-displacement relationship.
    stiffness, intercept = np.polyfit(x, y, 1)

    predicted = stiffness * x + intercept

    residual_sum = np.sum((y - predicted) ** 2)
    total_sum = np.sum((y - np.mean(y)) ** 2)

    r_squared = (
        1.0 - residual_sum / total_sum
        if total_sum > 0 else float("nan")
    )

    relative_error = (
        abs(stiffness - REFERENCE_STIFFNESS)
        / REFERENCE_STIFFNESS
    )

    report = {
        "estimated_stiffness_n_per_m": float(stiffness),
        "reference_stiffness_n_per_m": REFERENCE_STIFFNESS,
        "relative_error_percent": float(relative_error * 100),
        "fit_intercept_n": float(intercept),
        "r_squared": float(r_squared),
        "loading_samples": int(len(x)),
        "loading_window_s": [
            LOADING_START, LOADING_END
        ],
        "model_assumptions": (
            "Linear Kelvin-Voigt compliance; gravity and "
            "damping corrected; inertia not corrected."
        )
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUT_DIR / "stiffness_estimation.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        x * 1000,
        y,
        s=12,
        alpha=0.7,
        label="Simulation measurements"
    )

    order = np.argsort(x)
    ax.plot(
        x[order] * 1000,
        predicted[order],
        linewidth=2,
        label=f"Linear fit: {stiffness:.1f} N/m"
    )

    ax.plot(
        x[order] * 1000,
        REFERENCE_STIFFNESS * x[order],
        linestyle="--",
        label="Configured sample stiffness"
    )

    ax.set_xlabel("Sample compression (mm)")
    ax.set_ylabel("Gravity/damping-corrected force (N)")
    ax.set_title("ContactForge: Effective Stiffness Estimation")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "force_displacement.png",
        dpi=200
    )
    plt.close(fig)

    print("\nContactForge — Stiffness estimation")
    print("-" * 48)
    print(f"Loading samples: {len(x)}")
    print(f"Estimated stiffness: {stiffness:.2f} N/m")
    print(f"Configured stiffness: {REFERENCE_STIFFNESS:.2f} N/m")
    print(f"Relative error: {relative_error * 100:.3f}%")
    print(f"Fit intercept: {intercept:.4f} N")
    print(f"R-squared: {r_squared:.6f}")
    print(f"\nReport saved: {report_path}")
    print("Plot saved: results/force_displacement.png")


if __name__ == "__main__":
    main()
