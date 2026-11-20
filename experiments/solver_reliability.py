
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from experiments.indentation_compliant import run
from experiments.material_recovery import estimate_stiffness


MATERIAL_STIFFNESS = 12000.0
MATERIAL_DAMPING = 30.0

TIMESTEPS = [0.001, 0.0005, 0.00025]
CONTACT_TIME_CONSTANTS = [0.01, 0.005, 0.0025]

OUTPUT_DIR = Path("results")


def evaluate_configuration(timestep, contact_time_constant):
    start = perf_counter()

    samples = run(
        stiffness=MATERIAL_STIFFNESS,
        damping=MATERIAL_DAMPING,
        timestep=timestep,
        contact_time_constant=contact_time_constant,
    )

    elapsed = perf_counter() - start

    estimate = estimate_stiffness(samples)
    recovered = estimate["estimated_stiffness_n_m"]

    error_percent = (
        100.0 * abs(recovered - MATERIAL_STIFFNESS)
        / MATERIAL_STIFFNESS
    )

    overlap = np.array(
        [row["contact_overlap_m"] for row in samples]
    )

    compression = np.array(
        [row["sample_compression_m"] for row in samples]
    )

    contact_force = np.array(
        [row["contact_force_n"] for row in samples]
    )

    return {
        "timestep_ms": timestep * 1000,
        "contact_time_constant_ms": contact_time_constant * 1000,
        "configured_stiffness_n_m": MATERIAL_STIFFNESS,
        "estimated_stiffness_n_m": recovered,
        "stiffness_error_percent": error_percent,
        "r_squared": estimate["r_squared"],
        "max_overlap_mm": float(np.max(overlap) * 1000),
        "max_compression_mm": float(np.max(compression) * 1000),
        "peak_contact_force_n": float(np.max(contact_force)),
        "wall_time_s": elapsed,
        "sample_count": len(samples),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for contact_time_constant in CONTACT_TIME_CONSTANTS:
        for timestep in TIMESTEPS:
            result = evaluate_configuration(
                timestep,
                contact_time_constant,
            )
            results.append(result)

            print(
                f"dt={result['timestep_ms']:.3f} ms | "
                f"solref_tc={result['contact_time_constant_ms']:.2f} ms | "
                f"k_error={result['stiffness_error_percent']:.4f}% | "
                f"overlap={result['max_overlap_mm']:.4f} mm | "
                f"wall={result['wall_time_s']:.3f} s"
            )

    csv_path = OUTPUT_DIR / "solver_reliability.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    json_path = OUTPUT_DIR / "solver_reliability.json"
    json_path.write_text(
        json.dumps(results, indent=2) + "\n"
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5)
    )

    for time_constant in CONTACT_TIME_CONSTANTS:
        subset = sorted(
            [
                row for row in results
                if np.isclose(
                    row["contact_time_constant_ms"],
                    time_constant * 1000,
                )
            ],
            key=lambda row: row["timestep_ms"],
        )

        label = f"Contact tc = {time_constant * 1000:g} ms"

        axes[0].plot(
            [row["timestep_ms"] for row in subset],
            [row["stiffness_error_percent"] for row in subset],
            marker="o",
            label=label,
        )

        axes[1].plot(
            [row["timestep_ms"] for row in subset],
            [row["max_overlap_mm"] for row in subset],
            marker="o",
            label=label,
        )

    axes[0].set_title("Stiffness recovery error")
    axes[0].set_xlabel("Timestep (ms)")
    axes[0].set_ylabel("Relative error (%)")

    axes[1].set_title("Maximum contact overlap")
    axes[1].set_xlabel("Timestep (ms)")
    axes[1].set_ylabel("Contact overlap (mm)")

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    fig.tight_layout()

    plot_path = OUTPUT_DIR / "solver_reliability.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
