
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot indentation force against compression."
    )

    parser.add_argument("csv_file")

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/indentation_force_curve.png"),
    )

    args = parser.parse_args()

    compression = []
    contact_force = []
    spring_force = []

    with open(args.csv_file, encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            compression.append(
                float(row["compression_mm"])
            )
            contact_force.append(
                float(row["contact_normal_force_N"])
            )
            spring_force.append(
                float(row["spring_force_N"])
            )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        compression,
        contact_force,
        label="MuJoCo contact force",
        linewidth=2,
    )

    ax.plot(
        compression,
        spring_force,
        label="Elastic spring force",
        linestyle="--",
        linewidth=2,
    )

    ax.set_xlabel("Specimen spring compression (mm)")
    ax.set_ylabel("Force (N)")
    ax.set_title("ContactForge — Indentation Force vs Compression")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
