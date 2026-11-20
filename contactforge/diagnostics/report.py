
import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_RESULTS = Path("results")


def load_csv(path, required_columns):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        missing = set(required_columns) - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                f"{path}: missing columns {sorted(missing)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: no measurements")

    return rows


def number(row, name):
    try:
        value = float(row[name])
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(
            f"Invalid value for {name}: {row.get(name)}"
        ) from error

    if not math.isfinite(value):
        raise ValueError(f"Non-finite value for {name}")

    return value


def configuration_key(row):
    bodies = number(row, "bodies")
    timestep = number(row, "timestep_ms")

    if not bodies.is_integer() or bodies < 1:
        raise ValueError("Invalid body count")

    if timestep <= 0:
        raise ValueError("Invalid timestep")

    return int(bodies), round(timestep, 9)


def build_report(results_dir):
    trajectory = load_csv(
        results_dir / "stack_trajectory.csv",
        (
            "bodies",
            "timestep_ms",
            "reference_timestep_ms",
            "mean_trajectory_error_mm",
            "maximum_trajectory_error_mm",
            "final_position_error_mm",
        ),
    )

    reference_timesteps = {
        number(row, "reference_timestep_ms")
        for row in trajectory
    }

    if len(reference_timesteps) != 1:
        raise ValueError(
            "Trajectory results contain inconsistent reference timesteps."
        )

    reference_timestep = reference_timesteps.pop()

    if reference_timestep <= 0:
        raise ValueError(
            "Reference timestep must be positive."
        )

    contact = load_csv(
        results_dir / "contact_pair_diagnostics.csv",
        (
            "bodies",
            "timestep_ms",
            "contact_count_mismatch_percent",
            "same_pairs_different_counts_percent",
            "different_pairs_percent",
        ),
    )

    selection_path = (
        results_dir / "selected_configuration.json"
    )

    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)

    selection = json.loads(
        selection_path.read_text(encoding="utf-8")
    )

    contacts_by_key = {}

    for row in contact:
        key = configuration_key(row)

        if key in contacts_by_key:
            raise ValueError(
                f"Duplicate contact configuration: {key}"
            )

        contacts_by_key[key] = row

    combined = []
    seen = set()

    for row in trajectory:
        key = configuration_key(row)

        if key in seen:
            raise ValueError(
                f"Duplicate trajectory configuration: {key}"
            )

        seen.add(key)

        if key not in contacts_by_key:
            raise ValueError(
                f"Missing contact results for {key}"
            )

        contact_row = contacts_by_key[key]

        entry = {
            "bodies": key[0],
            "timestep_ms": key[1],
            "mean_trajectory_error_mm": number(
                row, "mean_trajectory_error_mm"
            ),
            "maximum_trajectory_error_mm": number(
                row, "maximum_trajectory_error_mm"
            ),
            "final_position_error_mm": number(
                row, "final_position_error_mm"
            ),
            "contact_count_mismatch_percent": number(
                contact_row,
                "contact_count_mismatch_percent",
            ),
            "same_pairs_different_counts_percent": number(
                contact_row,
                "same_pairs_different_counts_percent",
            ),
            "contact_pair_mismatch_percent": number(
                contact_row, "different_pairs_percent"
            ),
        }

        combined.append(entry)

    if set(contacts_by_key) != seen:
        raise ValueError(
            "Contact and trajectory configurations differ."
        )

    combined.sort(
        key=lambda item: (
            item["bodies"],
            -item["timestep_ms"],
        )
    )

    worst_trajectory = max(
        combined,
        key=lambda item: item[
            "maximum_trajectory_error_mm"
        ],
    )

    worst_pair_mismatch = max(
        combined,
        key=lambda item: item[
            "contact_pair_mismatch_percent"
        ],
    )

    return {
        "project": "ContactForge",
        "report_type": "numerical_reliability",
        "reference_timestep_ms": reference_timestep,
        "interpretation": (
            "Numerical comparisons against a finite-timestep "
            "reference; not an analytical error estimate."
        ),
        "indentation_configuration": {
            "criteria": selection.get("criteria"),
            "recommended": selection.get("recommended"),
            "accepted_count": selection.get(
                "accepted_count"
            ),
            "tested": selection.get("tested"),
        },
        "contact_stack_analysis": {
            "configurations": combined,
            "worst_trajectory_configuration":
                worst_trajectory,
            "worst_pair_mismatch_configuration":
                worst_pair_mismatch,
        },
        "limitations": [
            "Stack results are from a specific box geometry.",
            "Contact topology is sampled every 10 ms.",
            "Matching contact pairs do not imply matching forces.",
            "No universal solver-stability threshold is applied.",
            "Indentation selection and stack analysis use "
            "different benchmark scenes.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ContactForge diagnostics"
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
    )

    args = parser.parse_args()

    try:
        report = build_report(args.results_dir)

        output = (
            args.results_dir / "reliability_report.json"
        )

        output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

    except (OSError, ValueError) as error:
        parser.exit(2, f"ContactForge: {error}\n")

    print("\nContactForge — Numerical reliability report")
    print("-" * 60)

    configuration = report[
        "indentation_configuration"
    ]["recommended"]

    if configuration is not None:
        print("\nIndentation configuration:")
        print(
            f"  Timestep: "
            f"{configuration['timestep_ms']:.3f} ms"
        )
        print(
            f"  Contact time constant: "
            f"{configuration['contact_time_constant_ms']:.3f} ms"
        )

    stack = report["contact_stack_analysis"]

    print("\nWorst observed trajectory discrepancy:")
    worst = stack["worst_trajectory_configuration"]

    print(
        f"  {worst['bodies']} boxes, "
        f"dt={worst['timestep_ms']:.3f} ms"
    )
    print(
        f"  Peak RMS difference: "
        f"{worst['maximum_trajectory_error_mm']:.4f} mm"
    )

    print("\nWorst observed contact-pair mismatch:")
    worst = stack["worst_pair_mismatch_configuration"]

    print(
        f"  {worst['bodies']} boxes, "
        f"dt={worst['timestep_ms']:.3f} ms"
    )
    print(
        f"  Pair mismatch: "
        f"{worst['contact_pair_mismatch_percent']:.2f}%"
    )
    print(
        f"  Contact-count mismatch: "
        f"{worst['contact_count_mismatch_percent']:.2f}%"
    )

    print(f"\nReport: {output}")


if __name__ == "__main__":
    main()
