
import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED_FIELDS = (
    "timestep_ms",
    "contact_time_constant_ms",
    "stiffness_error_percent",
    "max_overlap_mm",
    "wall_time_s",
)


def load_results(csv_path):
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if not set(REQUIRED_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Benchmark CSV is missing required fields.")

        results = []

        for line_number, row in enumerate(reader, start=2):
            try:
                record = {
                    field: float(row[field])
                    for field in REQUIRED_FIELDS
                }
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(
                    f"Invalid numeric value on line {line_number}"
                ) from error

            if not all(math.isfinite(v) for v in record.values()):
                raise ValueError(
                    f"Non-finite measurement on line {line_number}"
                )

            if any(record[field] <= 0 for field in (
                "timestep_ms",
                "contact_time_constant_ms",
                "wall_time_s",
            )):
                raise ValueError(
                    f"Nonpositive timestep, contact constant, "
                    f"or runtime on line {line_number}"
                )

            if record["stiffness_error_percent"] < 0:
                raise ValueError("Stiffness error cannot be negative.")

            if record["max_overlap_mm"] < 0:
                raise ValueError("Contact overlap cannot be negative.")

            results.append(record)

    if not results:
        raise ValueError("Benchmark CSV contains no measurements.")

    return results


def select_configuration(
    results,
    max_error_percent=0.02,
    max_overlap_mm=1.0,
):
    if not math.isfinite(max_error_percent) or max_error_percent < 0:
        raise ValueError("Invalid maximum stiffness error.")

    if not math.isfinite(max_overlap_mm) or max_overlap_mm < 0:
        raise ValueError("Invalid maximum contact overlap.")

    accepted = []
    rejected = []

    for record in results:
        reasons = []

        if record["stiffness_error_percent"] > max_error_percent:
            reasons.append("stiffness error exceeds threshold")

        if record["max_overlap_mm"] > max_overlap_mm:
            reasons.append("contact overlap exceeds threshold")

        entry = {**record, "rejection_reasons": reasons}

        if reasons:
            rejected.append(entry)
        else:
            accepted.append(entry)

    # A larger timestep usually means fewer integration steps.
    # Do not use a single noisy wall-clock measurement to rank ties.
    accepted.sort(
        key=lambda row: (
            -row["timestep_ms"],
            row["max_overlap_mm"],
            row["stiffness_error_percent"],
            row["contact_time_constant_ms"],
        )
    )

    return {
        "criteria": {
            "max_stiffness_error_percent": max_error_percent,
            "max_contact_overlap_mm": max_overlap_mm,
            "selection_policy": (
                "Largest acceptable timestep, then smallest "
                "overlap and stiffness error"
            ),
        },
        "tested": len(results),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "recommended": accepted[0] if accepted else None,
        "accepted": accepted,
        "rejected": rejected,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Select an acceptable MuJoCo simulation configuration"
    )

    parser.add_argument(
        "--input",
        default="results/solver_reliability.csv"
    )
    parser.add_argument(
        "--output",
        default="results/selected_configuration.json"
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=0.02
    )
    parser.add_argument(
        "--max-overlap",
        type=float,
        default=1.0
    )

    args = parser.parse_args()

    try:
        records = load_results(args.input)
        report = select_configuration(
            records,
            max_error_percent=args.max_error,
            max_overlap_mm=args.max_overlap,
        )

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8"
        )

    except (OSError, ValueError) as error:
        parser.exit(2, f"ContactForge: {error}\n")

    print("\nContactForge — Configuration selection")
    print("-" * 48)
    print(f"Tested:   {report['tested']}")
    print(f"Accepted: {report['accepted_count']}")
    print(f"Rejected: {report['rejected_count']}")

    selected = report["recommended"]

    if selected is None:
        print("\nNo configuration satisfies all criteria.")
        print(f"Report: {output}")
        return 1

    print("\nRecommended configuration:")
    print(f"  Timestep: {selected['timestep_ms']:.3f} ms")
    print(
        f"  Contact time constant: "
        f"{selected['contact_time_constant_ms']:.3f} ms"
    )
    print(
        f"  Stiffness error: "
        f"{selected['stiffness_error_percent']:.5f}%"
    )
    print(
        f"  Contact overlap: "
        f"{selected['max_overlap_mm']:.4f} mm"
    )
    print(f"  Recorded runtime: {selected['wall_time_s']:.4f} s")
    print(f"\nReport: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
