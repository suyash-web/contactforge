
import csv
import json
from pathlib import Path

from experiments.stack_stability import measure_stack


BODY_COUNTS = [8, 16]
INITIAL_SPEEDS = [0.0, 0.01, 0.05]

OUTPUT_DIR = Path("results")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("\nContactForge — Stack disturbance sensitivity")
    print("-" * 80)

    for body_count in BODY_COUNTS:
        for initial_speed in INITIAL_SPEEDS:
            measurement = measure_stack(
                body_count,
                initial_lateral_speed=initial_speed,
            )

            result = {
                "bodies": body_count,
                "initial_speed_m_s": initial_speed,
                "max_drift_mm": measurement[
                    "max_horizontal_drift_mm"
                ],
                "max_tilt_deg": measurement[
                    "max_tilt_degrees"
                ],
                "mean_late_speed_m_s": measurement[
                    "mean_late_max_linear_speed_m_s"
                ],
                "final_contacts": measurement[
                    "final_contacts"
                ],
            }

            results.append(result)

            print(
                f"Bodies={body_count:2d} | "
                f"kick={initial_speed:.3f} m/s | "
                f"drift={result['max_drift_mm']:9.3f} mm | "
                f"tilt={result['max_tilt_deg']:8.4f} deg | "
                f"late speed={result['mean_late_speed_m_s']:.6f} m/s"
            )

    csv_path = OUTPUT_DIR / "stack_perturbation.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    json_path = OUTPUT_DIR / "stack_perturbation.json"
    json_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
