
import csv
import json
import statistics
from pathlib import Path
from time import perf_counter

from experiments.indentation_compliant import run


OUTPUT_DIR = Path("results")

CONFIGURATIONS = {
    "standard": {
        "timestep": 0.0005,
        "contact_time_constant": 0.0025,
    },
    "strict": {
        "timestep": 0.00025,
        "contact_time_constant": 0.0025,
    },
}

WARMUP_RUNS = 2
TIMED_RUNS = 10


def benchmark_configuration(name, settings):
    for _ in range(WARMUP_RUNS):
        run(**settings)

    durations = []
    sample_counts = []

    for _ in range(TIMED_RUNS):
        start = perf_counter()
        samples = run(**settings)
        elapsed = perf_counter() - start

        durations.append(elapsed)
        sample_counts.append(len(samples))

    median_seconds = statistics.median(durations)
    mean_seconds = statistics.mean(durations)
    std_seconds = statistics.stdev(durations)

    return {
        "configuration": name,
        "timestep_ms": settings["timestep"] * 1000,
        "contact_time_constant_ms":
            settings["contact_time_constant"] * 1000,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "simulation_steps": sample_counts[0],
        "median_runtime_ms": median_seconds * 1000,
        "mean_runtime_ms": mean_seconds * 1000,
        "std_runtime_ms": std_seconds * 1000,
        "median_sim_steps_per_second":
            sample_counts[0] / median_seconds,
        "individual_runtimes_ms": [
            duration * 1000 for duration in durations
        ],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for name, settings in CONFIGURATIONS.items():
        result = benchmark_configuration(name, settings)
        results.append(result)

        print(f"\n{name.upper()}")
        print(f"  Timestep: {result['timestep_ms']:.3f} ms")
        print(f"  Steps: {result['simulation_steps']}")
        print(
            f"  Median runtime: "
            f"{result['median_runtime_ms']:.3f} ms"
        )
        print(
            f"  Runtime std dev: "
            f"{result['std_runtime_ms']:.3f} ms"
        )
        print(
            f"  Median steps/s: "
            f"{result['median_sim_steps_per_second']:,.0f}"
        )

    report_path = OUTPUT_DIR / "performance_benchmark.json"
    report_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = OUTPUT_DIR / "performance_benchmark.csv"

    columns = [
        "configuration",
        "timestep_ms",
        "contact_time_constant_ms",
        "warmup_runs",
        "timed_runs",
        "simulation_steps",
        "median_runtime_ms",
        "mean_runtime_ms",
        "std_runtime_ms",
        "median_sim_steps_per_second",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for result in results:
            writer.writerow({
                field: result[field] for field in columns
            })

    standard = results[0]["median_runtime_ms"]
    strict = results[1]["median_runtime_ms"]

    print("\nPerformance comparison")
    print("-" * 45)
    print(f"Strict / standard runtime ratio: {strict / standard:.3f}x")
    print(f"\nJSON: {report_path}")
    print(f"CSV: {csv_path}")
    print(
        "\nBenchmark includes model creation, simulation, "
        "and Python measurement collection."
    )


if __name__ == "__main__":
    main()
