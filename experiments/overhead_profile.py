
import json
import statistics
from pathlib import Path
from time import perf_counter

import mujoco

from experiments.indentation_baseline import commanded_position
from experiments.indentation_compliant import create_model, run


OUTPUT = Path("results/overhead_profile.json")

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

DURATION = 2.5
WARMUP = 3
REPETITIONS = 20


def stepping_only(settings):
    model = create_model(**settings)
    data = mujoco.MjData(model)

    count = round(DURATION / settings["timestep"])

    # Keep model compilation and allocation outside the timer.
    start = perf_counter()

    for _ in range(count):
        data.ctrl[0] = commanded_position(data.time)
        mujoco.mj_step(model, data)

    elapsed = perf_counter() - start

    return elapsed, count


def full_pipeline(settings):
    start = perf_counter()
    samples = run(**settings)
    elapsed = perf_counter() - start

    return elapsed, len(samples)


def measure(workload, settings):
    for _ in range(WARMUP):
        workload(settings)

    durations = []
    counts = []

    for _ in range(REPETITIONS):
        elapsed, count = workload(settings)
        durations.append(elapsed * 1000)
        counts.append(count)

    if len(set(counts)) != 1:
        raise RuntimeError("Simulation step counts changed.")

    return {
        "median_ms": statistics.median(durations),
        "mean_ms": statistics.mean(durations),
        "std_ms": statistics.stdev(durations),
        "steps": counts[0],
        "runtimes_ms": durations,
    }


def main():
    results = {}

    for name, settings in CONFIGURATIONS.items():
        print(f"\n{name.upper()}")

        stepping = measure(stepping_only, settings)
        full = measure(full_pipeline, settings)

        ratio = full["median_ms"] / stepping["median_ms"]

        results[name] = {
            "settings": settings,
            "stepping_only": stepping,
            "full_pipeline": full,
            "full_to_stepping_ratio": ratio,
        }

        print(
            f"  Stepping-only median: "
            f"{stepping['median_ms']:.3f} ms"
        )
        print(
            f"  Full pipeline median: "
            f"{full['median_ms']:.3f} ms"
        )
        print(f"  Full/stepping ratio: {ratio:.3f}x")
        print(f"  Steps: {stepping['steps']}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved: {OUTPUT}")
    print(
        "Stepping-only includes Python actuator updates "
        "and MuJoCo stepping."
    )
    print(
        "Full pipeline also includes model creation, "
        "contact extraction, and measurement recording."
    )


if __name__ == "__main__":
    main()
