import math
import csv
import json
import statistics
from pathlib import Path
from time import perf_counter

import mujoco


BODY_COUNTS = [1, 4, 8, 16]

TIMESTEP = 0.001
DURATION = 1.0
BOX_HALF_SIZE = 0.05
BOX_MASS = 0.5

WARMUP_RUNS = 3
TIMED_RUNS = 12

OUTPUT_DIR = Path("results")


def create_model(body_count, timestep=TIMESTEP):
    if not math.isfinite(timestep) or timestep <= 0:
        raise ValueError("Timestep must be positive and finite.")
    if body_count < 1:
        raise ValueError("Body count must be positive.")

    bodies = []

    for index in range(body_count):
        # Small initial gaps avoid starting with interpenetration.
        height = 0.15 + index * 0.102

        bodies.append(f"""
        <body name="box_{index}" pos="0 0 {height}">
          <freejoint name="joint_{index}"/>

          <geom name="box_geom_{index}"
                type="box"
                size="{BOX_HALF_SIZE} {BOX_HALF_SIZE} {BOX_HALF_SIZE}"
                mass="{BOX_MASS}"
                friction="1.0 0.005 0.0001"
                solref="0.01 1"/>
        </body>
        """)

    xml = f"""
    <mujoco model="contactforge_stack">
      <option timestep="{timestep}"
              gravity="0 0 -9.81"
              integrator="implicitfast"
              iterations="100"/>

      <worldbody>
        <geom name="ground"
              type="plane"
              size="5 5 0.1"
              friction="1.0 0.005 0.0001"/>

        {''.join(bodies)}
      </worldbody>
    </mujoco>
    """

    return mujoco.MjModel.from_xml_string(xml)


def simulate(model, data, steps):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    contact_sum = 0
    peak_contacts = 0

    start = perf_counter()

    for _ in range(steps):
        mujoco.mj_step(model, data)

        count = int(data.ncon)
        contact_sum += count
        peak_contacts = max(peak_contacts, count)

    elapsed = perf_counter() - start

    return {
        "elapsed_s": elapsed,
        "mean_contacts": contact_sum / steps,
        "peak_contacts": peak_contacts,
        "final_contacts": int(data.ncon),
        "final_time_s": float(data.time),
    }


def benchmark(body_count):
    model = create_model(body_count)
    data = mujoco.MjData(model)

    steps = round(DURATION / TIMESTEP)

    for _ in range(WARMUP_RUNS):
        simulate(model, data, steps)

    runtimes = []
    mean_contacts = []
    peak_contacts = []
    final_contacts = []

    for _ in range(TIMED_RUNS):
        measurement = simulate(model, data, steps)

        runtimes.append(measurement["elapsed_s"])
        mean_contacts.append(measurement["mean_contacts"])
        peak_contacts.append(measurement["peak_contacts"])
        final_contacts.append(measurement["final_contacts"])

    median_runtime = statistics.median(runtimes)

    return {
        "bodies": body_count,
        "degrees_of_freedom": int(model.nv),
        "steps": steps,
        "timestep_ms": TIMESTEP * 1000,
        "median_runtime_ms": median_runtime * 1000,
        "std_runtime_ms": statistics.stdev(runtimes) * 1000,
        "steps_per_second": steps / median_runtime,
        "mean_contacts": statistics.mean(mean_contacts),
        "peak_contacts": max(peak_contacts),
        "final_contacts": final_contacts[-1],
        "timed_runs": TIMED_RUNS,
        "individual_runtimes_ms": [
            runtime * 1000 for runtime in runtimes
        ],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("\nContactForge — Contact-coupled stack scaling")
    print("-" * 75)

    for body_count in BODY_COUNTS:
        result = benchmark(body_count)
        results.append(result)

        print(
            f"Bodies={body_count:2d} | "
            f"DOFs={result['degrees_of_freedom']:3d} | "
            f"median={result['median_runtime_ms']:9.3f} ms | "
            f"mean contacts={result['mean_contacts']:7.2f} | "
            f"peak contacts={result['peak_contacts']:3d}"
        )

    baseline_runtime = results[0]["median_runtime_ms"]

    for result in results:
        result["runtime_vs_one_body"] = (
            result["median_runtime_ms"] / baseline_runtime
        )

    json_path = OUTPUT_DIR / "stack_scaling.json"
    json_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = OUTPUT_DIR / "stack_scaling.csv"

    columns = [
        key for key in results[0]
        if key != "individual_runtimes_ms"
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for result in results:
            writer.writerow({
                column: result[column] for column in columns
            })

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
