
import csv
import json
import math
import statistics
from pathlib import Path
from time import perf_counter

import mujoco


OUTPUT_DIR = Path("results")

BODY_COUNTS = [1, 10, 50, 100]

TIMESTEP = 0.001
DURATION = 0.6
SPHERE_RADIUS = 0.05
SPHERE_MASS = 0.2

WARMUP_RUNS = 3
TIMED_RUNS = 12


def create_model(body_count):
    if body_count < 1:
        raise ValueError("Body count must be positive.")

    columns = math.ceil(math.sqrt(body_count))
    spacing = 0.3
    bodies = []

    for index in range(body_count):
        row, column = divmod(index, columns)

        x = column * spacing
        y = row * spacing

        bodies.append(f"""
        <body name="sphere_{index}" pos="{x} {y} 0.3">
          <freejoint/>
          <geom name="collision_{index}"
                type="sphere"
                size="{SPHERE_RADIUS}"
                mass="{SPHERE_MASS}"
                solref="0.01 1"/>
        </body>
        """)

    xml = f"""
    <mujoco model="contactforge_scaling">
      <option timestep="{TIMESTEP}"
              gravity="0 0 -9.81"
              integrator="Euler"
              iterations="50"/>

      <worldbody>
        <geom name="ground"
              type="plane"
              size="20 20 0.1"/>

        {''.join(bodies)}
      </worldbody>
    </mujoco>
    """

    return mujoco.MjModel.from_xml_string(xml)


def simulate(model, data, steps):
    mujoco.mj_resetData(model, data)

    contact_sum = 0
    peak_contacts = 0

    start = perf_counter()

    for _ in range(steps):
        mujoco.mj_step(model, data)

        contacts = data.ncon
        contact_sum += contacts
        peak_contacts = max(peak_contacts, contacts)

    elapsed = perf_counter() - start

    return {
        "elapsed_s": elapsed,
        "mean_contacts": contact_sum / steps,
        "peak_contacts": peak_contacts,
        "final_contacts": int(data.ncon),
    }


def benchmark(body_count):
    model = create_model(body_count)
    data = mujoco.MjData(model)

    steps = round(DURATION / TIMESTEP)

    for _ in range(WARMUP_RUNS):
        simulate(model, data, steps)

    durations = []
    contact_counts = []

    for _ in range(TIMED_RUNS):
        result = simulate(model, data, steps)

        durations.append(result["elapsed_s"])
        contact_counts.append(result["mean_contacts"])

    median_runtime = statistics.median(durations)

    return {
        "bodies": body_count,
        "degrees_of_freedom": int(model.nv),
        "steps": steps,
        "timestep_ms": TIMESTEP * 1000,
        "median_runtime_ms": median_runtime * 1000,
        "mean_runtime_ms": statistics.mean(durations) * 1000,
        "std_runtime_ms": statistics.stdev(durations) * 1000,
        "simulation_steps_per_second": steps / median_runtime,
        "mean_active_contacts": statistics.mean(contact_counts),
        "peak_contacts": result["peak_contacts"],
        "final_contacts": result["final_contacts"],
        "timed_runs": TIMED_RUNS,
        "individual_runtimes_ms": [
            duration * 1000 for duration in durations
        ],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("\nContactForge — CPU scene-complexity scaling")
    print("-" * 65)

    for body_count in BODY_COUNTS:
        result = benchmark(body_count)
        results.append(result)

        print(
            f"Bodies={body_count:3d} | "
            f"DOFs={result['degrees_of_freedom']:4d} | "
            f"median={result['median_runtime_ms']:9.3f} ms | "
            f"steps/s={result['simulation_steps_per_second']:,.0f} | "
            f"peak contacts={result['peak_contacts']}"
        )

    baseline_runtime = results[0]["median_runtime_ms"]

    for result in results:
        result["runtime_vs_one_body"] = (
            result["median_runtime_ms"] / baseline_runtime
        )

    json_path = OUTPUT_DIR / "cpu_scaling.json"
    json_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8"
    )

    csv_path = OUTPUT_DIR / "cpu_scaling.csv"

    columns = [
        "bodies",
        "degrees_of_freedom",
        "steps",
        "timestep_ms",
        "median_runtime_ms",
        "mean_runtime_ms",
        "std_runtime_ms",
        "simulation_steps_per_second",
        "mean_active_contacts",
        "peak_contacts",
        "final_contacts",
        "runtime_vs_one_body",
        "timed_runs",
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
