
import csv
from pathlib import Path

import numpy as np

from experiments.stack_trajectory import simulate_trajectory


BODY_COUNTS = [8, 16]
TIMESTEPS = [0.002, 0.001, 0.0005, 0.00025]
REFERENCE_DT = 0.000125

OUTPUT = Path("results/contact_pair_diagnostics.csv")


def compare_pairs(candidate, reference):
    if not np.allclose(
        candidate["times"],
        reference["times"],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Trajectory sample times do not match.")

    times = candidate["times"]
    candidate_pairs = candidate["contact_pairs"]
    reference_pairs = reference["contact_pairs"]

    candidate_counts = candidate["contacts"]
    reference_counts = reference["contacts"]

    if candidate_counts.shape != reference_counts.shape:
        raise ValueError("Contact-count sample shapes do not match.")

    if len(candidate_pairs) != len(reference_pairs):
        raise ValueError("Contact sample counts do not match.")

    mismatches = []
    distances = []
    count_matches_but_pairs_differ = []

    for candidate_set, reference_set in zip(
        candidate_pairs, reference_pairs
    ):
        mismatch = candidate_set != reference_set
        union = candidate_set | reference_set
        intersection = candidate_set & reference_set

        jaccard_distance = (
            1.0 - len(intersection) / len(union)
            if union else 0.0
        )

        mismatches.append(mismatch)
        distances.append(jaccard_distance)

        count_matches_but_pairs_differ.append(
            mismatch and len(candidate_set) == len(reference_set)
        )

    same_pairs = np.array(
        [
            a == b
            for a, b in zip(candidate_pairs, reference_pairs)
        ],
        dtype=bool,
    )

    different_counts = candidate_counts != reference_counts

    same_pairs_different_counts = (
        same_pairs & different_counts
    )

    different_pairs = ~same_pairs

    mismatches = np.asarray(mismatches, dtype=bool)
    distances = np.asarray(distances, dtype=float)

    first_difference = (
        float(times[np.flatnonzero(mismatches)[0]])
        if np.any(mismatches) else None
    )

    early = (times >= 0.2) & (times <= 0.6)
    late = times >= 1.0

    return {
        "pair_mismatch_percent": float(np.mean(mismatches) * 100),
        "mean_jaccard_distance": float(np.mean(distances)),
        "first_pair_difference_s": first_difference,
        "early_mismatch_percent": float(
            np.mean(mismatches[early]) * 100
        ),
        "late_mismatch_percent": float(
            np.mean(mismatches[late]) * 100
        ),
        "same_pair_count_but_different_pairs_percent": float(
            np.mean(count_matches_but_pairs_differ) * 100
        ),
        "contact_count_mismatch_percent": float(
            np.mean(different_counts) * 100
        ),
        "same_pairs_different_counts_percent": float(
            np.mean(same_pairs_different_counts) * 100
        ),
        "different_pairs_percent": float(
            np.mean(different_pairs) * 100
        ),
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results = []

    print("\nContactForge — Contact-pair diagnostics")
    print("-" * 85)

    for body_count in BODY_COUNTS:
        reference = simulate_trajectory(
            body_count, REFERENCE_DT
        )

        print(f"\n{body_count}-box stack:")

        for timestep in TIMESTEPS:
            candidate = simulate_trajectory(
                body_count, timestep
            )

            comparison = compare_pairs(candidate, reference)

            result = {
                "bodies": body_count,
                "timestep_ms": timestep * 1000,
                **comparison,
            }

            results.append(result)

            print(
                f"dt={timestep * 1000:.3f} ms | "
                f"count mismatch={result['contact_count_mismatch_percent']:.1f}% | "
                f"same pairs, different counts="
                f"{result['same_pairs_different_counts_percent']:.1f}% | "
                f"different pairs={result['different_pairs_percent']:.1f}%"
            )

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
