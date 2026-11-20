
import pytest

from contactforge.diagnostics.selector import select_configuration


def sample_record(timestep, contact_constant, error, overlap):
    return {
        "timestep_ms": timestep,
        "contact_time_constant_ms": contact_constant,
        "stiffness_error_percent": error,
        "max_overlap_mm": overlap,
        "wall_time_s": 0.01,
    }


def test_selects_largest_acceptable_timestep():
    records = [
        sample_record(1.0, 2.5, 0.025, 0.60),
        sample_record(0.5, 2.5, 0.010, 0.60),
        sample_record(0.25, 2.5, 0.005, 0.60),
    ]

    report = select_configuration(records)

    if report["accepted_count"] != 2:
        pytest.fail("Expected two acceptable configurations.")

    if report["recommended"]["timestep_ms"] != 0.5:
        pytest.fail("Expected the 0.5 ms timestep.")


def test_rejects_excessive_contact_overlap():
    records = [
        sample_record(0.5, 10.0, 0.01, 1.6),
    ]

    report = select_configuration(records)

    if report["recommended"] is not None:
        pytest.fail("Excessive contact overlap was accepted.")


def test_no_acceptable_configuration():
    records = [
        sample_record(1.0, 5.0, 0.1, 2.0),
    ]

    report = select_configuration(records)

    if report["accepted_count"] != 0:
        pytest.fail("Unexpected acceptable configuration.")


def test_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        select_configuration([], max_error_percent=-1)
