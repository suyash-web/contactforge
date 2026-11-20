
from contactforge.assets.inertia import validate_inertia
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def parse_numbers(text, expected_count, field):
    """Read a fixed-length numeric MJCF attribute."""
    if text is None:
        raise ValueError(f"Missing attribute: {field}")

    try:
        values = [float(value) for value in text.split()]
    except ValueError as error:
        raise ValueError(
            f"{field} contains non-numeric values"
        ) from error

    if len(values) != expected_count:
        raise ValueError(
            f"{field} requires {expected_count} values, "
            f"received {len(values)}"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field} contains NaN or infinity")

    return values


def parse_inertia(inertial):
    """Extract explicit mass and inertia from an MJCF element."""
    mass_values = parse_numbers(
        inertial.get("mass"), 1, "mass"
    )
    mass = mass_values[0]

    diagonal = inertial.get("diaginertia")
    full = inertial.get("fullinertia")

    if diagonal is not None and full is not None:
        raise ValueError(
            "Specify diaginertia or fullinertia, not both"
        )

    if diagonal is not None:
        moments = parse_numbers(
            diagonal, 3, "diaginertia"
        )
        tensor = np.diag(moments)

    elif full is not None:
        values = parse_numbers(
            full, 6, "fullinertia"
        )
        xx, yy, zz, xy, xz, yz = values

        tensor = np.array([
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz]
        ])

    else:
        raise ValueError(
            "Missing diaginertia or fullinertia"
        )

    center_of_mass = parse_numbers(
        inertial.get("pos", "0 0 0"),
        3,
        "pos"
    )

    return mass, tensor, center_of_mass


def inspect_body(body, body_path):
    """Validate one body with explicitly defined inertia."""
    inertial = body.find("inertial")

    if inertial is None:
        return {
            "body": body_path,
            "status": "skipped",
            "reason": "No explicit inertial element"
        }

    try:
        mass, tensor, center_of_mass = parse_inertia(inertial)
        validation = validate_inertia(mass, tensor)

    except (ValueError, TypeError) as error:
        return {
            "body": body_path,
            "status": "failed",
            "errors": [str(error)]
        }

    return {
        "body": body_path,
        "status": "passed" if validation.valid else "failed",
        "mass": mass,
        "center_of_mass": center_of_mass,
        "inertia_tensor": tensor.tolist(),
        "principal_moments": validation.principal_moments,
        "errors": validation.errors
    }


def inspect_mjcf(file_path):
    """Inspect explicit rigid-body inertial properties in an MJCF file."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"MJCF file not found: {path}")

    try:
        tree = ET.parse(path)
    except ET.ParseError as error:
        raise ValueError(
            f"Invalid XML in {path}: {error}"
        ) from error

    root = tree.getroot()

    if root.tag != "mujoco":
        raise ValueError("Expected a <mujoco> root element")

    worldbody = root.find("worldbody")

    if worldbody is None:
        raise ValueError("MJCF has no <worldbody> element")

    results = []

    def visit(body, parent_path):
        name = body.get("name", "unnamed_body")
        current_path = f"{parent_path}/{name}"

        results.append(inspect_body(body, current_path))

        for child in body.findall("body"):
            visit(child, current_path)

    for body in worldbody.findall("body"):
        visit(body, "worldbody")

    passed = sum(r["status"] == "passed" for r in results)
    failed = sum(r["status"] == "failed" for r in results)
    skipped = sum(r["status"] == "skipped" for r in results)

    return {
        "file": str(path),
        "total_bodies": len(results),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "bodies": results
    }
