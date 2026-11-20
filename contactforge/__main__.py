import argparse
import json
import subprocess
import sys
from pathlib import Path

from contactforge.assets.mjcf import inspect_mjcf
from contactforge.assets.mesh import inspect_mesh


MODULE_COMMANDS = {
    "simulate": {
        "module": "contactforge.simulation.runner",
        "help": "Run a MuJoCo simulation scene",
    },
    "analyze": {
        "module": "contactforge.diagnostics.convergence",
        "help": "Analyze timestep sensitivity",
    },
    "indentation": {
        "module": "experiments.indentation_force",
        "help": "Measure indentation force and compression",
    },
    "reliability": {
        "module": "experiments.indentation_reliability",
        "help": "Analyze indentation numerical reliability",
    },
    "plot": {
        "module": "experiments.plot_indentation",
        "help": "Generate indentation force plots",
    },
    "report": {
        "module": "contactforge.reporting.html_report",
        "help": "Generate an offline HTML reliability report",
    },
}


def save_report(report, output):
    """Write a validation report to disk."""
    if not output:
        return

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nReport saved to {path}")


def inspect_command(args):
    """Original MJCF asset validation command."""
    report = inspect_mjcf(args.model)

    print(f"\nContactForge | {report['file']}")
    print("-" * 55)

    for body in report["bodies"]:
        print(f"\n[{body['status'].upper()}] {body['body']}")

        for error in body.get("errors", []):
            print(f"  Error: {error}")

        if body.get("reason"):
            print(f"  Note: {body['reason']}")

    print("\nSummary")
    print(f"  Passed:  {report['passed']}")
    print(f"  Failed:  {report['failed']}")
    print(f"  Skipped: {report['skipped']}")

    save_report(report, args.output)

    if report["failed"]:
        return 1

    if report["skipped"]:
        print("\nNote: Some bodies were not validated.")

    return 0


def inspect_mesh_command(args):
    """Original triangle-mesh inspection command."""
    report = inspect_mesh(args.mesh)

    print("\nContactForge | Mesh inspection")
    print("-" * 55)
    print(f"File:          {report['file']}")
    print(f"Status:        {report['status'].upper()}")

    if "faces" in report:
        print(f"Faces:         {report['faces']}")
        print(f"Vertices:      {report['vertices']}")
        print(f"Watertight:    {report['watertight']}")
        print(f"Valid volume:  {report['valid_volume']}")

    if "original_vertices" in report:
        print(
            f"Original vertices: {report['original_vertices']}"
        )

    if "volume_mesh_units_cubed" in report:
        print(
            f"Volume: {report['volume_mesh_units_cubed']:.8g} "
            "cubic mesh units"
        )

    for issue in report["issues"]:
        print(f"  Issue: {issue}")

    save_report(report, args.output)

    return 0 if report["status"] == "passed" else 1


def dispatch_module(command, arguments):
    """Run an existing simulation/diagnostic module."""
    module = MODULE_COMMANDS[command]["module"]

    process_command = [
        sys.executable,
        "-m",
        module,
        *arguments,
    ]

    try:
        result = subprocess.run(process_command, check=False)
    except OSError as error:
        print(
            f"ContactForge: failed to execute "
            f"'{command}': {error}",
            file=sys.stderr,
        )
        return 1

    return result.returncode


def build_parser():
    """Build the unified command parser."""
    parser = argparse.ArgumentParser(
        prog="contactforge",
        description="Physics asset validation and simulation diagnostics",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    mjcf_parser = commands.add_parser(
        "inspect",
        help="Validate explicit MJCF inertial properties",
    )
    mjcf_parser.add_argument("model")
    mjcf_parser.add_argument("--output", "-o")

    mesh_parser = commands.add_parser(
        "inspect-mesh",
        help="Inspect triangle mesh geometry",
    )
    mesh_parser.add_argument("mesh")
    mesh_parser.add_argument("--output", "-o")

    for name, config in MODULE_COMMANDS.items():
        commands.add_parser(
            name,
            help=config["help"],
            add_help=False,
        )

    return parser


def main(argv=None):
    """Run ContactForge's unified CLI."""
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] in MODULE_COMMANDS:
        return dispatch_module(argv[0], argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "inspect":
            return inspect_command(args)

        if args.command == "inspect-mesh":
            return inspect_mesh_command(args)

    except (OSError, ValueError) as error:
        print(f"ContactForge: {error}", file=sys.stderr)
        return 2

    return 2


if __name__ == "__main__":
    sys.exit(main())
