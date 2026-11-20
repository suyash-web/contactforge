# ContactForge

A Python toolkit for **MuJoCo physics-asset inspection, scene simulation, timestep-convergence analysis, and controlled-indentation diagnostics**.

ContactForge helps engineers identify *numerical sensitivity* in contact simulations. Agreement across timesteps is useful evidence of numerical convergence; it is **not** a guarantee of physical accuracy or agreement with experimental hardware.

## Requirements

- Python **3.12 or newer** (development baseline: Python 3.12)
- macOS, Linux, or another platform supported by the required MuJoCo Python wheels
- MuJoCo and other dependencies installed through pip

## Installation

From the repository root, create and activate a Python environment, then install the project:

```bash
python -m pip install -e '.[dev]'
```

For a regular installation without development dependencies:

```bash
python -m pip install .
```

The historical `requirements-lock.txt` records one known environment, while `pyproject.toml` specifies dependency compatibility ranges. A fresh install is not guaranteed to resolve to exactly the same versions as that lock file.

## Command line

Both entry points are supported:

```bash
contactforge --help
python -m contactforge --help
```

Commands:

| Command | Purpose |
|---|---|
| `inspect` | Validate explicit MJCF inertial properties |
| `inspect-mesh` | Inspect mesh geometry |
| `simulate` | Run a MuJoCo MJCF scene |
| `analyze` | Compare scene trajectories at different timesteps |
| `indentation` | Calculate indentation forces and compression |
| `reliability` | Evaluate indentation force convergence and stiffness estimates |
| `plot` | Plot indentation measurements |
| `report` | Create a self-contained offline HTML report |

Run `contactforge COMMAND --help` for command-specific options.

### Reproducible examples

Run the commands below from the **repository root**, where `examples/` is available. For a non-editable installation from another directory, supply absolute paths to your scene/configuration/input files. The sample files are part of the source repository rather than installed package data.

**Simulate a falling box:**

```bash
contactforge simulate examples/scenes/falling_box.xml \
  --timestep 0.0005 --duration 1.0 --sample-interval 0.01 \
  --output results/cli_falling_box.json
```

**Analyze a motor-controlled stage:**

```bash
contactforge analyze examples/scenes/actuated_stage.xml \
  --controls examples/configs/stage_controls.json \
  --duration 0.8 --sample-interval 0.01 \
  --timesteps 0.002 0.001 0.0005 0.00025 \
  --output results/cli_stage_analysis.json
```

**Evaluate indentation reliability:**

```bash
contactforge reliability examples/scenes/controlled_indentation.xml \
  --controls examples/configs/indentation_controls.json \
  --timesteps 0.002 0.001 0.0005 0.00025 \
  --duration 1.0 --sample-interval 0.01 \
  --output results/cli_indentation_reliability.json
```

**Build the offline HTML report:**

```bash
contactforge report \
  --stage results/cli_stage_analysis.json \
  --reliability results/cli_indentation_reliability.json \
  --indentation results/indentation_force.json \
  --output results/contactforge_report.html
```

The report command expects `results/indentation_force.json` to have been created by the indentation experiment first. Run `contactforge indentation --help` for its exact arguments.

On macOS, open the resulting report with:

```bash
open results/contactforge_report.html
```

## Testing

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/ -q
```

The baseline before release packaging was **72 passing tests**. The packaging changes add **five** new tests; expect 77 passing tests if all remain discoverable. The new tests check package metadata, importability of experiment-backed commands, and module execution outside the source directory.

## Technical scope and limitations

- The timestep-reference run is a **numerical** reference, not a ground-truth measurement.
- A low reported force-balance residual can coexist with large impact-force discrepancies between timesteps; numerical precision and physical fidelity are different questions.
- Some indentation routines assume experiment-specific actuator, joint, and geometry names from `examples/scenes/controlled_indentation.xml`.
- The indentation-force comparison is sensitive to impact-onset alignment and sampling interval.
- The `experiments` modules are shipped with the Python package because CLI subcommands currently delegate to them. Future versions may move reusable logic into `contactforge` directly.
- Model inputs (`examples/`) and generated outputs (`results/`) are not automatically installed as package resources.

## Project layout

```text
contactforge/       Main Python package and CLI
  assets/           Inertia, MJCF, and mesh validation
  diagnostics/      Convergence, selection, and reports
  reporting/        Offline HTML report generator
  simulation/       MuJoCo scene runner
experiments/        Reproducible research scripts
examples/           Scenes, mesh fixtures, and control schedules
tests/              Automated regression tests
results/            Generated files (ignored by Git)
```

## Development workflow

```bash
python -m pytest tests/ -q
git status
```

Tag the release only after the tests, console entry point, built-wheel installation, and representative simulation/report workflows succeed.

## License

No open-source license has been granted for this repository. Add a `LICENSE` file and update `pyproject.toml` if you decide to publish ContactForge under an open-source license.
