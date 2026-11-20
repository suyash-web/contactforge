"""Generate a self-contained ContactForge numerical experiment report."""

import argparse
import html
import json
import math
from pathlib import Path


def load_report(path, required):
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{source}: top-level JSON must be an object")
    missing = sorted(set(required) - set(data))
    if missing:
        raise ValueError(f"{source}: missing fields: {', '.join(missing)}")
    return data


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def fmt(value, digits=4):
    return f"{number(value, 'metric'):.{digits}f}"


def esc(value):
    return html.escape(str(value), quote=True)


def table(headers, rows):
    head = "".join(f"<th scope='col'>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def line_chart(points, x_label, y_label, color="#2563eb"):
    """Build an inline SVG chart; points must be finite numeric pairs."""
    coords = [(number(x, x_label), number(y, y_label)) for x, y in points]
    if not coords:
        return "<p class='muted'>No samples available.</p>"
    width, height = 780, 300
    left, right, top, bottom = 76, 24, 25, 56
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if math.isclose(xmin, xmax, abs_tol=1e-12):
        xmin -= 0.5
        xmax += 0.5
    if math.isclose(ymin, ymax, abs_tol=1e-12):
        ymin -= 0.5
        ymax += 0.5
    else:
        padding = (ymax - ymin) * 0.08
        ymin -= padding
        ymax += padding

    def sx(x):
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y):
        return top + (ymax - y) / (ymax - ymin) * plot_h

    polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in coords)
    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='{esc(y_label)} against {esc(x_label)}' xmlns='http://www.w3.org/2000/svg'>",
        f"<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='#f8fafc' />",
    ]
    for i in range(5):
        fraction = i / 4
        y = top + fraction * plot_h
        value = ymax - fraction * (ymax - ymin)
        parts.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{left + plot_w}' y2='{y:.2f}' stroke='#e2e8f0' />"
            f"<text x='{left - 10}' y='{y + 4:.2f}' text-anchor='end' font-size='12' fill='#475569'>{value:.3g}</text>"
        )
        x = left + fraction * plot_w
        xvalue = xmin + fraction * (xmax - xmin)
        parts.append(
            f"<text x='{x:.2f}' y='{height - 31}' text-anchor='middle' font-size='12' fill='#475569'>{xvalue:.3g}</text>"
        )
    parts += [
        f"<polyline points='{polyline}' fill='none' stroke='{color}' stroke-width='2.5' stroke-linejoin='round' />",
        f"<text x='{width / 2}' y='{height - 7}' text-anchor='middle' font-size='13' fill='#334155'>{esc(x_label)}</text>",
        f"<text transform='translate(18 {height / 2}) rotate(-90)' text-anchor='middle' font-size='13' fill='#334155'>{esc(y_label)}</text>",
        "</svg>",
    ]
    return "".join(parts)


def limitations_section(title, report):
    limitations = report.get("limitations", [])
    if not limitations:
        return ""
    if not isinstance(limitations, list):
        raise ValueError(f"{title} limitations must be a list")
    content = "".join(f"<li>{esc(item)}</li>" for item in limitations)
    return f"<h3>{esc(title)}</h3><ul>{content}</ul>"


def render(stage, reliability, force):
    for key in ("comparisons",):
        if not isinstance(stage[key], list) or not isinstance(reliability[key], list):
            raise ValueError("comparisons must be lists")
    if not isinstance(force["measurements"], list):
        raise ValueError("measurements must be a list")

    stage_rows = []
    stage_plot = []
    for row in sorted(stage["comparisons"], key=lambda r: number(r["timestep_ms"], "timestep_ms")):
        dt = number(row["timestep_ms"], "timestep_ms")
        peak = number(row["max_trajectory_error_mm"], "max_trajectory_error_mm")
        stage_rows.append((fmt(dt, 3), fmt(row["mean_trajectory_error_mm"]), fmt(peak),
                           fmt(row["max_orientation_error_degrees"]),
                           fmt(row["contact_pair_mismatch_percent"], 2)))
        stage_plot.append((dt, peak))

    reliable_rows = []
    reliable_plot = []
    for row in sorted(reliability["comparisons"], key=lambda r: number(r["timestep_ms"], "timestep_ms")):
        dt = number(row["timestep_ms"], "timestep_ms")
        is_ref = row.get("reference", False)
        difference = row.get("peak_abs_contact_force_difference_N")
        diff_text = "Reference" if is_ref else fmt(difference)
        reliable_rows.append((fmt(dt, 3), diff_text,
                              fmt(row["quasistatic_stiffness_estimate_N_per_m"], 2),
                              fmt(row["quasistatic_stiffness_error_percent"], 4),
                              fmt(row["max_abs_force_balance_residual_N"], 6)))
        if not is_ref:
            reliable_plot.append((dt, number(difference, "peak_abs_contact_force_difference_N")))

    measurements = force["measurements"]
    times_force = [(row["time_s"], row["contact_normal_force_N"]) for row in measurements]
    times_compression = [(row["time_s"], row["compression_mm"]) for row in measurements]
    force_curve = [(row["compression_mm"], row["contact_normal_force_N"]) for row in measurements]

    count = len(measurements)
    peak_force = max((number(row["contact_normal_force_N"], "contact_normal_force_N")
                      for row in measurements), default=0)
    final_compression = (number(measurements[-1]["compression_mm"], "compression_mm")
                         if measurements else 0)

    configuration = table(
        ("Input", "Scene", "Duration (s)", "Sampling interval (s)"),
        (
            ("Stage convergence", stage["scene"], fmt(stage["duration_s"], 3), fmt(stage["sample_interval_s"], 4)),
            ("Indentation reliability", reliability["scene"], fmt(reliability["duration_s"], 3), fmt(reliability["sample_interval_s"], 4)),
            ("Indentation measurements", force["scene"], fmt(force["duration_s"], 3), fmt(force["sample_interval_s"], 4)),
        ),
    )

    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>ContactForge — Numerical Reliability Report</title>
<style>
:root{{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#0f172a;background:#f1f5f9}}
*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1100px;margin:auto;padding:36px 22px 80px}}
header{{background:#0f172a;color:white;padding:36px;border-radius:16px;margin-bottom:24px}}
h1{{font-size:2rem;margin:0 0 10px}}h2{{margin-top:0;font-size:1.35rem}}
h3{{font-size:1rem}}p,li{{line-height:1.6}}.muted{{color:#64748b}}header p{{color:#cbd5e1}}
section{{background:white;padding:26px;margin:22px 0;border-radius:14px;box-shadow:0 1px 5px #0f172a10}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.metric{{padding:16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px}}
.metric strong{{font-size:1.45rem;display:block;margin-top:6px}}.metric span{{font-size:.85rem;color:#475569}}
.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:.9rem}}
th,td{{padding:11px 12px;text-align:left;border-bottom:1px solid #e2e8f0}}th{{background:#f8fafc;white-space:nowrap}}
figure{{margin:24px 0}}figcaption{{font-size:.85rem;color:#475569;margin-bottom:8px}}
svg{{width:100%;height:auto;display:block}}.note{{border-left:4px solid #64748b;padding:12px 16px;background:#f8fafc}}
@media print{{body{{background:white}}main{{padding:0}}section{{box-shadow:none;border:1px solid #e2e8f0;break-inside:avoid}}header{{border-radius:0}}}}
</style></head><body><main>
<header><h1>ContactForge</h1><p>Numerical convergence and controlled-indentation experiment report</p></header>
<section><h2>Executive summary</h2>
<div class='metrics'>
<div class='metric'><span>Stage reference timestep</span><strong>{fmt(stage['reference_timestep_ms'], 3)} ms</strong></div>
<div class='metric'><span>Indentation reference timestep</span><strong>{fmt(reliability['reference_timestep_ms'], 3)} ms</strong></div>
<div class='metric'><span>Sampled peak contact force</span><strong>{peak_force:.3f} N</strong></div>
<div class='metric'><span>Final sample compression</span><strong>{final_compression:.3f} mm</strong></div>
<div class='metric'><span>Indentation samples</span><strong>{count}</strong></div>
</div>
<p class='note'>Reference values are numerical baselines, not ground truth. These results do not establish physical accuracy, material properties, or experimental validation.</p></section>
<section><h2>Experiment configuration</h2>{configuration}
<p>Modeled sample stiffness: {fmt(reliability['model_stiffness_N_per_m'], 2)} N/m. Indentation force-series timestep: {fmt(number(force['timestep_s'], 'timestep_s') * 1000, 3)} ms.</p></section>
<section><h2>Stage trajectory convergence</h2>
{table(('Timestep (ms)', 'Mean position error (mm)', 'Peak position error (mm)', 'Peak orientation error (deg)', 'Contact pair mismatch (%)'), stage_rows)}
<figure><figcaption>Peak position difference versus numerical reference, by timestep</figcaption>{line_chart(stage_plot, 'Timestep (ms)', 'Peak position difference (mm)')}</figure></section>
<section><h2>Indentation reliability</h2>
{table(('Timestep (ms)', 'Peak contact-force difference (N)', 'Estimated stiffness (N/m)', 'Stiffness error (%)', 'Peak force-balance residual (N)'), reliable_rows)}
<figure><figcaption>Maximum sampled absolute contact-force difference versus numerical reference (excluding reference itself)</figcaption>{line_chart(reliable_plot, 'Timestep (ms)', 'Peak force difference (N)', '#dc2626')}</figure>
<p class='note'>Force differences may be dominated by impact onset falling between output samples; a large maximum difference does not necessarily indicate the same error in impulse, settled force, or stiffness. A small force-balance residual checks internal equation consistency, not experimental accuracy.</p></section>
<section><h2>Controlled-indentation measurements</h2>
<figure><figcaption>Contact normal force over time</figcaption>{line_chart(times_force, 'Time (s)', 'Contact normal force (N)', '#0f766e')}</figure>
<figure><figcaption>Specimen compression over time</figcaption>{line_chart(times_compression, 'Time (s)', 'Compression (mm)', '#7c3aed')}</figure>
<figure><figcaption>Contact force versus specimen compression, in time order (dynamic effects may produce a loop)</figcaption>{line_chart(force_curve, 'Compression (mm)', 'Contact normal force (N)', '#b45309')}</figure>
</section>
<section><h2>Analyzer limitations</h2>{limitations_section('Stage convergence', stage)}{limitations_section('Indentation reliability', reliability)}
<p class='muted'>Generated offline from ContactForge JSON results. Every visual is embedded in this HTML file.</p></section>
</main></body></html>"""


def generate_report(stage_path, reliability_path, force_path, output_path):
    stage = load_report(stage_path, ("scene", "reference_timestep_ms", "duration_s", "sample_interval_s", "comparisons"))
    reliability = load_report(reliability_path, ("scene", "reference_timestep_ms", "duration_s", "sample_interval_s", "model_stiffness_N_per_m", "comparisons"))
    force = load_report(force_path, ("scene", "duration_s", "sample_interval_s", "timestep_s", "measurements"))
    content = render(stage, reliability, force)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a self-contained ContactForge HTML report")
    parser.add_argument("--stage", required=True, help="Stage convergence JSON")
    parser.add_argument("--reliability", required=True, help="Indentation reliability JSON")
    parser.add_argument("--indentation", required=True, help="Indentation force JSON")
    parser.add_argument("--output", required=True, help="Destination HTML file")
    args = parser.parse_args(argv)
    try:
        destination = generate_report(args.stage, args.reliability, args.indentation, args.output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"ContactForge report written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
