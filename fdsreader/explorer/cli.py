"""Command line front end.

A one-shot renderer: it prints what a simulation contains, or draws one slice time step
or a set of curves, as text. Needs nothing but numpy, so it works over SSH on a machine
that has no plotting stack and no browser.

    fdsreader-explorer-cli CASE                       what is in it
    fdsreader-explorer-cli CASE --slice 0 --time 4.4  draw a slice
    fdsreader-explorer-cli CASE --curve T_1.0 --curve HRR
"""

import argparse
import json
import sys

import numpy as np

import fdsreader
from fdsreader import settings

from .data import build_series, list_devices, quantity_of, unit_of
from .fields import fields_of
from .render import RAMP, series_lines, slice_lines
from .state import ExplorerState

PROG = "fdsreader-explorer-cli"


def _load(path, caching=False):
    settings.ENABLE_CACHING = caching
    return fdsreader.Simulation(str(path))


def _state_from(args, fields):
    """Build the shared state from the command line arguments."""
    state = ExplorerState(scale_mode="step")
    if args.slice is not None:
        state.field_index = args.slice
    if args.scale in ("global", "step"):
        state.scale_mode = args.scale
    else:
        try:
            low, high = (float(part) for part in args.scale.split(","))
        except ValueError:
            raise SystemExit(f"{PROG}: --scale wants 'global', 'step' or 'LOW,HIGH', not {args.scale!r}")
        if not low < high:
            raise SystemExit(f"{PROG}: --scale LOW must be below HIGH")
        state.scale_mode, state.manual_limits = "manual", (low, high)
    if args.time is not None and fields:
        state.seek(fields, args.time)
    return state


def overview(sim):
    """What the simulation holds, as text lines."""
    devices = list_devices(sim)
    series = build_series(sim)
    n_hrr = sum(1 for s in series if s["source"] == "hrr")

    starts = [s["times"][0] for s in series if len(s["times"])]
    ends = [s["times"][-1] for s in series if len(s["times"])]

    lines = [f"{sim.chid}   {sim.root_path}", ""]
    lines.append(f"  meshes            {len(sim.meshes)}")
    lines.append(f"  devices (DEVC)    {len(devices)}")
    lines.append(f"  HRR quantities    {n_hrr}")
    lines.append(f"  slices (SLCF)     {len(fields_of(sim))}")
    if starts:
        lines.append(f"  time              {min(starts):g} … {max(ends):g} s")

    groups = {}
    for device in devices:
        groups.setdefault(quantity_of(device) or "—", []).append(unit_of(device))
    if groups:
        lines += ["", "  quantity                        #  unit"]
        for name, units in sorted(groups.items()):
            shown = ", ".join(sorted(set(units) - {""})) or "—"
            lines.append(f"  {name:<28} {len(units):>4}  {shown}")
    return lines


def list_contents(sim):
    """Every slice and every curve, with the names the other options take."""
    fields = fields_of(sim)
    lines = ["slices:"]
    if not fields:
        lines.append("  (none)")
    for i, field in enumerate(fields):
        lines.append(
            f"  --slice {i:<3} {field.label}   {len(field.times)} steps, t = {field.times[0]:g} … {field.times[-1]:g} s"
        )

    lines.append("")
    lines.append("curves:")
    series = build_series(sim)
    if not series:
        lines.append("  (none)")
    for entry in series:
        lines.append(f"  --curve {entry['name']:<22} {entry['label']}")
    return lines


def render_slice(fields, state, height, width, ramp):
    field = state.field(fields)
    if field is None:
        raise SystemExit(f"{PROG}: this simulation has no slice output")

    timestep = state.clamp(fields)
    frame = field.frame(timestep)
    vmin, vmax, warning = state.limits(field, frame=frame)

    lines = [f"{field.label}   ·   t = {field.times[timestep]:.3f} s (step {timestep + 1} of {len(field.times)})", ""]
    if warning:
        lines.append(f"  note: {warning}")
    lines += slice_lines(field, timestep, vmin, vmax, height=height, width=width, ramp=ramp, frame=frame)
    return lines


def save_figure(fields, state, series, path, height, width):
    """Write the current view to an image or an animation, using matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise SystemExit(f'{PROG}: --save needs matplotlib: pip install "fdsreader[notebook]"')
    from .plots import plot_series, plot_slice

    field = state.field(fields)
    animated = path.suffix.lower() in (".gif", ".mp4")

    if animated:
        if field is None:
            raise SystemExit(f"{PROG}: --save {path.suffix} needs a slice to animate")
        from matplotlib import animation

        steps = range(0, len(field.times), max(1, len(field.times) // 120))
        fig, ax = plt.subplots()

        def draw(step):
            ax.clear()
            frame = field.frame(step)
            vmin, vmax, _ = state.limits(field, frame=frame)
            plot_slice(
                field,
                step,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cmap=state.colormap(field),
                render=state.render,
                n_levels=state.n_levels,
                frame=frame,
            )

        writer = "pillow" if path.suffix.lower() == ".gif" else "ffmpeg"
        anim = animation.FuncAnimation(fig, draw, frames=list(steps), interval=80)
        try:
            anim.save(str(path), writer=writer)
        except Exception as exc:
            raise SystemExit(f"{PROG}: could not write {path}: {exc}")
        plt.close(fig)
        return f"wrote {path} ({len(list(steps))} frames)"

    panels = (1 if field is None else 1) + (1 if series else 0)
    fig, axes = plt.subplots(1, max(1, panels), figsize=(7 * max(1, panels), 5.5))
    axes = np.atleast_1d(axes)
    at = 0
    if field is not None:
        timestep = state.clamp(fields)
        frame = field.frame(timestep)
        vmin, vmax, _ = state.limits(field, frame=frame)
        plot_slice(
            field,
            timestep,
            ax=axes[at],
            vmin=vmin,
            vmax=vmax,
            cmap=state.colormap(field),
            render=state.render,
            n_levels=state.n_levels,
            frame=frame,
        )
        at += 1
    if series:
        plot_series(series, ax=axes[at], cursor_time=state.time(fields) if field is not None else None)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    return f"wrote {path}"


def as_json(sim, fields, state, series):
    """The same information as the text output, for a script to consume."""
    devices = list_devices(sim)
    all_series = build_series(sim)
    payload = {
        "chid": sim.chid,
        "path": sim.root_path,
        "meshes": len(sim.meshes),
        "devices": len(devices),
        "hrr_quantities": sum(1 for e in all_series if e["source"] == "hrr"),
        "slices": [
            {
                "index": i,
                "label": f.label,
                "quantity": f.quantity,
                "unit": f.unit,
                "shape": list(f.shape),
                "axes": list(f.axes),
                "n_timesteps": len(f.times),
                "time": [float(f.times[0]), float(f.times[-1])],
                "range": [float(f.value_range()[0]), float(f.value_range()[1])],
            }
            for i, f in enumerate(fields)
        ],
        "curves": [
            {
                "name": e["name"],
                "label": e["label"],
                "quantity": e["quantity"],
                "unit": e["unit"],
                "source": e["source"],
                "n_samples": int(len(e["values"])),
                "time": [float(e["times"][0]), float(e["times"][-1])],
                "min": float(np.nanmin(e["values"])),
                "max": float(np.nanmax(e["values"])),
                "final": float(e["values"][-1]),
            }
            for e in all_series
        ],
    }
    if series:
        payload["selected"] = [
            {"name": e["name"], "times": [float(v) for v in e["times"]], "values": [float(v) for v in e["values"]]}
            for e in series
        ]
    return payload


def resolve_curves(sim, names):
    """Look up the requested curves by name."""
    by_name = {}
    for entry in build_series(sim):
        by_name.setdefault(entry["name"], entry)
    chosen = []
    for name in names:
        if name not in by_name:
            raise SystemExit(f"{PROG}: no curve called {name!r} — try --list")
        chosen.append(by_name[name])
    return chosen


def build_parser():
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Look at FDS output in a terminal.",
        epilog="With no options it prints what the simulation contains.",
    )
    parser.add_argument("path", help="simulation directory, or the .smv file")
    parser.add_argument("--list", action="store_true", help="list every slice and curve with the name to pass back in")
    parser.add_argument("--slice", type=int, metavar="N", help="draw slice N")
    parser.add_argument(
        "--time", type=float, metavar="SECONDS", help="time step to draw, the nearest one is used (default: first)"
    )
    parser.add_argument(
        "--scale", default="step", metavar="MODE", help="colour limits: 'global', 'step' (default) or 'LOW,HIGH'"
    )
    parser.add_argument(
        "--curve",
        action="append",
        default=[],
        metavar="NAME",
        help="draw this device or HRR quantity; repeat for several",
    )
    parser.add_argument(
        "--height", type=int, default=26, metavar="ROWS", help="character rows for the slice (default: 26)"
    )
    parser.add_argument(
        "--width", type=int, default=None, metavar="COLS", help="character columns (default: fit the terminal)"
    )
    parser.add_argument("--ramp", default=None, metavar="CHARS", help="characters from empty to full, e.g. ' .oO@'")
    parser.add_argument("--json", action="store_true", help="print the same information as JSON, for scripting")
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="write the current view to an image (.png/.pdf/.svg) or an "
        "animation (.gif/.mp4) instead of drawing it as text; "
        "needs matplotlib",
    )
    parser.add_argument(
        "--render",
        default="image",
        choices=("image", "filled", "lines"),
        help="how --save draws a slice (default: image)",
    )
    parser.add_argument(
        "--caching",
        action="store_true",
        help="let fdsreader cache the parsed simulation as a .pickle "
        "next to the data; off by default so that read-only "
        "directories work",
    )
    return parser


def main(argv=None):
    """Entry point. Returns a process exit status."""
    args = build_parser().parse_args(argv)

    try:
        sim = _load(args.path, caching=args.caching)
    except Exception as exc:
        print(f"{PROG}: could not load {args.path}: {exc}", file=sys.stderr)
        return 1

    fields = fields_of(sim)
    if args.slice is not None and not 0 <= args.slice < len(fields):
        print(
            f"{PROG}: --slice {args.slice} out of range (0 … {len(fields) - 1})"
            if fields
            else f"{PROG}: this simulation has no slice output",
            file=sys.stderr,
        )
        return 1

    state = _state_from(args, fields)
    state.render = args.render
    series = resolve_curves(sim, args.curve)

    if args.json:
        json.dump(as_json(sim, fields, state, series), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.save:
        from pathlib import Path

        print(save_figure(fields, state, series, Path(args.save), args.height, args.width))
        return 0

    lines = []
    if args.list:
        lines += list_contents(sim)
    elif args.slice is None and not args.curve:
        lines += overview(sim)

    if args.slice is not None:
        lines += render_slice(fields, state, args.height, args.width, args.ramp or RAMP)
    if series:
        if lines:
            lines.append("")
        lines += series_lines(series, height=max(6, args.height // 2), width=args.width, cursor_time=args.time)

    print("\n".join(lines))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
