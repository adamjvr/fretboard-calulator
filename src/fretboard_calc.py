#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fretboard_calc.py
=================

Professional-grade fretboard calculator & CAD generator
------------------------------------------------------
This script computes 12-TET fret locations for single-scale and multiscale
(fanned-fret) instruments, and emits:
  • Markdown (tables to stdout),
  • CSV (positions + spacings),
  • CAD-friendly JSON (full geometry),
  • SVG or DXF drawings (frets, strings, outline, nut, bridge, slot offsets).

✨ Now supports:
  ✅ Datum angle rotation (rotate entire fretboard around neutral fret origin)
  ✅ Multiscale (linear or exponential taper)
  ✅ Parametric outline, kerf offsets, and CAD export
  ✅ Per-string spacing (uniform, CSV list, or file)

COORDINATES & CONVENTIONS
-------------------------
• 12-TET along-string distances (from nut) use the closed form:
    pos(n) = L - L / (2 ** (n/12))
  where L is per-string scale length, n is fret number.

• Multiscale = a scale per string. We interpolate from bass to treble either:
    - LINEAR:   t = s/(S-1)
    - EXPONENTIAL: t = (s/(S-1)) ** gamma
  and blend L = L_bass*(1-t) + L_treble*t

• Neutral-fret-relative X:
    x_s(n) = pos_s(n) - pos_s(N)    (N is --neutral-fret)
  String Y positions are built from spacing (uniform or list/file).

• Datum rotation (--datum-angle deg) is applied AFTER building neutral-fret
  geometry, rotating everything (frets, strings, outline, nut, bridge, slots)
  around the origin by theta = radians(datum_angle).

• Units: choose --unit {in, mm}. Any numeric CLI value may include "in"/"mm"
  suffix; if omitted it is interpreted in --unit.

USAGE EXAMPLES
--------------
# Single-scale → Markdown
python fretboard_calc.py --frets 22 --strings 6 --scale 25.5in --unit in

# Multiscale → JSON + SVG (with strings), neutral-fret=7, rotated 5°
python fretboard_calc.py --frets 24 --strings 8   --bass-scale 27in --treble-scale 25.5in   --neutral-fret 7 --string-spacing 0.35in   --datum-angle 5.0   --json board.json   --svg board.svg --draw-strings

# DXF with kerf-offset slot toolpaths
python fretboard_calc.py --frets 24 --strings 6 --scale 25.5in   --nut-width 1.70in --bridge-width 2.20in   --slot-kerf 0.023in --emit-slot-offsets   --dxf board.dxf
"""

# NOTE: __future__ imports must appear at module top (after docstring/comments) or Python raises a SyntaxError.
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from typing import List, Tuple, Optional, Iterable, Dict

IN_PER_MM = 1.0 / 25.4
MM_PER_IN = 25.4


def parse_length_with_unit(text: str, default_unit: str) -> Tuple[float, str]:
    t = text.strip().lower()
    inch_suffixes = ("in", "inch", "inches")
    mm_suffixes = ("mm",)
    for suf in inch_suffixes:
        if t.endswith(suf):
            return float(t[: -len(suf)].strip()), "in"
    for suf in mm_suffixes:
        if t.endswith(suf):
            return float(t[: -len(suf)].strip()), "mm"
    return float(t), default_unit


def to_unit(value: float, from_unit: str, to_unit_: str) -> float:
    if from_unit == to_unit_:
        return value
    if from_unit == "in" and to_unit_ == "mm":
        return value * MM_PER_IN
    if from_unit == "mm" and to_unit_ == "in":
        return value * IN_PER_MM
    raise ValueError(f"Unsupported unit conversion: {from_unit} -> {to_unit_}")


def parse_len_arg(text: str, unit: str) -> float:
    v, u = parse_length_with_unit(text, unit)
    return to_unit(v, u, unit)


def fret_positions_along_string(scale: float, frets: int) -> List[float]:
    positions = [0.0]
    for n in range(1, frets + 1):
        positions.append(scale - scale / (2.0 ** (n / 12.0)))
    return positions


def fret_spacings_from_positions(positions: List[float]) -> List[float]:
    out = [0.0]
    for i in range(1, len(positions)):
        out.append(positions[i] - positions[i - 1])
    return out


def interpolate_scales_for_strings(
    strings: int,
    L_bass: float,
    L_treble: float,
    map_mode: str = "linear",
    gamma: float = 1.0,
) -> List[float]:
    if strings == 1:
        return [L_bass]
    out: List[float] = []
    for s in range(strings):
        t = s / (strings - 1)
        if map_mode == "exp":
            t = (t**gamma) if strings > 1 else 0.0
        L = L_bass * (1.0 - t) + L_treble * t
        out.append(L)
    return out


def parse_spacing_list(text: str, unit: str) -> List[float]:
    parts = [p for p in (x.strip() for x in text.split(",")) if p]
    return [parse_len_arg(p, unit) for p in parts]


def parse_spacing_file(path: str, unit: str) -> List[float]:
    gaps: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            gaps.append(parse_len_arg(s, unit))
    return gaps


def build_string_y_positions(
    strings: int, uniform_spacing: Optional[float], gap_list: Optional[List[float]]
) -> List[float]:
    if strings < 1:
        return []
    if strings == 1:
        return [0.0]
    if gap_list is not None:
        if len(gap_list) != strings - 1:
            raise SystemExit(
                f"--string-spacing-list/file must specify exactly {strings - 1} gaps."
            )
        gaps = gap_list
    else:
        if uniform_spacing is None:
            raise SystemExit(
                "Internal: need either uniform spacing value or a gap list."
            )
        gaps = [uniform_spacing] * (strings - 1)

    y = [0.0]
    acc = 0.0
    for g in gaps:
        acc += g
        y.append(acc)
    return y


def calculate_fret_geometry(
    positions: List[List[float]], y_positions: List[float], neutral_fret: int
) -> List[dict]:
    S = len(positions)
    F = len(positions[0]) - 1
    geometry: List[dict] = []
    offs = [positions[s][neutral_fret] for s in range(S)]
    for n in range(F + 1):
        coords = []
        for s in range(S):
            x = positions[s][n] - offs[s]
            y = y_positions[s]
            coords.append({"string": s + 1, "x": round(x, 6), "y": round(y, 6)})
        x0, y0 = coords[0]["x"], coords[0]["y"]
        x1, y1 = coords[-1]["x"], coords[-1]["y"]
        dx, dy = (x1 - x0), (y1 - y0)
        angle_deg = math.degrees(math.atan2(dx, dy)) if abs(dy) > 1e-12 else 0.0
        geometry.append(
            {"fret": n, "angle_deg": round(angle_deg, 6), "coordinates": coords}
        )
    return geometry


def outline_corners(
    scales: List[float],
    positions: List[List[float]],
    neutral_fret: int,
    nut_width: float,
    bridge_width: float,
    y_positions: List[float],
) -> Dict[str, Tuple[float, float]]:
    # FIX: center outline vertically on the actual string span midline
    s_bass = 0
    s_treb = len(scales) - 1
    x_nut_bass = positions[s_bass][0] - positions[s_bass][neutral_fret]
    x_nut_treb = positions[s_treb][0] - positions[s_treb][neutral_fret]
    x_bridge_bass = scales[s_bass] - positions[s_bass][neutral_fret]
    x_bridge_treb = scales[s_treb] - positions[s_treb][neutral_fret]
    y_center = (min(y_positions) + max(y_positions)) / 2.0
    BN = (x_nut_bass, y_center - nut_width / 2.0)
    TN = (x_nut_treb, y_center + nut_width / 2.0)
    BB = (x_bridge_bass, y_center - bridge_width / 2.0)
    TB = (x_bridge_treb, y_center + bridge_width / 2.0)
    return {"BN": BN, "TN": TN, "TB": TB, "BB": BB}


def _extend_line_to_y_span(
    x0: float, y0: float, x1: float, y1: float, y_min: float, y_max: float
) -> Tuple[float, float, float, float]:
    if abs(y1 - y0) < 1e-12:
        return (x0, y_min, x0, y_max)

    def x_of_y(y: float) -> float:
        t = (y - y0) / (y1 - y0)
        return x0 + (x1 - x0) * t

    return (x_of_y(y_min), y_min, x_of_y(y_max), y_max)


def collect_fret_segments(
    geometry: List[dict], y_min: float, y_max: float
) -> List[Tuple[float, float, float, float]]:
    segs: List[Tuple[float, float, float, float]] = []
    for fret in geometry:
        (x0, y0) = (fret["coordinates"][0]["x"], fret["coordinates"][0]["y"])
        (x1, y1) = (fret["coordinates"][-1]["x"], fret["coordinates"][-1]["y"])
        segs.append(_extend_line_to_y_span(x0, y0, x1, y1, y_min, y_max))
    return segs


def offset_line_normal(
    x1: float, y1: float, x2: float, y2: float, offset: float
) -> Tuple[float, float, float, float]:
    dx, dy = (x2 - x1), (y2 - y1)
    L = math.hypot(dx, dy)
    if L < 1e-12:
        return (x1, y1, x2, y2)
    nx, ny = (-dy / L, dx / L)
    return (x1 + offset * nx, y1 + offset * ny, x2 + offset * nx, y2 + offset * ny)


def collect_slot_offsets(
    fret_segments: List[Tuple[float, float, float, float]], kerf: float
) -> Tuple[
    List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]
]:
    half = kerf / 2.0
    left, right = [], []
    for x1, y1, x2, y2 in fret_segments:
        left.append(offset_line_normal(x1, y1, x2, y2, -half))
        right.append(offset_line_normal(x1, y1, x2, y2, +half))
    return left, right


def rotate_point(x: float, y: float, theta_rad: float) -> Tuple[float, float]:
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    X = x * cos_t - y * sin_t
    Y = x * sin_t + y * cos_t
    return X, Y


def rotate_segment(
    seg: Tuple[float, float, float, float], theta_rad: float
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = seg
    X1, Y1 = rotate_point(x1, y1, theta_rad)
    X2, Y2 = rotate_point(x2, y2, theta_rad)
    return (X1, Y1, X2, Y2)


def rotate_geometry(geometry: List[dict], theta_rad: float) -> List[dict]:
    rotated = []
    for fret in geometry:
        new_coords = []
        for c in fret["coordinates"]:
            X, Y = rotate_point(c["x"], c["y"], theta_rad)
            new_coords.append(
                {"string": c["string"], "x": round(X, 6), "y": round(Y, 6)}
            )
        rotated.append(
            {
                "fret": fret["fret"],
                "angle_deg": fret["angle_deg"] + math.degrees(theta_rad),
                "coordinates": new_coords,
            }
        )
    return rotated


def fmt_val(x: float, unit: str, decimals: int) -> str:
    return f"{x:.{decimals}f}"


def normalize_scales_for_bridge_angle(
    Lb: float,
    Lt: float,
    neutral_fret: int,
    bridge_width: float,
    target_bridge_angle_deg: float,
    mode: str = "treble",
) -> Tuple[float, float]:
    """
    Adjust scale lengths so that the bridge edge tilts *relative* to the current fan angle.

    This version interprets --target-bridge-angle as a RELATIVE adjustment (Δθ)
    to the current fan slope rather than an absolute geometry angle vs. X-axis.
    That prevents non-physical scale lengths for small angles and matches
    what luthiers intuitively mean when specifying bridge tilt.

    Args:
        Lb: current bass string scale length
        Lt: current treble string scale length
        neutral_fret: integer index of neutral fret
        bridge_width: total width at the bridge (edge-to-edge)
        target_bridge_angle_deg: desired *relative* bridge tilt in degrees
        mode: "treble" = adjust treble scale only, "both" = adjust both equally

    Returns:
        (Lb_new, Lt_new): new adjusted scale lengths
    """

    if bridge_width <= 0:
        raise SystemExit("Bridge width must be > 0 for angle normalization.")

    # Neutral-fret scaling factor
    k = 2.0 ** (-(neutral_fret / 12.0))

    # Compute current slope (Δy/Δx) and current bridge angle
    current_dx = k * (Lt - Lb)
    if abs(current_dx) < 1e-9:
        raise SystemExit("Scale difference too small for normalization.")

    current_angle = math.degrees(math.atan2(bridge_width, current_dx))
    target_angle = current_angle + target_bridge_angle_deg
    theta_new = math.radians(target_angle)

    # Compute new Δx for the desired bridge slope
    new_dx = bridge_width / math.tan(theta_new)
    delta = new_dx / k

    if mode == "treble":
        Lt_new = Lb + delta
        Lb_new = Lb
    else:
        mean = 0.5 * (Lb + Lt)
        Lb_new = mean - 0.5 * delta
        Lt_new = mean + 0.5 * delta

    if Lb_new <= 0 or Lt_new <= 0:
        raise SystemExit(
            "Normalization produced non-physical scale length(s). Adjust parameters."
        )

    return (Lb_new, Lt_new)


def make_markdown_table(
    data: List[List[float]], unit: str, decimals: int, title: str
) -> str:
    S = len(data)
    F = len(data[0]) - 1
    header = ["Fret"] + [f"String {i + 1} ({unit})" for i in range(S)]
    sep = ["---"] * len(header)
    lines = [
        f"## {title}",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for n in range(F + 1):
        row = [str(n)] + [fmt_val(data[s][n], unit, decimals) for s in range(S)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_csv_file(
    filename: str,
    positions: List[List[float]],
    spacings: List[List[float]],
    unit: str,
    decimals: int,
):
    S = len(positions)
    F = len(positions[0]) - 1
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"Nut-to-Fret Positions ({unit})"])
        w.writerow(["Fret"] + [f"String {i + 1}" for i in range(S)])
        for n in range(F + 1):
            w.writerow([n] + [f"{positions[s][n]:.{decimals}f}" for s in range(S)])
        w.writerow([])
        w.writerow([f"Per-Fret Spacings ({unit})"])
        w.writerow(["Fret"] + [f"String {i + 1}" for i in range(S)])
        for n in range(F + 1):
            w.writerow([n] + [f"{spacings[s][n]:.{decimals}f}" for s in range(S)])


def write_json_file(
    filename: str,
    positions: List[List[float]],
    spacings: List[List[float]],
    scales: List[float],
    unit: str,
    neutral_fret: int,
    y_positions: List[float],
    nut_width: float,
    bridge_width: float,
    geometry: List[dict],
    datum_angle_deg: float,
):
    data = {
        "unit": unit,
        "strings": len(positions),
        "frets": len(positions[0]) - 1,
        "scales": scales,
        "neutral_fret": neutral_fret,
        "datum_angle_deg": datum_angle_deg,
        "string_y_positions": y_positions,
        "nut_width": nut_width,
        "bridge_width": bridge_width,
        "fret_positions": positions,
        "fret_spacings": spacings,
        "fret_geometry": geometry,
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def svg_write(
    filename: str,
    unit: str,
    stroke_width: float,
    frets: List[Tuple[float, float, float, float]],
    slots_left: List[Tuple[float, float, float, float]],
    slots_right: List[Tuple[float, float, float, float]],
    strings: List[Tuple[float, float, float, float]],
    outline: List[Tuple[float, float, float, float]],
    nut: Tuple[float, float, float, float],
    bridge: Tuple[float, float, float, float],
    pad: float = 0.0,
):
    def add_bounds(
        seglist: Iterable[Tuple[float, float, float, float]],
        xs: List[float],
        ys: List[float],
    ):
        for x1, y1, x2, y2 in seglist:
            xs.extend([x1, x2])
            ys.extend([y1, y2])

    xs: List[float] = []
    ys: List[float] = []
    add_bounds(frets, xs, ys)
    add_bounds(slots_left, xs, ys)
    add_bounds(slots_right, xs, ys)
    add_bounds(strings, xs, ys)
    add_bounds(outline, xs, ys)
    xs.extend([nut[0], nut[2], bridge[0], bridge[2]])
    ys.extend([nut[1], nut[3], bridge[1], bridge[3]])
    if not xs:
        xs, ys = [0.0, 1.0], [0.0, 1.0]
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    width, height = (xmax - xmin), (ymax - ymin)

    def line_el(x1, y1, x2, y2) -> str:
        return f'<line x1="{x1:.6f}" y1="{y1:.6f}" x2="{x2:.6f}" y2="{y2:.6f}" />'

    with open(filename, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin:.6f} {ymin:.6f} {width:.6f} {height:.6f}">\n'
        )
        f.write(f"  <desc>Units: {unit}. Datum rotation supported.</desc>\n")
        f.write("  <style>\n")
        f.write(
            f"    .FRETS   {{ stroke:black; fill:none; stroke-width:{stroke_width:.6f}; }}\n"
        )
        f.write(
            f"    .SLOTS   {{ stroke:#444;  fill:none; stroke-width:{max(stroke_width * 0.6, 0.001):.6f}; }}\n"
        )
        f.write(
            f"    .STRINGS {{ stroke:#999;  fill:none; stroke-width:{max(stroke_width * 0.5, 0.001):.6f}; }}\n"
        )
        f.write(
            f"    .OUTLINE {{ stroke:#07c;  fill:none; stroke-width:{max(stroke_width * 0.8, 0.001):.6f}; }}\n"
        )
        f.write(
            f"    .NUT, .BRIDGE {{ stroke:#c07; fill:none; stroke-width:{max(stroke_width * 0.8, 0.001):.6f}; }}\n"
        )
        f.write("  </style>\n")

        def write_seg(
            gid: str, cls: str, segs: List[Tuple[float, float, float, float]]
        ):
            f.write(f'  <g id="{gid}" class="{cls}">\n')
            for x1, y1, x2, y2 in segs:
                f.write("    " + line_el(x1, y1, x2, y2) + "\n")
            f.write("  </g>\n")

        write_seg("OUTLINE", "OUTLINE", outline)
        f.write('  <g id="NUT" class="NUT">\n')
        f.write("    " + line_el(*nut) + "\n")
        f.write("  </g>\n")
        f.write('  <g id="BRIDGE" class="BRIDGE">\n')
        f.write("    " + line_el(*bridge) + "\n")
        f.write("  </g>\n")
        write_seg("FRETS", "FRETS", frets)
        write_seg("SLOTS", "SLOTS", slots_left + slots_right)
        write_seg("STRINGS", "STRINGS", strings)
        f.write("</svg>\n")


def dxf_write(
    filename: str,
    frets: List[Tuple[float, float, float, float]],
    slots_left: List[Tuple[float, float, float, float]],
    slots_right: List[Tuple[float, float, float, float]],
    strings: List[Tuple[float, float, float, float]],
    outline: List[Tuple[float, float, float, float]],
    nut: Tuple[float, float, float, float],
    bridge: Tuple[float, float, float, float],
):
    def line_entity(x1, y1, x2, y2, layer) -> str:
        return (
            "0\nLINE\n"
            f"8\n{layer}\n"
            f"10\n{x1:.6f}\n20\n{y1:.6f}\n11\n{x2:.6f}\n21\n{y2:.6f}\n"
        )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("0\nSECTION\n2\nENTITIES\n")
        for x1, y1, x2, y2 in outline:
            f.write(line_entity(x1, y1, x2, y2, "OUTLINE"))
        f.write(line_entity(*nut, "NUT"))
        f.write(line_entity(*bridge, "BRIDGE"))
        for x1, y1, x2, y2 in frets:
            f.write(line_entity(x1, y1, x2, y2, "FRETS"))
        for x1, y1, x2, y2 in slots_left:
            f.write(line_entity(x1, y1, x2, y2, "SLOTS"))
        for x1, y1, x2, y2 in slots_right:
            f.write(line_entity(x1, y1, x2, y2, "SLOTS"))
        for x1, y1, x2, y2 in strings:
            f.write(line_entity(x1, y1, x2, y2, "STRINGS"))
        f.write("0\nENDSEC\n0\nEOF\n")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Full-featured fretboard calculator + CAD (SVG/DXF) generator"
    )
    p.add_argument("--frets", type=int, required=True, help="Number of frets (>=1).")
    p.add_argument(
        "--strings", type=int, required=True, help="Number of strings (>=1)."
    )
    p.add_argument(
        "--scale",
        type=str,
        default=None,
        help='Single-scale, e.g. "25.5in" or "648mm".',
    )
    p.add_argument(
        "--bass-scale",
        type=str,
        default=None,
        help='Multiscale: bass side, e.g. "27in".',
    )
    p.add_argument(
        "--treble-scale",
        type=str,
        default=None,
        help='Multiscale: treble side, e.g. "25.5in".',
    )
    p.add_argument(
        "--scale-map",
        choices=["linear", "exp"],
        default="linear",
        help="Per-string scale interpolation: linear or exponential.",
    )
    p.add_argument(
        "--scale-gamma",
        type=float,
        default=1.0,
        help="Gamma for --scale-map exp (t -> t**gamma). 1.0 = linear.",
    )
    p.add_argument(
        "--unit", choices=["in", "mm"], default="in", help="Computation/output unit."
    )
    p.add_argument("--decimals", type=int, default=3, help="Decimal places for tables.")
    p.add_argument(
        "--neutral-fret",
        type=int,
        default=0,
        help="Neutral fret index (0..F). Default 0 (nut).",
    )
    p.add_argument(
        "--string-spacing",
        type=str,
        default=None,
        help='Uniform per-adjacent-string gap, e.g. "0.35in" or "9mm".',
    )
    p.add_argument(
        "--string-spacing-list",
        type=str,
        default=None,
        help='CSV list of S-1 gaps, e.g. "0.35in,0.35in,0.36in,...".',
    )
    p.add_argument(
        "--string-spacing-file",
        type=str,
        default=None,
        help="Path to file containing S-1 gaps, one per line.",
    )
    p.add_argument(
        "--nut-width",
        type=str,
        default="1.70in",
        help='Total board width at nut, e.g. "1.70in".',
    )
    p.add_argument(
        "--bridge-width",
        type=str,
        default="2.20in",
        help='Total board width at bridge, e.g. "2.20in".',
    )
    p.add_argument("--markdown", action="store_true", help="Force Markdown to stdout.")
    p.add_argument("--csv", type=str, help="Write CSV file.")
    p.add_argument("--json", type=str, help="Write JSON file.")
    p.add_argument("--svg", type=str, help="Write SVG drawing.")
    p.add_argument("--dxf", type=str, help="Write DXF drawing.")
    p.add_argument(
        "--draw-strings",
        action="store_true",
        help="Include string centerlines in drawings.",
    )
    p.add_argument(
        "--board-margin",
        type=str,
        default="0.10in",
        help="Extra vertical margin beyond bass/treble strings (adds to y-span).",
    )
    p.add_argument(
        "--board-margin-x",
        type=str,
        default=None,
        help="Horizontal margin for string lines (defaults to board-margin).",
    )
    p.add_argument(
        "--stroke",
        type=float,
        default=0.6,
        help="SVG stroke width in user units (same as --unit).",
    )
    p.add_argument(
        "--slot-kerf",
        type=str,
        default=None,
        help='Kerf width for slot toolpaths, e.g. "0.023in" (saw blade width).',
    )
    p.add_argument(
        "--emit-slot-offsets",
        action="store_true",
        help="If set and --slot-kerf provided, emit left/right slot lines on SLOTS layer.",
    )
    p.add_argument(
        "--datum-angle",
        type=float,
        default=0.0,
        help="Rotate entire fretboard around neutral fret origin by this angle (degrees).",
    )
    p.add_argument(
        "--target-bridge-angle",
        type=float,
        default=None,
        help="Target bridge *edge* angle in degrees (relative to +X).",
    )
    p.add_argument(
        "--normalize-scale",
        action="store_true",
        help="Enable scale normalization to satisfy --target-bridge-angle.",
    )
    p.add_argument(
        "--normalize-mode",
        choices=["treble", "both"],
        default="treble",
        help="When normalizing: adjust only treble scale (treble), or adjust both endpoints equally (both).",
    )

    return p


def validate_and_get_scales(args) -> List[float]:
    if args.bass_scale and args.treble_scale:
        Lb = parse_len_arg(args.bass_scale, args.unit)
        Lt = parse_len_arg(args.treble_scale, args.unit)
        if args.strings < 2:
            raise SystemExit("Multiscale requires --strings >= 2.")
        return interpolate_scales_for_strings(
            args.strings, Lb, Lt, args.scale_map, args.scale_gamma
        )
    if args.scale:
        L = parse_len_arg(args.scale, args.unit)
        return [L] * args.strings
    raise SystemExit(
        "Provide --scale (single) OR both --bass-scale and --treble-scale (multi)."
    )


def main():
    args = build_arg_parser().parse_args()
    if args.frets < 1:
        raise SystemExit("--frets must be >= 1.")
    if args.strings < 1:
        raise SystemExit("--strings must be >= 1.")
    if not (0 <= args.neutral_fret <= args.frets):
        raise SystemExit(f"--neutral-fret must be in [0..{args.frets}].")
    scales = validate_and_get_scales(args)
    uniform_spacing: Optional[float] = None
    gap_list: Optional[List[float]] = None
    if args.string_spacing_list and args.string_spacing_file:
        raise SystemExit(
            "Use either --string-spacing-list OR --string-spacing-file, not both."
        )
    if args.string_spacing_list:
        gap_list = parse_spacing_list(args.string_spacing_list, args.unit)
    elif args.string_spacing_file:
        gap_list = parse_spacing_file(args.string_spacing_file, args.unit)
    else:
        if args.string_spacing is None:
            uniform_spacing = parse_len_arg(
                "0.354in" if args.unit == "in" else "9mm", args.unit
            )
        else:
            uniform_spacing = parse_len_arg(args.string_spacing, args.unit)
    y_positions = build_string_y_positions(args.strings, uniform_spacing, gap_list)
    all_positions = [fret_positions_along_string(L, args.frets) for L in scales]
    all_spacings = [fret_spacings_from_positions(pos) for pos in all_positions]
    geometry = calculate_fret_geometry(all_positions, y_positions, args.neutral_fret)
    nut_width = parse_len_arg(args.nut_width, args.unit)
    bridge_width = parse_len_arg(args.bridge_width, args.unit)
    # ------------------------------------------------------
    # Optional bridge-angle normalization
    # ------------------------------------------------------
    if args.normalize_scale and (args.target_bridge_angle is not None):
        # Save current scales (from args or defaults)
        Lb_current = 0.0
        Lt_current = 0.0
        if args.bass_scale:
            Lb_current = parse_len_arg(args.bass_scale, args.unit)
        elif args.scale:
            Lb_current = parse_len_arg(args.scale, args.unit)
        if args.treble_scale:
            Lt_current = parse_len_arg(args.treble_scale, args.unit)
        elif args.scale:
            Lt_current = parse_len_arg(args.scale, args.unit)

        # Perform normalization
        (Lb_new, Lt_new) = normalize_scales_for_bridge_angle(
            Lb_current,
            Lt_current,
            args.neutral_fret,
            parse_len_arg(args.bridge_width, args.unit),
            args.target_bridge_angle,
            args.normalize_mode,
        )

        # ------------------------------------------------------
        # 🎯 Normalization Summary Printout
        # ------------------------------------------------------
        old_delta = abs(Lt_current - Lb_current)
        new_delta = abs(Lt_new - Lb_new)
        print("\n🎯 Bridge normalization summary")
        print(f"   Bass scale:   {Lb_current:.3f} → {Lb_new:.3f} {args.unit}")
        print(f"   Treble scale: {Lt_current:.3f} → {Lt_new:.3f} {args.unit}")
        print(f"   Fan delta:    {old_delta:.3f} → {new_delta:.3f} {args.unit}")
        print(f"   Δ angle:      +{args.target_bridge_angle:.3f}° (relative)\n")

        # Update current scales for downstream geometry
        Lb = Lb_new
        Lt = Lt_new
    else:
        # Use provided or default scales
        Lb = parse_len_arg(args.bass_scale or args.scale, args.unit)
        Lt = parse_len_arg(args.treble_scale or args.scale, args.unit)

    corners = outline_corners(
        scales, all_positions, args.neutral_fret, nut_width, bridge_width, y_positions
    )
    BN, TN, TB, BB = corners["BN"], corners["TN"], corners["TB"], corners["BB"]
    outline_segments = [
        (BN[0], BN[1], TN[0], TN[1]),
        (TN[0], TN[1], TB[0], TB[1]),
        (TB[0], TB[1], BB[0], BB[1]),
        (BB[0], BB[1], BN[0], BN[1]),
    ]
    nut_line = (BN[0], BN[1], TN[0], TN[1])
    bridge_line = (BB[0], BB[1], TB[0], TB[1])
    board_margin_val = parse_len_arg(args.board_margin, args.unit)
    y_min = min(y_positions) - board_margin_val
    y_max = max(y_positions) + board_margin_val
    fret_segments = collect_fret_segments(geometry, y_min, y_max)
    slots_left: List[Tuple[float, float, float, float]] = []
    slots_right: List[Tuple[float, float, float, float]] = []
    if args.emit_slot_offsets:
        if not args.slot_kerf:
            raise SystemExit("--emit-slot-offsets requires --slot-kerf.")
        kerf_val = parse_len_arg(args.slot_kerf, args.unit)
        slots_left, slots_right = collect_slot_offsets(fret_segments, kerf_val)
    strings_segments: List[Tuple[float, float, float, float]] = []
    if args.draw_strings:
        xs = [BN[0], TN[0], TB[0], BB[0]]
        board_margin_x_val = (
            parse_len_arg(args.board_margin_x, args.unit)
            if args.board_margin_x
            else board_margin_val
        )
        x_min = min(xs) - board_margin_x_val
        x_max = max(xs) + board_margin_x_val
        for y in y_positions:
            strings_segments.append((x_min, y, x_max, y))
    theta_rad = math.radians(args.datum_angle)
    if abs(args.datum_angle) > 1e-9:
        geometry = rotate_geometry(geometry, theta_rad)
        fret_segments = [rotate_segment(s, theta_rad) for s in fret_segments]
        slots_left = [rotate_segment(s, theta_rad) for s in slots_left]
        slots_right = [rotate_segment(s, theta_rad) for s in slots_right]
        strings_segments = [rotate_segment(s, theta_rad) for s in strings_segments]
        outline_segments = [rotate_segment(s, theta_rad) for s in outline_segments]
        nut_line = rotate_segment(nut_line, theta_rad)
        bridge_line = rotate_segment(bridge_line, theta_rad)
    did_any_file = False
    if args.csv:
        write_csv_file(args.csv, all_positions, all_spacings, args.unit, args.decimals)
        print(f"✅ CSV written to {args.csv}")
        did_any_file = True
    if args.json:
        write_json_file(
            args.json,
            all_positions,
            all_spacings,
            scales,
            args.unit,
            args.neutral_fret,
            y_positions,
            nut_width,
            bridge_width,
            geometry,
            args.datum_angle,
        )
        print(f"✅ JSON written to {args.json}")
        did_any_file = True
    if args.markdown or (
        not args.csv and not args.json and not args.svg and not args.dxf
    ):
        mode = "Multiscale" if len(set(scales)) > 1 else "Single-Scale"
        print(f"# Fretboard Tables ({mode})")
        print(
            f"**Unit:** {args.unit}  |  **Neutral Fret:** {args.neutral_fret}  |  **Datum Angle:** {args.datum_angle:.3f}°"
        )
        print(
            f"**Nut Width:** {nut_width:.3f} {args.unit}  |  **Bridge Width:** {bridge_width:.3f} {args.unit}"
        )
        print(
            "**Per-string scales:** "
            + " | ".join(
                f"S{i + 1}: {scales[i]:.{args.decimals}f} {args.unit}"
                for i in range(args.strings)
            )
        )
        print(
            "**String Y positions (bass=0):** "
            + " | ".join(f"{y:.3f}" for y in y_positions)
            + f" ({args.unit})\n"
        )
        print(
            make_markdown_table(
                all_positions,
                args.unit,
                args.decimals,
                "Nut-to-Fret Positions (from nut)",
            )
        )
        print()
        print(
            make_markdown_table(
                all_spacings,
                args.unit,
                args.decimals,
                "Per-Fret Spacings (incremental)",
            )
        )
    if args.svg:
        svg_write(
            filename=args.svg,
            unit=args.unit,
            stroke_width=max(args.stroke, 0.001),
            frets=fret_segments,
            slots_left=slots_left,
            slots_right=slots_right,
            strings=strings_segments,
            outline=outline_segments,
            nut=nut_line,
            bridge=bridge_line,
            pad=0.0,
        )
        print(f"✅ SVG written to {args.svg}")
        did_any_file = True
    if args.dxf:
        dxf_write(
            filename=args.dxf,
            frets=fret_segments,
            slots_left=slots_left,
            slots_right=slots_right,
            strings=strings_segments,
            outline=outline_segments,
            nut=nut_line,
            bridge=bridge_line,
        )
        print(f"✅ DXF written to {args.dxf}")
        did_any_file = True
    if (args.csv or args.json or args.svg or args.dxf) and not did_any_file:
        print(
            "⚠️  No file outputs were written; check your paths/flags.", file=sys.stderr
        )


if __name__ == "__main__":
    main()
