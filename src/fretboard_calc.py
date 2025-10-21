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
python fretboard_calc.py --frets 24 --strings 8 \
  --bass-scale 27in --treble-scale 25.5in \
  --neutral-fret 7 --string-spacing 0.35in \
  --datum-angle 5.0 \
  --json board.json \
  --svg board.svg --draw-strings

# DXF with kerf-offset slot toolpaths
python fretboard_calc.py --frets 24 --strings 6 --scale 25.5in \
  --nut-width 1.70in --bridge-width 2.20in \
  --slot-kerf 0.023in --emit-slot-offsets \
  --dxf board.dxf
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from typing import List, Tuple, Optional, Iterable, Dict


# ---------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------
IN_PER_MM = 1.0 / 25.4
MM_PER_IN = 25.4

def parse_length_with_unit(text: str, default_unit: str) -> Tuple[float, str]:
    """
    Parse a numeric string that may have a unit suffix.
    Accepts: "25.5", "25.5in", "25.5inch", "25.5inches", "648mm".
    If no suffix → assume default_unit. Returns (value, unit).
    """
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
    """
    Convert between inches and millimeters.
    """
    if from_unit == to_unit_:
        return value
    if from_unit == "in" and to_unit_ == "mm":
        return value * MM_PER_IN
    if from_unit == "mm" and to_unit_ == "in":
        return value * IN_PER_MM
    raise ValueError(f"Unsupported unit conversion: {from_unit} -> {to_unit_}")

def parse_len_arg(text: str, unit: str) -> float:
    """
    Convenience wrapper: parse a possibly-suffixed length and convert to target unit.
    """
    v, u = parse_length_with_unit(text, unit)
    return to_unit(v, u, unit)


# ---------------------------------------------------------------------
# Fret math (12-TET, along-string positions)
# ---------------------------------------------------------------------
def fret_positions_along_string(scale: float, frets: int) -> List[float]:
    """
    Cumulative positions from the nut for frets 0..F inclusive.
    pos(0) = 0.0 exactly. Uses 12-TET closed-form formula.
    """
    positions = [0.0]
    for n in range(1, frets + 1):
        positions.append(scale - scale / (2.0 ** (n / 12.0)))
    return positions

def fret_spacings_from_positions(positions: List[float]) -> List[float]:
    """
    Per-fret increments (distance from fret n-1 to n).
    spacing[0] = 0 for alignment with positions indexing.
    """
    out = [0.0]
    for i in range(1, len(positions)):
        out.append(positions[i] - positions[i - 1])
    return out


# ---------------------------------------------------------------------
# Multiscale mapping (linear / exponential)
# ---------------------------------------------------------------------
def interpolate_scales_for_strings(
    strings: int,
    L_bass: float,
    L_treble: float,
    map_mode: str = "linear",
    gamma: float = 1.0,
) -> List[float]:
    """
    Compute per-string scales L_s (s=0..S-1).
    map_mode:
      - "linear": t = s/(S-1)
      - "exp":    t = (s/(S-1))**gamma
        gamma > 1 biases lengths toward bass; gamma < 1 toward treble.
    """
    if strings == 1:
        return [L_bass]
    out: List[float] = []
    for s in range(strings):
        t = s / (strings - 1)
        if map_mode == "exp":
            t = (t ** gamma) if strings > 1 else 0.0
        # Blend bass→treble
        L = L_bass * (1.0 - t) + L_treble * t
        out.append(L)
    return out


# ---------------------------------------------------------------------
# String spacing (uniform, list, or file)
# ---------------------------------------------------------------------
def parse_spacing_list(text: str, unit: str) -> List[float]:
    """
    Parse a comma-separated list of gaps (S-1 items), possibly with units.
    Example: "0.35in,0.35in,0.36in" -> [0.35, 0.35, 0.36] (converted).
    """
    parts = [p for p in (x.strip() for x in text.split(",")) if p]
    return [parse_len_arg(p, unit) for p in parts]

def parse_spacing_file(path: str, unit: str) -> List[float]:
    """
    Read S-1 gaps from a file, one gap per line. Blank lines ignored.
    """
    gaps: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            gaps.append(parse_len_arg(s, unit))
    return gaps

def build_string_y_positions(strings: int,
                             uniform_spacing: Optional[float],
                             gap_list: Optional[List[float]]) -> List[float]:
    """
    Return absolute y positions for each string index (0..S-1), where
    y[0]=0 on the bass string and y increases toward treble side.
    If 'gap_list' is provided, it must have length S-1. Otherwise we
    use 'uniform_spacing' for all adjacent gaps.
    """
    if strings < 1:
        return []
    if strings == 1:
        return [0.0]
    if gap_list is not None:
        if len(gap_list) != strings - 1:
            raise SystemExit(f"--string-spacing-list/file must specify exactly {strings-1} gaps.")
        gaps = gap_list
    else:
        if uniform_spacing is None:
            raise SystemExit("Internal: need either uniform spacing value or a gap list.")
        gaps = [uniform_spacing] * (strings - 1)

    y = [0.0]
    acc = 0.0
    for g in gaps:
        acc += g
        y.append(acc)
    return y


# ---------------------------------------------------------------------
# Geometry construction (neutral-fret-relative)
# ---------------------------------------------------------------------
def calculate_fret_geometry(
    positions: List[List[float]],
    y_positions: List[float],
    neutral_fret: int,
) -> List[dict]:
    """
    For each fret n, compute a polyline of points (x,y) across strings,
    with x measured relative to the neutral fret on each string:
      x_s(n) = pos_s(n) - pos_s(neutral_fret)
      y_s    = y_positions[s]

    Returns list of dicts:
      {
        "fret": n,
        "angle_deg": angle from bass→treble endpoint,
        "coordinates": [ {"string": s+1, "x": x, "y": y_s}, ... ]
      }
    """
    S = len(positions)
    F = len(positions[0]) - 1
    geometry: List[dict] = []

    # Cache neutral offsets (pos_s(N)) for faster loops
    offs = [positions[s][neutral_fret] for s in range(S)]

    for n in range(F + 1):
        coords = []
        for s in range(S):
            x = positions[s][n] - offs[s]
            y = y_positions[s]
            coords.append({"string": s + 1, "x": round(x, 6), "y": round(y, 6)})

        # Angle uses just bass & treble endpoints
        x0, y0 = coords[0]["x"], coords[0]["y"]
        x1, y1 = coords[-1]["x"], coords[-1]["y"]
        dx, dy = (x1 - x0), (y1 - y0)
        angle_deg = math.degrees(math.atan2(dx, dy)) if abs(dy) > 1e-12 else 0.0

        geometry.append({
            "fret": n,
            "angle_deg": round(angle_deg, 6),
            "coordinates": coords
        })

    return geometry


# ---------------------------------------------------------------------
# Outline construction (tapered quad) + nut/bridge lines
# ---------------------------------------------------------------------
def outline_corners(
    scales: List[float],
    positions: List[List[float]],
    neutral_fret: int,
    nut_width: float,
    bridge_width: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute the 4 corners of the board outline polygon:

      Bass Nut Corner     (BN): at the NUT along bass string, y = -nut_width/2
      Treble Nut Corner   (TN): at the NUT along treble string, y = +nut_width/2
      Bass Bridge Corner  (BB): at the BRIDGE (scale length) on bass, y = -bridge_width/2
      Treble Bridge Corner(TB): at the BRIDGE (scale length) on treble, y = +bridge_width/2

    x at nut for string s  = pos_s(0)   - pos_s(N)
    x at bridge for string s ~ L_s      - pos_s(N)
    """
    s_bass = 0
    s_treb = len(scales) - 1

    # x at nut (fret 0) relative to neutral
    x_nut_bass = positions[s_bass][0] - positions[s_bass][neutral_fret]   # = -pos_s(N)
    x_nut_treb = positions[s_treb][0] - positions[s_treb][neutral_fret]

    # x at bridge (scale length) relative to neutral
    x_bridge_bass = scales[s_bass] - positions[s_bass][neutral_fret]
    x_bridge_treb = scales[s_treb] - positions[s_treb][neutral_fret]

    BN = (x_nut_bass, -nut_width / 2.0)
    TN = (x_nut_treb, +nut_width / 2.0)
    BB = (x_bridge_bass, -bridge_width / 2.0)
    TB = (x_bridge_treb, +bridge_width / 2.0)

    return {"BN": BN, "TN": TN, "TB": TB, "BB": BB}


# ---------------------------------------------------------------------
# Fret line construction across full board span (and kerf offsets)
# ---------------------------------------------------------------------
def _extend_line_to_y_span(x0: float, y0: float, x1: float, y1: float,
                           y_min: float, y_max: float) -> Tuple[float, float, float, float]:
    """
    Extend (or trim) the line through (x0,y0)-(x1,y1) so that it runs from y_min to y_max.
    If the line is (nearly) horizontal, we still return a vertical segment at x=x0 spanning
    y_min..y_max (this matches the neutral-fret-relative scheme and keeps CAD robust).
    """
    if abs(y1 - y0) < 1e-12:
        return (x0, y_min, x0, y_max)
    def x_of_y(y: float) -> float:
        t = (y - y0) / (y1 - y0)
        return x0 + (x1 - x0) * t
    return (x_of_y(y_min), y_min, x_of_y(y_max), y_max)

def collect_fret_segments(
    geometry: List[dict],
    y_min: float,
    y_max: float,
) -> List[Tuple[float, float, float, float]]:
    """
    For each fret, form a long segment spanning from y_min to y_max by extending the
    bass↔treble chord (two endpoints in geometry).
    """
    segs: List[Tuple[float, float, float, float]] = []
    for fret in geometry:
        (x0, y0) = (fret["coordinates"][0]["x"], fret["coordinates"][0]["y"])
        (x1, y1) = (fret["coordinates"][-1]["x"], fret["coordinates"][-1]["y"])
        segs.append(_extend_line_to_y_span(x0, y0, x1, y1, y_min, y_max))
    return segs

def offset_line_normal(
    x1: float, y1: float, x2: float, y2: float, offset: float
) -> Tuple[float, float, float, float]:
    """
    Offset a line segment by a signed distance along its left-hand normal.
    For a segment vector v = (dx, dy), a unit left-normal is n = (-dy, dx)/|v|.
    New endpoints: p1' = p1 + offset*n, p2' = p2 + offset*n.
    """
    dx, dy = (x2 - x1), (y2 - y1)
    L = math.hypot(dx, dy)
    if L < 1e-12:
        return (x1, y1, x2, y2)
    nx, ny = (-dy / L, dx / L)
    return (x1 + offset * nx, y1 + offset * ny, x2 + offset * nx, y2 + offset * ny)

def collect_slot_offsets(
    fret_segments: List[Tuple[float, float, float, float]],
    kerf: float
) -> Tuple[List[Tuple[float,float,float,float]], List[Tuple[float,float,float,float]]]:
    """
    Given center fret segments and a kerf width, produce two toolpath lines:
    LEFT  = center offset by -kerf/2
    RIGHT = center offset by +kerf/2
    """
    half = kerf / 2.0
    left, right = [], []
    for (x1,y1,x2,y2) in fret_segments:
        left.append( offset_line_normal(x1,y1,x2,y2, -half) )
        right.append( offset_line_normal(x1,y1,x2,y2, +half) )
    return left, right


# ---------------------------------------------------------------------
# ✨ DATUM ROTATION (Step 1) ✨
# ---------------------------------------------------------------------
def rotate_point(x: float, y: float, theta_rad: float) -> Tuple[float, float]:
    """
    Rotate a point (x,y) counterclockwise around (0,0) by theta_rad radians.
    """
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    X = x * cos_t - y * sin_t
    Y = x * sin_t + y * cos_t
    return X, Y

def rotate_segment(seg: Tuple[float,float,float,float], theta_rad: float) -> Tuple[float,float,float,float]:
    """
    Rotate a line segment (x1,y1,x2,y2) around (0,0) by theta_rad radians.
    """
    x1, y1, x2, y2 = seg
    X1, Y1 = rotate_point(x1, y1, theta_rad)
    X2, Y2 = rotate_point(x2, y2, theta_rad)
    return (X1, Y1, X2, Y2)

def rotate_geometry(geometry: List[dict], theta_rad: float) -> List[dict]:
    """
    Rotate all fret geometry coordinates around the origin by theta_rad.
    Returns a deep-rotated copy of the geometry list.
    """
    rotated = []
    for fret in geometry:
        new_coords = []
        for c in fret["coordinates"]:
            X, Y = rotate_point(c["x"], c["y"], theta_rad)
            new_coords.append({"string": c["string"], "x": round(X, 6), "y": round(Y, 6)})
        rotated.append({
            "fret": fret["fret"],
            "angle_deg": fret["angle_deg"] + math.degrees(theta_rad),
            "coordinates": new_coords
        })
    return rotated


# ---------------------------------------------------------------------
# Output emitters (Markdown, CSV, JSON)
# ---------------------------------------------------------------------
def fmt_val(x: float, unit: str, decimals: int) -> str:
    return f"{x:.{decimals}f}"

def make_markdown_table(data: List[List[float]], unit: str, decimals: int, title: str) -> str:
    S = len(data)
    F = len(data[0]) - 1
    header = ["Fret"] + [f"String {i+1} ({unit})" for i in range(S)]
    sep = ["---"] * len(header)
    lines = [f"## {title}", "| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for n in range(F + 1):
        row = [str(n)] + [fmt_val(data[s][n], unit, decimals) for s in range(S)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

def write_csv_file(filename: str, positions: List[List[float]], spacings: List[List[float]], unit: str, decimals: int):
    S = len(positions)
    F = len(positions[0]) - 1
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"Nut-to-Fret Positions ({unit})"])
        w.writerow(["Fret"] + [f"String {i+1}" for i in range(S)])
        for n in range(F + 1):
            w.writerow([n] + [f"{positions[s][n]:.{decimals}f}" for s in range(S)])
        w.writerow([])
        w.writerow([f"Per-Fret Spacings ({unit})"])
        w.writerow(["Fret"] + [f"String {i+1}" for i in range(S)])
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
        "fret_geometry": geometry
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------
# SVG/DXF Writers
# ---------------------------------------------------------------------
def svg_write(
    filename: str,
    unit: str,
    stroke_width: float,
    frets: List[Tuple[float,float,float,float]],
    slots_left: List[Tuple[float,float,float,float]],
    slots_right: List[Tuple[float,float,float,float]],
    strings: List[Tuple[float,float,float,float]],
    outline: List[Tuple[float,float,float,float]],
    nut: Tuple[float,float,float,float],
    bridge: Tuple[float,float,float,float],
    pad: float = 0.0
):
    """
    Emit an SVG with groups for OUTLINE, NUT, BRIDGE, FRETS, SLOTS, STRINGS.
    Stroke widths are in user units (same as --unit).
    """
    def add_bounds(seglist: Iterable[Tuple[float,float,float,float]], xs: List[float], ys: List[float]):
        for (x1,y1,x2,y2) in seglist:
            xs.extend([x1,x2]); ys.extend([y1,y2])

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

    xmin, xmax = min(xs)-pad, max(xs)+pad
    ymin, ymax = min(ys)-pad, max(ys)+pad
    width, height = (xmax-xmin), (ymax-ymin)

    def line_el(x1,y1,x2,y2) -> str:
        return f'<line x1="{x1:.6f}" y1="{y1:.6f}" x2="{x2:.6f}" y2="{y2:.6f}" />'

    with open(filename, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin:.6f} {ymin:.6f} {width:.6f} {height:.6f}">\n')
        f.write(f'  <desc>Units: {unit}. Datum rotation supported.</desc>\n')
        f.write('  <style>\n')
        f.write(f'    .FRETS   {{ stroke:black; fill:none; stroke-width:{stroke_width:.6f}; }}\n')
        f.write(f'    .SLOTS   {{ stroke:#444;  fill:none; stroke-width:{max(stroke_width*0.6,0.001):.6f}; }}\n')
        f.write(f'    .STRINGS {{ stroke:#999;  fill:none; stroke-width:{max(stroke_width*0.5,0.001):.6f}; }}\n')
        f.write(f'    .OUTLINE {{ stroke:#07c;  fill:none; stroke-width:{max(stroke_width*0.8,0.001):.6f}; }}\n')
        f.write(f'    .NUT, .BRIDGE {{ stroke:#c07; fill:none; stroke-width:{max(stroke_width*0.8,0.001):.6f}; }}\n')
        f.write('  </style>\n')

        def write_seg(gid: str, cls: str, segs: List[Tuple[float,float,float,float]]):
            f.write(f'  <g id="{gid}" class="{cls}">\n')
            for (x1,y1,x2,y2) in segs:
                f.write("    " + line_el(x1,y1,x2,y2) + "\n")
            f.write('  </g>\n')

        write_seg("OUTLINE", "OUTLINE", outline)
        f.write('  <g id="NUT" class="NUT">\n'); f.write("    " + line_el(*nut) + "\n"); f.write('  </g>\n')
        f.write('  <g id="BRIDGE" class="BRIDGE">\n'); f.write("    " + line_el(*bridge) + "\n"); f.write('  </g>\n')
        write_seg("FRETS", "FRETS", frets)
        write_seg("SLOTS", "SLOTS", slots_left + slots_right)
        write_seg("STRINGS", "STRINGS", strings)

        f.write('</svg>\n')

def dxf_write(
    filename: str,
    frets: List[Tuple[float,float,float,float]],
    slots_left: List[Tuple[float,float,float,float]],
    slots_right: List[Tuple[float,float,float,float]],
    strings: List[Tuple[float,float,float,float]],
    outline: List[Tuple[float,float,float,float]],
    nut: Tuple[float,float,float,float],
    bridge: Tuple[float,float,float,float],
):
    """
    Minimal ASCII DXF (ENTITIES only). Each LINE goes to a named LAYER.
    """
    def line_entity(x1, y1, x2, y2, layer) -> str:
        return ("0\nLINE\n"
                f"8\n{layer}\n"
                f"10\n{x1:.6f}\n20\n{y1:.6f}\n11\n{x2:.6f}\n21\n{y2:.6f}\n")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("0\nSECTION\n2\nENTITIES\n")
        for (x1,y1,x2,y2) in outline:
            f.write(line_entity(x1,y1,x2,y2,"OUTLINE"))
        f.write(line_entity(*nut,"NUT"))
        f.write(line_entity(*bridge,"BRIDGE"))
        for (x1,y1,x2,y2) in frets:
            f.write(line_entity(x1,y1,x2,y2,"FRETS"))
        for (x1,y1,x2,y2) in slots_left:
            f.write(line_entity(x1,y1,x2,y2,"SLOTS"))
        for (x1,y1,x2,y2) in slots_right:
            f.write(line_entity(x1,y1,x2,y2,"SLOTS"))
        for (x1,y1,x2,y2) in strings:
            f.write(line_entity(x1,y1,x2,y2,"STRINGS"))
        f.write("0\nENDSEC\n0\nEOF\n")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Full-featured fretboard calculator + CAD (SVG/DXF) generator"
    )
    # Core counts
    p.add_argument("--frets",   type=int, required=True, help="Number of frets (>=1).")
    p.add_argument("--strings", type=int, required=True, help="Number of strings (>=1).")

    # Scale(s): single or multi
    p.add_argument("--scale",        type=str, default=None, help='Single-scale, e.g. "25.5in" or "648mm".')
    p.add_argument("--bass-scale",   type=str, default=None, help='Multiscale: bass side, e.g. "27in".')
    p.add_argument("--treble-scale", type=str, default=None, help='Multiscale: treble side, e.g. "25.5in".')

    # Multiscale mapping
    p.add_argument("--scale-map",   choices=["linear","exp"], default="linear",
                   help="Per-string scale interpolation: linear or exponential.")
    p.add_argument("--scale-gamma", type=float, default=1.0,
                   help="Gamma for --scale-map exp (t -> t**gamma). 1.0 = linear.")

    # Units & formatting
    p.add_argument("--unit",     choices=["in","mm"], default="in", help="Computation/output unit.")
    p.add_argument("--decimals", type=int, default=3, help="Decimal places for tables.")

    # Neutral fret & spacing (y-axis)
    p.add_argument("--neutral-fret", type=int, default=0, help="Neutral fret index (0..F). Default 0 (nut).")

    # String spacing:
    p.add_argument("--string-spacing",      type=str, default=None,
                   help='Uniform per-adjacent-string gap, e.g. "0.35in" or "9mm".')
    p.add_argument("--string-spacing-list", type=str, default=None,
                   help='CSV list of S-1 gaps, e.g. "0.35in,0.35in,0.36in,...".')
    p.add_argument("--string-spacing-file", type=str, default=None,
                   help='Path to file containing S-1 gaps, one per line.')

    # Board outline widths (total edge-to-edge)
    p.add_argument("--nut-width",    type=str, default="1.70in", help='Total board width at nut, e.g. "1.70in".')
    p.add_argument("--bridge-width", type=str, default="2.20in", help='Total board width at bridge, e.g. "2.20in".')

    # Text/number outputs
    p.add_argument("--markdown", action="store_true", help="Force Markdown to stdout.")
    p.add_argument("--csv",      type=str, help="Write CSV file.")
    p.add_argument("--json",     type=str, help="Write JSON file.")

    # Drawings
    p.add_argument("--svg", type=str, help="Write SVG drawing.")
    p.add_argument("--dxf", type=str, help="Write DXF drawing.")

    # Drawing style & extras
    p.add_argument("--draw-strings", action="store_true", help="Include string centerlines in drawings.")
    p.add_argument("--board-margin",  type=str, default="0.10in",
                   help='Extra vertical margin beyond bass/treble strings (adds to y-span).')
    p.add_argument("--board-margin-x", type=str, default=None,
                   help='Horizontal margin for string lines (defaults to board-margin).')
    p.add_argument("--stroke", type=float, default=0.6,
                   help="SVG stroke width in user units (same as --unit).")

    # Slot kerf offsets
    p.add_argument("--slot-kerf",        type=str, default=None,
                   help='Kerf width for slot toolpaths, e.g. "0.023in" (saw blade width).')
    p.add_argument("--emit-slot-offsets", action="store_true",
                   help="If set and --slot-kerf provided, emit left/right slot lines on SLOTS layer.")

    # Datum angle rotation (Step 1)
    p.add_argument("--datum-angle", type=float, default=0.0,
                   help="Rotate entire fretboard around neutral fret origin by this angle (degrees).")

    return p

def validate_and_get_scales(args) -> List[float]:
    """
    Return per-string scales in chosen --unit, using single or multiscale options
    and selected mapping mode (linear/exp + gamma).
    """
    if args.bass_scale and args.treble_scale:
        Lb = parse_len_arg(args.bass_scale, args.unit)
        Lt = parse_len_arg(args.treble_scale, args.unit)
        if args.strings < 2:
            raise SystemExit("Multiscale requires --strings >= 2.")
        return interpolate_scales_for_strings(args.strings, Lb, Lt, args.scale_map, args.scale_gamma)
    if args.scale:
        L = parse_len_arg(args.scale, args.unit)
        return [L] * args.strings
    raise SystemExit("Provide --scale (single) OR both --bass-scale and --treble-scale (multi).")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = build_arg_parser().parse_args()

    # Sanity
    if args.frets < 1:
        raise SystemExit("--frets must be >= 1.")
    if args.strings < 1:
        raise SystemExit("--strings must be >= 1.")
    if not (0 <= args.neutral_fret <= args.frets):
        raise SystemExit(f"--neutral-fret must be in [0..{args.frets}].")

    # Per-string scale lengths (single/multiscale)
    scales = validate_and_get_scales(args)

    # String spacing handling (Y positions)
    uniform_spacing: Optional[float] = None
    gap_list: Optional[List[float]] = None

    if args.string_spacing_list and args.string_spacing_file:
        raise SystemExit("Use either --string-spacing-list OR --string-spacing-file, not both.")

    if args.string_spacing_list:
        gap_list = parse_spacing_list(args.string_spacing_list, args.unit)
    elif args.string_spacing_file:
        gap_list = parse_spacing_file(args.string_spacing_file, args.unit)
    else:
        if args.string_spacing is None:
            # Sensible default when nothing provided (roughly ~9mm):
            uniform_spacing = parse_len_arg("0.354in" if args.unit=="in" else "9mm", args.unit)
        else:
            uniform_spacing = parse_len_arg(args.string_spacing, args.unit)

    y_positions = build_string_y_positions(args.strings, uniform_spacing, gap_list)

    # Positions & spacings per string
    all_positions = [fret_positions_along_string(L, args.frets) for L in scales]
    all_spacings  = [fret_spacings_from_positions(pos) for pos in all_positions]

    # Geometry (neutral-fret-relative X across strings)
    geometry = calculate_fret_geometry(all_positions, y_positions, args.neutral_fret)

    # Outline + nut/bridge in neutral frame
    nut_width    = parse_len_arg(args.nut_width, args.unit)
    bridge_width = parse_len_arg(args.bridge_width, args.unit)
    corners = outline_corners(scales, all_positions, args.neutral_fret, nut_width, bridge_width)
    BN, TN, TB, BB = corners["BN"], corners["TN"], corners["TB"], corners["BB"]

    # Build OUTLINE edges (BN->TN->TB->BB->BN)
    outline_segments = [
        (BN[0], BN[1], TN[0], TN[1]),  # nut edge across width
        (TN[0], TN[1], TB[0], TB[1]),  # treble edge downboard
        (TB[0], TB[1], BB[0], BB[1]),  # bridge edge across width
        (BB[0], BB[1], BN[0], BN[1]),  # bass edge upboard
    ]
    nut_line    = (BN[0], BN[1], TN[0], TN[1])
    bridge_line = (BB[0], BB[1], TB[0], TB[1])

    # Fret segments spanning vertically a little beyond strings
    board_margin_val = parse_len_arg(args.board_margin, args.unit)
    y_min = min(y_positions) - board_margin_val
    y_max = max(y_positions) + board_margin_val
    fret_segments = collect_fret_segments(geometry, y_min, y_max)

    # Slot kerf offsets (optional)
    slots_left: List[Tuple[float,float,float,float]] = []
    slots_right: List[Tuple[float,float,float,float]] = []
    if args.emit_slot_offsets:
        if not args.slot_kerf:
            raise SystemExit("--emit-slot-offsets requires --slot-kerf.")
        kerf_val = parse_len_arg(args.slot_kerf, args.unit)
        slots_left, slots_right = collect_slot_offsets(fret_segments, kerf_val)

    # Optional string centerlines over x-span of the board outline
    strings_segments: List[Tuple[float,float,float,float]] = []
    if args.draw_strings:
        xs = [BN[0], TN[0], TB[0], BB[0]]
        board_margin_x_val = parse_len_arg(args.board_margin_x, args.unit) if args.board_margin_x else board_margin_val
        x_min = min(xs) - board_margin_x_val
        x_max = max(xs) + board_margin_x_val
        for y in y_positions:
            strings_segments.append((x_min, y, x_max, y))

    # =====================================
    # DATUM ROTATION APPLY (Step 1)
    # =====================================
    theta_rad = math.radians(args.datum_angle)
    if abs(args.datum_angle) > 1e-9:
        geometry = rotate_geometry(geometry, theta_rad)
        # Rotate all segments in place
        fret_segments   = [rotate_segment(s, theta_rad) for s in fret_segments]
        slots_left      = [rotate_segment(s, theta_rad) for s in slots_left]
        slots_right     = [rotate_segment(s, theta_rad) for s in slots_right]
        strings_segments= [rotate_segment(s, theta_rad) for s in strings_segments]
        outline_segments= [rotate_segment(s, theta_rad) for s in outline_segments]
        nut_line        = rotate_segment(nut_line, theta_rad)
        bridge_line     = rotate_segment(bridge_line, theta_rad)

    # ----------------------------
    # Outputs
    # ----------------------------
    did_any_file = False

    if args.csv:
        write_csv_file(args.csv, all_positions, all_spacings, args.unit, args.decimals)
        print(f"✅ CSV written to {args.csv}")
        did_any_file = True

    if args.json:
        write_json_file(args.json, all_positions, all_spacings, scales, args.unit,
                        args.neutral_fret, y_positions, nut_width, bridge_width,
                        geometry, args.datum_angle)
        print(f"✅ JSON written to {args.json}")
        did_any_file = True

    # Markdown if asked or if *nothing* else was requested
    if args.markdown or (not args.csv and not args.json and not args.svg and not args.dxf):
        mode = "Multiscale" if len(set(scales)) > 1 else "Single-Scale"
        print(f"# Fretboard Tables ({mode})")
        print(f"**Unit:** {args.unit}  |  **Neutral Fret:** {args.neutral_fret}  |  **Datum Angle:** {args.datum_angle:.3f}°")
        print(f"**Nut Width:** {nut_width:.3f} {args.unit}  |  **Bridge Width:** {bridge_width:.3f} {args.unit}")
        print("**Per-string scales:** " + " | ".join(f"S{i+1}: {scales[i]:.{args.decimals}f} {args.unit}" for i in range(args.strings)))
        print("**String Y positions (bass=0):** " + " | ".join(f"{y:.3f}" for y in y_positions) + f" ({args.unit})\n")
        print(make_markdown_table(all_positions, args.unit, args.decimals, "Nut-to-Fret Positions (from nut)"))
        print()
        print(make_markdown_table(all_spacings, args.unit, args.decimals, "Per-Fret Spacings (incremental)"))

    # Drawings
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
            pad=0.0
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
        print("⚠️  No file outputs were written; check your paths/flags.", file=sys.stderr)


if __name__ == "__main__":
    main()
