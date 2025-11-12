# -*- coding: utf-8 -*-
"""
fretboard_json_to_featurescript.py

A very verbose, heavily commented Python utility that converts the JSON output from
the user's fretboard calculator into FeatureScript code for Onshape. The generated
FeatureScript produces a *solid fretboard part* suitable for assemblies, and can
optionally include construction geometry for frets and strings.

USAGE (from a shell):
    python fretboard_json_to_featurescript.py \
        --json path/to/fretboard.json \
        --out  path/to/FretboardFromJSON.fs \
        --thickness 6.0 \
        --include-frets \
        --include-strings \
        --include-labels

Key goals:
- Accept "as-is" JSON produced by the fretboard calculator, but be robust to minor schema
  variations by pattern-matching common field names.
- Preserve units from the JSON (e.g., "in", "inch", "mm", "millimeter") and map them to
  FeatureScript units (inch, millimeter).
- Emit FeatureScript that:
    * Creates a top-level sketch with a closed outer profile for the fretboard outline.
    * Extrudes the closed region into a solid with the requested thickness.
    * Adds optional construction lines for each fret location (spanning between the tapered sides).
    * Adds optional construction lines for each string path (nut-to-bridge).
    * (Optional) Adds text labels for fret numbers and string indices (as sketch text).
- Provide detailed comments and explicit steps to facilitate user modification.

NOTE:
- The exact JSON schema from your fretboard calculator may differ by version.
  This script tries multiple likely keys and falls back sensibly.
- If your JSON already contains an explicit "outline" polygon (list of (x,y) points),
  we will use it as the outer profile. Otherwise, we synthesize a trapezoid/tapered
  outline based on nut width, bridge width, and scale length.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# -----------------------------
# Helper: Unit mapping and math
# -----------------------------

def detect_units(j: Dict[str, Any]) -> str:
    """
    Inspect the JSON to decide whether units are inches or millimeters.

    We deliberately check several likely fields to be robust:
      - "units", "unit", "scale_units", "length_units" (case-insensitive)
    Fallback to "mm" if unknown, but we also log a note in the generated FeatureScript.
    """
    candidates = [
        j.get("units"),
        j.get("unit"),
        j.get("scale_units"),
        j.get("length_units"),
        # Some formats tuck it into metadata
        (j.get("metadata") or {}).get("units") if isinstance(j.get("metadata"), dict) else None,
        (j.get("metadata") or {}).get("length_units") if isinstance(j.get("metadata"), dict) else None,
    ]
    candidates = [str(c).strip().lower() for c in candidates if c is not None]

    for c in candidates:
        if c in ("in", "inch", "inches"):
            return "inch"
        if c in ("mm", "millimeter", "millimeters"):
            return "millimeter"

    # Fallback: If any numeric dimensions obviously look like inches (e.g., nut_width ~ 1.6-2.2),
    # we *could* try to infer. But it's safer to just default to millimeter and record a comment.
    return "millimeter"


def fs_unit_symbol(units: str) -> str:
    """
    Convert a normalized unit string to a FeatureScript unit symbol.
    Valid outputs are "inch" or "millimeter".
    """
    if units == "inch":
        return "inch"
    return "millimeter"


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation utility."""
    return a + (b - a) * t


# --------------------------------------------
# JSON shape extraction with robust fallbacks
# --------------------------------------------

def get_float(j: Dict[str, Any], *keys: str) -> Optional[float]:
    """
    Attempt to retrieve a floating-point value from the JSON using any of the provided keys.
    Returns None if not found.
    """
    for k in keys:
        if k in j and isinstance(j[k], (int, float)):
            return float(j[k])
    return None


def get_outline_points(j: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
    """
    Attempt to find an explicit outline polygon in the JSON.
    We check a variety of likely shapes:
      - j["outline"]["points"] -> list of {x:..., y:...}
      - j["outline"] -> list of points directly
      - j["body"]["outline"]["points"], etc.

    Returns list of (x, y) tuples if found, else None.
    """
    # A small internal utility to normalize any "point-like" structure to (x, y)
    def norm_pt(p) -> Optional[Tuple[float, float]]:
        if isinstance(p, dict) and "x" in p and "y" in p:
            return (float(p["x"]), float(p["y"]))
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            return (float(p[0]), float(p[1]))
        return None

    candidates = []

    # Direct outline
    if isinstance(j.get("outline"), dict) and isinstance(j["outline"].get("points"), list):
        candidates.append(j["outline"]["points"])
    elif isinstance(j.get("outline"), list):
        candidates.append(j["outline"])

    # Sometimes nested under "body", "fretboard", etc.
    for top_key in ("body", "fretboard", "geometry", "profile"):
        sub = j.get(top_key)
        if isinstance(sub, dict):
            if isinstance(sub.get("outline"), dict) and isinstance(sub["outline"].get("points"), list):
                candidates.append(sub["outline"]["points"])
            elif isinstance(sub.get("outline"), list):
                candidates.append(sub["outline"])

    # Try each candidate list for point-like items
    for cand in candidates:
        pts: List[Tuple[float, float]] = []
        ok = True
        for p in cand:
            np = norm_pt(p)
            if np is None:
                ok = False
                break
            pts.append(np)
        if ok and len(pts) >= 3:
            return pts

    return None


def extract_scalar_dimensions(j: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Pull out common scalar dimensions with multiple fallback key names.
    This is used when we must *synthesize* a simple tapered outline.

    Returns dictionary with keys:
      - scale_length
      - nut_width
      - bridge_width
      - overall_length (if provided)
    """
    return {
        "scale_length": get_float(j, "scale_length", "scaleLen", "scale", "length", "L"),
        "nut_width": get_float(j, "nut_width", "nutWidth", "nut", "width_nut"),
        "bridge_width": get_float(j, "bridge_width", "bridgeWidth", "width_bridge"),
        "overall_length": get_float(j, "overall_length", "board_length", "fretboard_length", "total_length"),
    }


def extract_frets(j: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract a list of fret descriptors from the JSON.
    We accept several schemes:
      - Each fret dict has x coordinate along centerline: {"x": ...} (preferred)
      - Or "distance_from_nut" (d)
      - As a last resort, the list might be numbers representing distances.

    We return a normalized list of dicts like:
      [{"index": 1, "x": 12.345}, ...]
    """
    raw = j.get("frets")
    result: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return result

    for i, f in enumerate(raw, start=1):
        # Case 1: explicit object with x
        if isinstance(f, dict):
            if "x" in f:
                try:
                    result.append({"index": f.get("index", i), "x": float(f["x"])})
                    continue
                except Exception:
                    pass
            # Case 2: distance from nut
            for alt in ("d", "distance", "distance_from_nut", "pos", "position"):
                if alt in f:
                    try:
                        result.append({"index": f.get("index", i), "x": float(f[alt])})
                        break
                    except Exception:
                        pass
            else:
                # If we didn't break, we didn't append
                continue
        # Case 3: it's a number (assume x)
        if isinstance(f, (int, float)):
            result.append({"index": i, "x": float(f)})

    # Sort by x increasing just in case
    result.sort(key=lambda d: d["x"])
    return result


def extract_strings(j: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract string geometry. We expect either:
      - A list of strings, each with nut and bridge points:
          {"nut": {"x":..., "y":...}, "bridge": {"x":..., "y":...}}
      - Or a list of y positions at nut and y positions at bridge paired by index.

    We normalize to:
      [{"index": 1, "nut": (xn, yn), "bridge": (xb, yb)}, ...]
    """
    res: List[Dict[str, Any]] = []

    raw = j.get("strings")
    if isinstance(raw, list) and raw:
        # Try explicit endpoints per string
        for i, s in enumerate(raw, start=1):
            if isinstance(s, dict) and "nut" in s and "bridge" in s:
                nut = s["nut"]
                br = s["bridge"]
                try:
                    xn, yn = float(nut["x"]), float(nut["y"])
                    xb, yb = float(br["x"]), float(br["y"])
                    res.append({"index": i, "nut": (xn, yn), "bridge": (xb, yb)})
                    continue
                except Exception:
                    pass

        if res:
            return res

        # If not explicit, try a scheme with per-string y-at-nut and y-at-bridge
        y_nut = j.get("strings_y_at_nut")
        y_bridge = j.get("strings_y_at_bridge")
        x_nut = j.get("nut", {}).get("x", 0.0)
        x_bridge = j.get("bridge", {}).get("x") or extract_scalar_dimensions(j)["scale_length"]

        if isinstance(y_nut, list) and isinstance(y_bridge, list) and len(y_nut) == len(y_bridge) and x_bridge is not None:
            for i, (yn, yb) in enumerate(zip(y_nut, y_bridge), start=1):
                try:
                    res.append({"index": i, "nut": (float(x_nut), float(yn)), "bridge": (float(x_bridge), float(yb))})
                except Exception:
                    pass

    return res


# ------------------------------------------------
# Outline synthesis (when outline points are absent)
# ------------------------------------------------

def synthesize_outline(dim: Dict[str, Optional[float]]) -> Optional[List[Tuple[float, float]]]:
    """
    Create a simple tapered 2D outline (a trapezoid) for the fretboard if the JSON
    doesn't include an explicit polygon. The coordinate frame is:

      X: along the centerline from nut (x=0) to bridge/end (x=L)
      Y: + upwards across the fretboard width (top edge), - downwards (bottom edge)

    The outline corners (clockwise starting at top-nut) are:
      (0, +nut_width/2), (L, +bridge_width/2), (L, -bridge_width/2), (0, -nut_width/2)

    Returns the ordered list of 4 points closing back to start (we'll close in FS).
    """
    L = dim.get("overall_length") or dim.get("scale_length")
    Wn = dim.get("nut_width")
    Wb = dim.get("bridge_width")
    if L is None or Wn is None or Wb is None:
        return None

    half_nut = Wn / 2.0
    half_bridge = Wb / 2.0

    return [
        (0.0, +half_nut),   # Top at nut
        (L,   +half_bridge),# Top at bridge
        (L,   -half_bridge),# Bottom at bridge
        (0.0, -half_nut),   # Bottom at nut
    ]


# -----------------------------------------
# FeatureScript text generation primitives
# -----------------------------------------

def fs_vec2(x: float, y: float, unit_sym: str) -> str:
    """Format a 2D vector in FeatureScript using units, e.g., vector(10*millimeter, 5*millimeter)."""
    return f"vector({x:.6f}*{unit_sym}, {y:.6f}*{unit_sym})"


def fs_len(v: float, unit_sym: str) -> str:
    """Format a scalar length in FeatureScript with units, e.g., 6.0*millimeter."""
    return f"{v:.6f}*{unit_sym}"


def generate_featurescript(
    *,
    units: str,
    outline_pts: List[Tuple[float, float]],
    frets: List[Dict[str, Any]],
    strings: List[Dict[str, Any]],
    thickness: float,
    include_frets: bool,
    include_strings: bool,
    include_labels: bool,
    outline_label: str = "Fretboard Outline"
) -> str:
    """
    Emit a complete, self-contained FeatureScript file that:
      - Creates a new sketch on the Top plane
      - Draws the outline polygon as solid profile
      - Adds construction geometry for frets and strings (optional)
      - Solves sketch and extrudes into a solid with the given thickness
    """
    unit_sym = fs_unit_symbol(units)

    # Header and imports: using standard feature library so sketch/extrude ops exist.
    lines: List[str] = []
    lines.append("// AUTOGENERATED by fretboard_json_to_featurescript.py")
    lines.append("// This FeatureScript was generated from your JSON fretboard model.")
    lines.append("// It preserves the original units and creates a solid part plus optional construction geometry.")
    lines.append("")
    lines.append("FeatureScript 1500;")
    lines.append('import(path : "onshape/std/feature.fs", version : "1500.0");')
    lines.append("")
    lines.append("annotation {")
    lines.append('  "Feature Type Name" : "Fretboard From JSON",')
    lines.append('  "Feature Name Template" : "Fretboard~",')
    lines.append('  "Editing Logic Function" : "fretboardFromJSON"')
    lines.append("}")
    lines.append("export const fretboardFromJSON = defineFeature(function(context is Context, id is Id, definition is map)")
    lines.append("{")
    lines.append("    // --- No user-editable precondition for this generated feature ---")
    lines.append("    //     This code is tailored to the specific JSON used to generate it.")
    lines.append("")
    lines.append("    // Create a new sketch on the Top plane (XY plane).")
    lines.append("    var sk = newSketchOnPlane(context, id + \"sketch\", {\"sketchPlane\" : qCreatedBy(makeId(\"Top\"), EntityType.FACE)});")
    lines.append("")
    lines.append("    // ---------------------------")
    lines.append("    // Outline polygon (solid)")
    lines.append("    // ---------------------------")
    lines.append(f"    // {outline_label} with {len(outline_pts)} points; units preserved as {unit_sym}.")
    # Draw outline edges point-by-point
    for i in range(len(outline_pts)):
        x1, y1 = outline_pts[i]
        x2, y2 = outline_pts[(i + 1) % len(outline_pts)]  # wrap around to close
        lines.append(f"    skLine(sk, \"edge_{i}\", {{")
        lines.append(f"        \"start\" : {fs_vec2(x1, y1, unit_sym)},")
        lines.append(f"        \"end\"   : {fs_vec2(x2, y2, unit_sym)},")
        lines.append(f"        \"construction\" : false")
        lines.append("    });")
    lines.append("")
    lines.append("    // Optional: add a centerline (purely cosmetic/constructive).")
    # Compute approximate length for centerline by taking max X of outline
    max_x = max(x for x, _ in outline_pts)
    lines.append(f"    skLine(sk, \"centerline\", {{")
    lines.append(f"        \"start\" : {fs_vec2(0.0, 0.0, unit_sym)},")
    lines.append(f"        \"end\"   : {fs_vec2(max_x, 0.0, unit_sym)},")
    lines.append(f"        \"construction\" : true")
    lines.append("    });")
    lines.append("")

    # Add fret construction lines if requested
    if include_frets and frets:
        lines.append("    // ---------------------------")
        lines.append("    // Fret construction lines")
        lines.append("    // ---------------------------")
        lines.append("    // Each fret is drawn normal to the taper by spanning from bottom to top outline at that X.")
        # To span between tapered sides, we compute the local half-width at the fret's x
        # Approach: derive side edges from outline as lines; but the outline may be arbitrary polygon.
        # Simplification: assume outline is a 4-point trapezoid in order [top-nut, top-bridge, bot-bridge, bot-nut].")
        # If not trapezoid, we still span using a vertical segment from minY_at_x to maxY_at_x approx by linear edge lerp.
        # Here we *assume* trapezoid ordering and compute via linear interpolation.
        top_nut = outline_pts[0]
        top_bridge = outline_pts[1]
        bot_bridge = outline_pts[2]
        bot_nut = outline_pts[3]
        lines.append("    // NOTE: Spanning uses linear interpolation between the upper and lower edges.")
        for f in frets:
            x = float(f["x"])
            # local t along length, guard divide by zero
            t = 0.0 if max_x == 0 else min(max(x / max_x, 0.0), 1.0)
            # Interpolate y on top and bottom edges at parameter t
            y_top = lerp(top_nut[1], top_bridge[1], t)
            y_bot = lerp(bot_nut[1], bot_bridge[1], t)
            idx = f.get("index", None)
            label = f"fret_{idx}" if idx is not None else f"fret_x_{x:.3f}"
            lines.append(f"    // Fret {idx if idx is not None else '?'} at x = {x:.3f} {unit_sym}")
            lines.append("    skLine(sk, \"" + label + "\", {")
            lines.append(f"        \"start\" : {fs_vec2(x, y_bot, unit_sym)},")
            lines.append(f"        \"end\"   : {fs_vec2(x, y_top, unit_sym)},")
            lines.append("        \"construction\" : true")
            lines.append("    });")
            if include_labels and idx is not None:
                # Place a small label slightly above the top edge
                lines.append(f"    skText(sk, \"label_{idx}\", {{")
                lines.append(f"        \"text\" : \"F{idx}\",")
                lines.append(f"        \"position\" : {fs_vec2(x, y_top + 1.5, unit_sym)},")
                lines.append(f"        \"height\" : {fs_len(1.5, unit_sym)},")
                lines.append("        \"construction\" : true")
                lines.append("    });")
        lines.append("")

    # Add string construction lines if requested
    if include_strings and strings:
        lines.append("    // ---------------------------")
        lines.append("    // String construction lines")
        lines.append("    // ---------------------------")
        for s in strings:
            i = s.get("index", None)
            (xn, yn) = s["nut"]
            (xb, yb) = s["bridge"]
            label = f"string_{i}" if i is not None else f"string"
            lines.append(f"    // String {i if i is not None else '?'} from nut to bridge")
            lines.append("    skLine(sk, \"" + label + "\", {")
            lines.append(f"        \"start\" : {fs_vec2(xn, yn, unit_sym)},")
            lines.append(f"        \"end\"   : {fs_vec2(xb, yb, unit_sym)},")
            lines.append("        \"construction\" : true")
            lines.append("    });")
            if include_labels and i is not None:
                # Label near nut
                lines.append(f"    skText(sk, \"slabel_{i}\", {{")
                lines.append(f"        \"text\" : \"S{i}\",")
                lines.append(f"        \"position\" : {fs_vec2(xn + 2.0, yn + 2.0, unit_sym)},")
                lines.append(f"        \"height\" : {fs_len(1.5, unit_sym)},")
                lines.append("        \"construction\" : true")
                lines.append("    });")
        lines.append("")

    lines.append("    // Solve the sketch to generate a closed region for extrusion.")
    lines.append("    skSolve(sk);")
    lines.append("")
    lines.append("    // Extrude the closed sketch region(s) into a solid fretboard part.")
    lines.append("    // We query faces created by this sketch and blind-extrude them by the requested thickness.")
    lines.append("    var regions = qCreatedBy(id + \"sketch\", EntityType.FACE);")
    lines.append("    opExtrude(context, id + \"extrude\", {")
    lines.append("        \"entities\" : regions,")
    lines.append("        \"endBound\" : BoundingType.BLIND,")
    lines.append(f"        \"endDepth\" : {fs_len(thickness, unit_sym)},")
    lines.append("        \"operationType\" : NewBodyOperationType.NEW")
    lines.append("    });")
    lines.append("});")
    lines.append("")

    return "\n".join(lines)


# ----------------------
# Main CLI entry point
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="Convert fretboard JSON to Onshape FeatureScript (.fs)")
    parser.add_argument("--json", required=True, help="Path to the JSON file produced by the fretboard calculator.")
    parser.add_argument("--out", required=True, help="Output .fs path for the generated FeatureScript.")
    parser.add_argument("--thickness", type=float, default=None,
                        help="Board thickness (in the SAME UNITS as the JSON). If omitted, tries JSON['thickness'] or uses sensible default (6.0 mm or 0.25 in).")
    parser.add_argument("--include-frets", action="store_true", help="Include construction lines for frets.")
    parser.add_argument("--include-strings", action="store_true", help="Include construction lines for strings.")
    parser.add_argument("--include-labels", action="store_true", help="Add small labels for frets/strings in the sketch.")
    args = parser.parse_args()

    # 1) Load JSON
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2) Determine units (inch or millimeter)
    units = detect_units(data)

    # 3) Decide thickness
    if args.thickness is not None:
        thickness = float(args.thickness)
    else:
        # Try to read from JSON, otherwise choose a unit-appropriate default
        thickness = get_float(data, "thickness", "board_thickness", "fb_thickness") or (
            0.25 if units == "inch" else 6.0
        )

    # 4) Get outline points; if missing, synthesize from scalar dims
    outline = get_outline_points(data)
    if outline is None:
        dims = extract_scalar_dimensions(data)
        outline = synthesize_outline(dims)
        if outline is None:
            raise RuntimeError(
                "Could not find an explicit outline in the JSON and insufficient scalar dimensions "
                "to synthesize one (need scale_length, nut_width, bridge_width)."
            )

    # Guard: ensure outline has at least 3 points
    if len(outline) < 3:
        raise RuntimeError("Outline contains fewer than 3 points; cannot create a region to extrude.")

    # 5) Extract optional frets and strings
    frets = extract_frets(data)
    strings = extract_strings(data)

    # 6) Generate FeatureScript text
    fs_text = generate_featurescript(
        units=units,
        outline_pts=outline,
        frets=frets,
        strings=strings,
        thickness=thickness,
        include_frets=args.include_frets,
        include_strings=args.include_strings,
        include_labels=args.include_labels,
    )

    # 7) Write output .fs file
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(fs_text, encoding="utf-8")

    print(f"✅ FeatureScript written to: {out_path}")
    print(f"   Units preserved as: {units}")
    print(f"   Outline points: {len(outline)}")
    print(f"   Frets included: {len(frets)} (construction only) -> {'ON' if args.include_frets else 'OFF'}")
    print(f"   Strings included: {len(strings)} (construction only) -> {'ON' if args.include_strings else 'OFF'}")
    print(f"   Labels: {'ON' if args.include_labels else 'OFF'}")
    print(f"   Thickness: {thickness} {units}")
    print("   Import the .fs file into an Onshape custom feature and place it in a Part Studio to generate the solid.")
    print("   Tip: If your JSON contains a richer outline polygon, this script will use it verbatim for higher fidelity.")

# When executed as a script, run main().
if __name__ == "__main__":
    main()
