# Fretboard Calculator & CAD Generator (`fretboard_calc.py`)

## 📜 Overview

`fretboard_calc.py` is a **professional-grade, command-line fretboard calculation and CAD export tool** intended for **luthiers, guitar builders, CNC operators, and CAD/CAM users** who need **precise fret positions** for musical instruments — including both **traditional single-scale** and **modern multiscale (fanned fret)** designs.

Unlike many basic calculators, this script does **not** simply dump fret distances; it can:

- Compute **per-string 12-TET fret positions** accurately using the closed-form equation.  
- Support **multiscale interpolation** (linear or exponential).  
- Output **tabular and structured data** suitable for spreadsheets and downstream CAD workflows.  
- Generate **parametric SVG or DXF drawings** of the fretboard including frets, strings, nut, bridge, and board outline.  
- Apply a **datum angle rotation**, allowing the entire fretboard to be rotated around the neutral fret without changing the geometry.

The goal is to **bridge the gap between theoretical fret spacing math and practical CAD/CAM fabrication**.

---

## 🧮 Core Functionality

The script is built around **12-Tone Equal Temperament (12-TET)**:

\[
\text{pos}(n) = L - \frac{L}{2^{(n/12)}}
\]

- `L` = scale length for that string
- `n` = fret number
- `pos(n)` = distance from the nut to the `n`-th fret

This gives exact results without cumulative rounding errors.

For **multiscale instruments**, the script computes a different scale length for each string by interpolating between bass and treble scale lengths:

- **Linear mode**: even interpolation across strings.
- **Exponential mode**: interpolation weighted by a `gamma` factor, biasing toward bass or treble.

All geometry is constructed in a **neutral-fret coordinate system**, meaning the `x = 0` line runs through the chosen neutral fret. This simplifies later transformations, like rotating the entire board to a specific datum angle.

---

## 🧭 Coordinate System & Geometry

- **X-axis** = along-string direction (positive away from nut)  
- **Y-axis** = string spacing direction (bass string at `y = 0`)  
- All fret coordinates are stored **relative to the neutral fret** (the fret around which the board may be rotated).

### Key geometric features produced:
- Fret lines (full width across board)  
- Nut and bridge edges  
- Outline polygon (tapered quadrilateral)  
- Optional slot kerf offsets for toolpath generation  
- Optional string centerlines  

---

## 🪚 Datum Angle Rotation

After the entire fretboard is computed, you can optionally **rotate everything by a given angle** around the neutral fret.

This is useful for:
- Aligning the board to your machine’s datum
- Simulating fan alignment changes
- Adjusting zeroing strategies without recomputing the entire geometry

All segments and coordinates (frets, nut, bridge, outline, slots, strings) are rotated consistently.

---

## 🧾 Output Formats

The script can emit multiple outputs in a single run:

- 📝 **Markdown (stdout)** — Readable tables of fret positions and spacings. Great for inspection and version control.  
- 📊 **CSV** — Easy to load into spreadsheets or CAD/CAM tools.  
- 🧱 **JSON** — Fully structured geometric and parametric data. Ideal for downstream automation.  
- 🖼 **SVG** — Lightweight vector drawing with logical layers for frets, strings, outline, nut, bridge, and slot offsets.  
- 📐 **DXF (ASCII)** — Minimal DXF file containing only LINE entities on named layers, suitable for import into Inkscape, Fusion 360, LightBurn, FreeCAD, etc.

---

## 🧭 Units & Measurement

- Supported units: **inches** (`in`) and **millimeters** (`mm`).  
- You can specify units globally with `--unit in|mm`.  
- Most length arguments accept inline suffixes:
  - `25.5in` or `25.5inch`
  - `648mm`
  - or just a bare number, which will be interpreted in the chosen default unit.

---

---

## 🖥️ Graphical User Interface — `fretboard_calc_gui.py`

### 📘 Overview

`fretboard_calc_gui.py` is a **PyQt6-based graphical frontend** for the `fretboard_calc.py` CLI tool.  
It provides a **live, interactive environment** for designing fretboards visually — with all command-line parameters exposed as GUI controls.

This GUI is especially useful for **rapid iteration**, **visual confirmation**, and **preset management** when experimenting with different multiscale configurations.

---

### 🧩 Features

**Top Half:**
- 🖼 **Live SVG Preview:** Renders the fretboard drawing in real-time using `QGraphicsView` and `QGraphicsSvgItem`.  
- 🖱 **Interactive Controls:**
  - Mouse wheel zoom centered at cursor.
  - Middle-mouse pan, or *Space + Left-drag* pan.
- 🔍 **Toolbar Controls:**
  - Zoom In / Zoom Out / 100% / Fit to View
  - Live zoom percentage display.
  - Export buttons for SVG and CSV outputs.

**Bottom Half:**
- ⚙️ **All CLI Parameters as Input Fields:**
  - Frets, strings, scale lengths, neutral fret, units, board widths, margins, datum angle, slot kerf, etc.
  - Full support for string spacing modes: uniform, CSV list, or external file.
- 💾 **Preset Manager:**
  - Create, apply, update, or delete presets.
  - Presets stored automatically in `~/.fretboard_gui_presets.json`.
- 📊 **Numeric Table:**
  - Displays live fret positions and spacings parsed from the CLI’s temporary JSON output.
- 🧾 **Log Window:**
  - Shows CLI command output, normalization summaries, and any error messages.

---

### 🧠 How It Works

Internally, the GUI acts as a **smart wrapper** around the CLI script:

1. Each time you modify a parameter, the GUI builds a full command-line argument list (just like you’d type in the terminal).
2. The command is executed in a **background QThread** using Python’s `subprocess.run()`, so the interface never freezes.
3. The CLI writes temporary preview files (`preview.svg` and `preview.json`) into a system temp directory.
4. The GUI then:
   - Loads the SVG into the live preview area.
   - Parses the JSON into the numeric table.
   - Displays stdout and stderr logs in the bottom panel.
5. You can export the current SVG or CSV table to any chosen location at any time.

---

### ⚙️ Installation

Install PyQt6 (and SVG support):

```bash
pip install PyQt6 PyQt6-QtSvg
```

Make sure `fretboard_calc_gui.py` is located **in the same directory** as `fretboard_calc.py`.

---

### 🚀 Launching the GUI

From your project’s source folder:

```bash
python3 fretboard_calc_gui.py
```

When launched:

* A window appears with a toolbar, SVG preview, and full control panel.
* Auto-preview is **enabled by default** — the fretboard regenerates ~300 ms after changes.
* You can toggle Auto Preview using the button at the bottom of the control column.

---

### 🧾 Files and Persistence

| File                                    | Description                                                         |
| --------------------------------------- | ------------------------------------------------------------------- |
| `~/.fretboard_gui_presets.json`         | Stores all user-defined presets                                     |
| `~/.fretboard_gui_config.json`          | Remembers window geometry, auto-preview state, and last used preset |
| `/tmp/fretboard_gui_*/preview.svg/json` | Temporary working files generated during live preview               |

These are automatically managed — you can delete them safely if needed.

---

### 💡 Tips

| Action                 | Shortcut or Control                          |
| ---------------------- | -------------------------------------------- |
| Zoom in/out            | Mouse wheel or toolbar buttons               |
| Pan                    | Middle-mouse drag or Space + Left-drag       |
| Fit preview            | Toolbar “Fit” button or “Fit Preview” button |
| Export SVG             | Toolbar “Export SVG…”                        |
| Export numeric table   | Toolbar “Export Table CSV…”                  |
| Regenerate immediately | “Generate Now” button                        |
| Toggle auto preview    | “Auto Preview: ON/OFF” button                |
| Clear console log      | “Clear Log” button                           |

---

### 📄 Example Workflow

1. Run `python3 fretboard_calc_gui.py`
2. Choose preset **“8-String Fan (27 → 25.5”)”**
3. Adjust parameters (e.g. neutral fret, margins, angle)
4. Watch the SVG preview update automatically
5. Use zoom/pan to inspect geometry
6. Click **Export SVG…** to save a finalized design
7. Optionally export the fret table via **Export Table CSV…**

---

### 🧰 Why Use the GUI?

* No need to re-enter long command lines.
* Instantly visualize scale, fan angle, and spacing effects.
* Manage reusable configurations via presets.
* Non-blocking — you can continue interacting while it computes.
* Perfect companion for builders doing **rapid design iteration** before sending CAD files to CAM/CNC.

---

### ⚠️ Troubleshooting

| Issue                         | Fix                                                           |
| ----------------------------- | ------------------------------------------------------------- |
| GUI won’t start               | Ensure `PyQt6` and `PyQt6-QtSvg` are installed                |
| SVG preview blank             | Check for errors in the Log tab — usually a missing parameter |
| Nothing regenerates           | Verify *Auto Preview* is ON or click *Generate Now*           |
| `fretboard_calc.py` not found | Keep both scripts in the same directory                       |

---

### 🧩 Summary

`fretboard_calc_gui.py` provides a **complete visual frontend** for `fretboard_calc.py`:

* 🔁 Live SVG preview
* ⚙️ Full parameter coverage
* 💾 Presets & config persistence
* 🧵 Threaded execution
* 🧮 Numeric fret table output
* 🧰 Export tools for SVG & CSV

This makes it ideal for **professional luthiers, CNC designers, and anyone integrating fretboard design into digital fabrication workflows**.

---


## 🧰 Command Line Arguments

Run:

```
python fretboard_calc.py --help
```

to see the complete list.  
Below is a breakdown of the most important options:

| Argument | Description |
|----------|-------------|
| `--frets N` | Total number of frets (required). |
| `--strings N` | Number of strings (required). |
| `--scale` | Single scale length for all strings. |
| `--bass-scale` / `--treble-scale` | Define bass and treble side scales for multiscale layouts. |
| `--scale-map` | Interpolation mode: `linear` (default) or `exp`. |
| `--scale-gamma` | Exponent for `exp` mapping (biasing). |
| `--unit` | Units: `in` or `mm`. |
| `--neutral-fret` | Index of the neutral fret (default `0`, i.e., nut). |
| `--string-spacing` | Uniform spacing between adjacent strings. |
| `--string-spacing-list` | Comma-separated list of gaps (per string). |
| `--string-spacing-file` | Path to a file containing gaps, one per line. |
| `--nut-width` / `--bridge-width` | Total fretboard width at nut and bridge. |
| `--datum-angle` | Rotation angle (degrees) applied after geometry is computed. |
| `--stroke` | Stroke width in SVG. |
| `--slot-kerf` | Saw kerf width for slot toolpaths. |
| `--emit-slot-offsets` | Emit left/right slot lines offset from fret centers. |
| `--draw-strings` | Include string centerlines in SVG/DXF. |
| `--csv` | Path to write CSV file. |
| `--json` | Path to write JSON file. |
| `--svg` | Path to write SVG drawing. |
| `--dxf` | Path to write DXF drawing. |
| `--markdown` | Force Markdown output even if files are written. |

---

## 💡 Usage Examples

### 1. Basic Single-Scale Output (Markdown Only)

```
python fretboard_calc.py \
  --frets 22 --strings 6 \
  --scale 25.5in --unit in
```

This will:
- Compute 6-string single-scale fret positions
- Print nicely formatted Markdown tables to stdout
- Use 25.5 inches as scale length

---

### 2. Multiscale Output with Datum Rotation and SVG

```
python fretboard_calc.py \
  --frets 24 --strings 8 \
  --bass-scale 27in --treble-scale 25.5in \
  --neutral-fret 7 --string-spacing 0.35in \
  --datum-angle 5 \
  --svg board.svg --draw-strings
```

This will:
- Compute a multiscale board
- Use fret 7 as neutral
- Add 5° rotation
- Output an SVG with strings drawn

---

### 3. DXF with Kerf Slot Toolpaths

```
python fretboard_calc.py \
  --frets 24 --strings 6 \
  --scale 25.5in \
  --nut-width 1.70in --bridge-width 2.20in \
  --slot-kerf 0.023in --emit-slot-offsets \
  --dxf board.dxf
```

This will:
- Generate left/right slot offsets for each fret
- Export a DXF with separate layers for OUTLINE, NUT, BRIDGE, FRETS, SLOTS, and STRINGS.

---

## 🧪 Advanced Features

### Neutral Fret Geometry
All frets are defined relative to the neutral fret, simplifying transformations and toolpath planning.

### Datum Rotation
Datum angle rotates everything about the neutral fret — useful for aligning the design to workholding setups without changing the math.

### Kerf Offsets
For sawed fret slots, the script can emit offset toolpaths corresponding to half the kerf width on either side of each fret.

### String Layout
String positions can be defined with uniform spacing or arbitrary gaps per string, enabling compound layouts.

---

## 📂 Layer Naming (DXF/SVG)

| Layer / Group | Description |
|---------------|-------------|
| `OUTLINE` | Perimeter of the fretboard |
| `NUT` | Nut edge |
| `BRIDGE` | Bridge edge |
| `FRETS` | Center fret lines |
| `SLOTS` | Left and right offset slot toolpaths (if enabled) |
| `STRINGS` | String centerlines (if enabled) |

These layer names are intentionally simple to allow **easy layer-based CAM automation**.

---

## 🛡 Recommended Workflows

- **CAD import (SVG)** — Import into Illustrator, Inkscape, Fusion 360, or FreeCAD to add additional construction features.  
- **CAM import (DXF)** — Import into LightBurn or Fusion to generate toolpaths directly from named layers.  
- **Spreadsheet analysis (CSV)** — Use fret positions in templates for production jigs.  
- **Automation (JSON)** — Integrate with parametric CAD or custom generation pipelines.

---

## ⚠️ Common Pitfalls & Tips

- **Unrecognized arguments**: Check `--help` to make sure your CLI flags match the version of the script you're running.  
- **Units**: Remember that numeric values without suffixes use the global `--unit`.  
- **Neutral fret**: If you rotate around a fret other than 0, the geometry may shift visually in SVG — this is intended.  
- **Datum angle precision**: A few decimal places are fine; extreme angles are allowed but will rotate the entire geometry.  
- **Kerf offsets**: If `--emit-slot-offsets` is set but no `--slot-kerf` is provided, the script will exit with an error.

---

## 🧭 Development Notes

- Written in pure Python 3 — no external dependencies.  
- Designed to run headless (no GUI).  
- Well-structured for extension and integration with other scripts or toolchains.  
- Uses minimal SVG and DXF dialects for maximum interoperability.  
- Fully compatible with version control workflows (outputs are deterministic given the same input).

---

## 🧰 Example Integration

```
import json

with open('board.json') as f:
    data = json.load(f)

print("Fret count:", data["frets"])
for fret in data["fret_geometry"]:
    print(f"Fret {fret['fret']} angle {fret['angle_deg']} degrees")
```

## 🎸 Custom String Spacing Input

You can control the **Y-axis layout** of the strings (distance between each string centerline) using either:
- `--string-spacing` → a single uniform gap applied between all adjacent strings.
- `--string-spacing-list` → a comma-separated list of custom gaps.
- `--string-spacing-file` → a text file containing one gap per line.

---

### 🔹 1. `--string-spacing` (Uniform)
Specifies one value applied between all strings.

Example:
```bash
python3 fretboard_calc_patched.py \
  --frets 24 \
  --strings 6 \
  --scale 25.5in \
  --string-spacing 0.35in \
  --svg board_uniform.svg
```

All 5 inter-string gaps are 0.35 inches.

---

### 🔹 2. `--string-spacing-list` (Inline Custom Values)
Accepts a comma-separated list of S−1 gaps (where S = number of strings).  
Each value may include a unit (`in` or `mm`).

Example:
```bash
python3 fretboard_calc_patched.py \
  --frets 24 \
  --strings 8 \
  --bass-scale 27in \
  --treble-scale 25.5in \
  --neutral-fret 7 \
  --string-spacing-list "0.35in,0.35in,0.36in,0.36in,0.37in,0.37in,0.38in" \
  --svg test_spacing_list.svg \
  --stroke 0.02
```

---

### 🔹 3. `--string-spacing-file` (File-Based Custom Values)
Use a plain text file containing exactly **(strings − 1)** lines.  
Each line defines the gap between one pair of adjacent strings.

Example file: `spacing.txt`
```
0.35in
0.35in
0.36in
0.36in
0.37in
0.37in
0.38in
```

Run:
```bash
python3 fretboard_calc_patched.py \
  --frets 24 \
  --strings 8 \
  --bass-scale 27in \
  --treble-scale 25.5in \
  --neutral-fret 7 \
  --string-spacing-file spacing.txt \
  --svg test_spacing_file.svg \
  --stroke 0.02
```

✅ Expected output:
```
✅ SVG written to test_spacing_file.svg
```

---

### 💡 Tips

| Behavior | Explanation |
|-----------|--------------|
| Blank lines are ignored | Helpful when commenting or spacing visually |
| Units are optional | If omitted, defaults to the unit selected by `--unit` |
| Mixed units allowed | e.g., `9mm` and `0.35in` in the same file are fine |
| Mismatched count errors | You must provide exactly S−1 lines for S strings |
| File encoding | Must be UTF-8 or plain ASCII |

---

### Example Verification

You can confirm spacing was applied correctly by checking the line:
```
**String Y positions (bass=0):** 0.000 | 0.350 | 0.700 | 1.060 | ...
```
printed in the Markdown output or within the JSON (`"string_y_positions"` field).


---

## 📜 License and Use

This script is free to use and modify for personal or professional luthiery and CAD/CAM workflows.  
Attribution is appreciated but not required.  

MIT License 

---

## 🆘 Getting Help

- Run `python fretboard_calc.py --help` for usage.  
- Check for spelling of CLI flags (many changed between versions).  
- Remember to include units where appropriate.  
- If you encounter geometry or import issues, inspect the SVG/DXF in a vector viewer to confirm layer contents.

---

## 🏁 Summary

`fretboard_calc.py` is designed to be a **robust, parametric, precision tool** for fretboard design:

- ✅ Exact 12-TET calculations  
- ✅ Multiscale fan support  
- ✅ Flexible output formats  
- ✅ CNC/CAD ready geometry  
- ✅ Scriptable and automatable

This makes it an excellent choice for **builders who need precision**, **automation pipelines**, and **repeatable manufacturing**.
