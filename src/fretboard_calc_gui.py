#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fretboard_gui.py
================

A professional, threaded PyQt6 GUI wrapper for your `fretboard_calc.py` CLI that provides:

TOP HALF:
    • Zoomable + pannable SVG preview (QGraphicsView + QGraphicsSvgItem).
    • Mouse-wheel zoom centered at cursor; middle-mouse pan (or Space + left-drag).
    • Zoom toolbar: Zoom In / Zoom Out / 100% / Fit; live zoom % readout.
    • "Export SVG" button to copy the current preview SVG to a user-chosen path.

BOTTOM HALF:
    • Left: ALL CLI parameters as GUI controls (every flag mapped, with tooltips).
      - Geometry: frets, strings, scale/s, scale-map, scale-gamma, neutral-fret.
      - Spacing: uniform OR comma-list OR file (mutually exclusive).
      - Board: widths, margins, datum angle, stroke width, draw-strings.
      - Normalization: --normalize-scale, --normalize-mode, --target-bridge-angle.
      - Kerf/toolpaths: --slot-kerf, --emit-slot-offsets.
      - Outputs: CSV/JSON/DXF toggles and output directory.
    • Presets row: choose, apply, save/update, delete. Stored in ~/.fretboard_gui_presets.json
    • Right: Numeric table (positions + spacings) populated from the temp JSON the CLI writes.
    • Bottom-right: Log / Summary (captures stdout/stderr; shows normalization summary).

SYSTEM:
    • Threaded: runs the CLI in a QThread so the UI is never blocked.
    • Debounced: regenerates ~300 ms after user stops typing/toggling controls.
    • Persistent config: window geometry, last output dir, last preset, auto-preview,
      saved in ~/.fretboard_gui_config.json

SETUP:
    pip install PyQt6 PyQt6-QtSvg
    python3 fretboard_gui.py

IMPORTANT:
    Keep this file in the same directory as your working `fretboard_calc.py`.
    The GUI *always* asks the CLI to emit a temp SVG + temp JSON for preview + table.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import (
    QByteArray,
    QObject,
    QPointF,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
)
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtSvgWidgets import QGraphicsSvgItem

# SVG rendering support (QGraphicsScene/QGraphicsView + QGraphicsSvgItem)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# -------------------------------------------------------------------------------------------------
# Disk locations for user-level persistence (presets + config)
# -------------------------------------------------------------------------------------------------

PRESETS_PATH = Path.home() / ".fretboard_gui_presets.json"
CONFIG_PATH = Path.home() / ".fretboard_gui_config.json"


# -------------------------------------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------------------------------------


def which_python() -> str:
    """
    Return the most reliable Python executable to use for launching the CLI.
    We prefer the currently running interpreter for virtualenv/conda correctness.
    """
    return sys.executable or shutil.which("python3") or "python3"


def calc_script_path() -> Path:
    """
    Resolve the local fretboard_calc.py path. Assumes this GUI lives next to it.
    """
    here = Path(__file__).resolve().parent
    return here / "fretboard_calc.py"


def safe_text(s: str) -> str:
    """
    Trim and return a string; tolerates None. Used for unit-suffixed numeric text
    and standard integer fields. The CLI parses units for numeric fields like "25.5in".
    """
    return (s or "").strip()


def load_json_file(path: Path, default: Any) -> Any:
    """
    Read a JSON file if it exists; otherwise return `default`.
    """
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json_file(path: Path, data: Any) -> None:
    """
    Write a JSON file atomically (write-then-replace) for safety.
    """
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
    except Exception:
        # If anything fails, we don't crash the GUI; user keeps working.
        pass


# -------------------------------------------------------------------------------------------------
# Zoom + Pan SVG View: mouse-wheel zoom, middle-mouse pan, Space+left pan, toolbar hooks
# -------------------------------------------------------------------------------------------------


class ZoomPanSvgView(QGraphicsView):
    """
    A QGraphicsView subclass with:
      • Scene hosting an SVG item (QGraphicsSvgItem bound to QSvgRenderer)
      • Wheel zoom centered at cursor
      • Middle mouse pan (or Space + left-drag)
      • Fit-to-view and 100% zoom helpers
      • A `zoomChanged` signal so parent can update a UI label with zoom %

    We treat current transform scaling as our zoom factor and avoid surprise anchors.
    """

    zoomChanged = pyqtSignal(float)  # Emits current zoom factor (1.0 == 100%)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Scene + item + renderer
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._renderer: Optional[QSvgRenderer] = None
        self._svg_item: Optional[QGraphicsSvgItem] = None

        # Interaction and render prefs
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)

        # Pan state
        self._panning = False
        self._pan_start = None
        self._space_down = False

        # Logical zoom factor (1.0 => 100%)
        self._zoom = 1.0

    # ------------------ SVG loading and fitting ------------------

    def clear_svg(self):
        """Remove any existing SVG item and reset the view/zoom."""
        if self._svg_item:
            self._scene.removeItem(self._svg_item)
            self._svg_item = None
        self._renderer = None
        self._scene.setSceneRect(0, 0, 0, 0)
        self.resetTransform()
        self._zoom = 1.0
        self.zoomChanged.emit(self._zoom)

    def load_svg_file(self, filepath: str) -> bool:
        """
        Load an SVG file into the scene. Returns True on success, False otherwise.
        """
        try:
            self.clear_svg()
            self._renderer = QSvgRenderer(filepath)
            if not self._renderer.isValid():
                return False
            self._svg_item = QGraphicsSvgItem()
            self._svg_item.setSharedRenderer(self._renderer)
            self._svg_item.setFlags(
                QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            )
            self._scene.addItem(self._svg_item)

            # Resize scene rect to match the SVG's viewBox
            bounds = self._renderer.viewBoxF()
            self._scene.setSceneRect(bounds)

            # Fit the view initially (padding ~5%)
            self.fit_to_view(padding=0.05)
            return True
        except Exception:
            return False

    def fit_to_view(self, padding: float = 0.05):
        """
        Fit the current SVG fully into the viewport with a small padding margin.
        Resets zoom to 1.0 logical factor (we treat the fitted state as 1x).
        """
        if not self._svg_item:
            return
        rect = self._scene.sceneRect()
        if rect.isNull():
            return
        pad_x = rect.width() * padding
        pad_y = rect.height() * padding
        rect_padded = rect.adjusted(-pad_x, -pad_y, pad_x, pad_y)
        self.resetTransform()
        super().fitInView(rect_padded, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 1.0
        self.zoomChanged.emit(self._zoom)

    def set_zoom_100(self):
        """
        Reset to exactly 100% logical zoom (1.0 in our model), centered.
        """
        if not self._svg_item:
            return
        self.resetTransform()
        self._zoom = 1.0
        self.centerOn(self._scene.sceneRect().center())
        self.zoomChanged.emit(self._zoom)

    def zoom_in(self):
        self._apply_zoom(1.15)

    def zoom_out(self):
        self._apply_zoom(1.0 / 1.15)

    def _apply_zoom(self, factor: float, center_pos: Optional[QPointF] = None):
        """
        Apply a scale factor around a given scene position (default = view center).
        """
        if not self._svg_item:
            return
        if center_pos is None:
            center_pos = self.mapToScene(self.viewport().rect().center())
        old_pos = center_pos
        self.scale(factor, factor)
        new_pos = center_pos
        # Keep the chosen point visually fixed by translating the view
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        self._zoom *= factor
        self.zoomChanged.emit(self._zoom)

    # ------------------ Mouse / keyboard interaction ------------------

    def wheelEvent(self, event):
        """
        Zoom at the cursor using the wheel. Simple and predictable for CAD-like usage.
        """
        if not self._svg_item:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
        scene_pos = self.mapToScene(event.position().toPoint())
        self._apply_zoom(factor, center_pos=scene_pos)

    def mousePressEvent(self, event):
        """
        Enable panning via middle mouse button, or Space + left button.
        """
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space_down
        ):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Translate the view while panning is active.
        """
        if self._panning and self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.translate(delta.x() * -1.0, delta.y() * -1.0)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Stop panning when user releases middle/left (if Space mode was active).
        """
        if self._panning and (
            event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton)
        ):
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """
        Hold Space to pan with left mouse button (ergonomic alternative).
        """
        if event.key() == Qt.Key.Key_Space:
            self._space_down = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """
        Release Space to exit that panning mode.
        """
        if event.key() == Qt.Key.Key_Space:
            self._space_down = False
            if not self._panning:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().keyReleaseEvent(event)


# -------------------------------------------------------------------------------------------------
# Worker object running the CLI in a background thread (non-blocking UI)
# -------------------------------------------------------------------------------------------------


class CalcWorker(QObject):
    """
    A QObject used in a QThread to run `fretboard_calc.py` with a constructed command.

    Signals:
        started(list[str]): emitted before running for logging purposes.
        finished(svg_path, json_path, stdout, stderr): emitted after process completes.

    We capture stdout/stderr (text), and return the temp file paths where SVG/JSON were requested.
    """

    started = pyqtSignal(list)
    finished = pyqtSignal(str, str, str, str)

    def __init__(
        self,
        cmd: List[str],
        svg_path: str,
        json_path: str,
        workdir: Optional[str] = None,
    ):
        super().__init__()
        self.cmd = cmd
        self.svg_path = svg_path
        self.json_path = json_path
        self.workdir = workdir

    def run(self):
        """
        Execute the CLI using subprocess.run with capture_output=True, then emit results.
        """
        try:
            self.started.emit(self.cmd)
            proc = subprocess.run(
                self.cmd,
                cwd=self.workdir,
                text=True,
                capture_output=True,
            )
            out = proc.stdout or ""
            err = proc.stderr or ""
            self.finished.emit(self.svg_path, self.json_path, out, err)
        except Exception as e:
            self.finished.emit(
                self.svg_path, self.json_path, "", f"[GUI] Subprocess error: {e!r}"
            )


# -------------------------------------------------------------------------------------------------
# Main GUI Window
# -------------------------------------------------------------------------------------------------


class FretboardGUI(QWidget):
    """
    The main window builds a single, unified layout:

      TOP:
        • ToolBar (zoom controls + zoom % label + Export SVG)
        • ZoomPanSvgView (preview)

      BOTTOM (split horizontally):
        • LEFT:  Controls group boxes covering all CLI args + Presets manager
        • RIGHT: Numeric Table (from temp JSON) + Log/Summary panel
    """

    # ------------------------------ Construction ------------------------------

    def __init__(self):
        super().__init__()

        # ---------- Window meta ----------
        self.setWindowTitle(
            "Fretboard Calculator — Live GUI (Zoom/Pan + Presets + Full Controls)"
        )
        self.resize(1360, 920)

        # ---------- Engine script path ----------
        self._calc_script = calc_script_path()
        if not self._calc_script.exists():
            raise SystemExit(
                f"ERROR: Could not find fretboard_calc.py at: {self._calc_script}"
            )

        # ---------- Temporary artifacts ----------
        # We use a temp directory for preview assets (SVG + JSON) to avoid cluttering user dirs.
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="fretboard_gui_")
        self._tmp_svg = str(Path(self._tmp_dir.name) / "preview.svg")
        self._tmp_json = str(Path(self._tmp_dir.name) / "preview.json")

        # ---------- Threading + debounce control ----------
        self._thread: Optional[QThread] = None
        self._worker: Optional[CalcWorker] = None
        self._regen_timer = QTimer(self)
        self._regen_timer.setSingleShot(True)
        self._regen_timer.timeout.connect(self._regenerate_now)

        # ---------- Persistent state (presets + config) ----------
        self._presets: Dict[str, Dict[str, str]] = self._load_presets()
        self._config: Dict[str, Any] = self._load_config()

        # ---------- Build UI ----------
        self._build_ui()

        # ---------- Restore persisted UI state ----------
        self._restore_config()

        # ---------- Auto-preview initial run ----------
        QTimer.singleShot(150, self.queue_regeneration)

        # ------------------------------ UI Build ------------------------------

    def _build_ui(self):
        """
        Build the entire window:
          - Top toolbar + SVG preview
          - Bottom split: left controls + right numeric table + log
        """
        outer = QVBoxLayout(self)

        # ---------------- FIRST: Create the SVG view so toolbar can safely reference it ----------------
        self.svg_view = ZoomPanSvgView(self)
        self.svg_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.svg_view.zoomChanged.connect(self._on_zoom_changed)

        # ---------------- SECOND: Create the toolbar (now svg_view exists) ----------------
        self.toolbar = QToolBar("Preview Controls", self)
        self._build_toolbar(self.toolbar)
        outer.addWidget(self.toolbar)

        # Add the SVG preview below the toolbar
        outer.addWidget(self.svg_view, stretch=1)

        # ---------------- BOTTOM: Split horizontally (Left controls | Right table+log) ----------------
        bottom = QSplitter(Qt.Orientation.Horizontal, self)
        outer.addWidget(bottom, stretch=1)

        # LEFT column: full controls + presets
        left_container = QWidget(self)
        left_layout = QVBoxLayout(left_container)
        left_layout.setSpacing(8)

        left_layout.addWidget(self._build_presets_group())
        left_layout.addWidget(self._build_parameters_group())
        left_layout.addWidget(self._build_outputs_group())
        left_layout.addWidget(self._build_actions_row())

        bottom.addWidget(left_container)

        # RIGHT column: numeric table + log
        right_container = QWidget(self)
        right_layout = QVBoxLayout(right_container)
        right_layout.setSpacing(8)

        # Numeric table label + widget
        title_table = QLabel("Fret Table (Nut-to-Fret Positions & Per-Fret Spacings)")
        title_table.setStyleSheet("font-weight:600;")
        right_layout.addWidget(title_table)

        self.table = QTableWidget(self)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        right_layout.addWidget(self.table, stretch=1)

        # Log label + widget
        title_log = QLabel("Log / Summary")
        title_log.setStyleSheet("font-weight:600;")
        right_layout.addWidget(title_log)

        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        right_layout.addWidget(self.log)

        bottom.addWidget(right_container)
        # Initial split sizes: give the table/log column generous width
        bottom.setSizes([740, 620])

    def _build_toolbar(self, tb: QToolBar):
        """
        Construct the top toolbar with:
          - Zoom In / Zoom Out / 100% / Fit buttons
          - Zoom % label
          - Export SVG button
        """
        tb.setMovable(False)

        # Zoom In
        act_zoom_in = QAction("Zoom In", self)
        act_zoom_in.triggered.connect(self.svg_view.zoom_in)
        tb.addAction(act_zoom_in)

        # Zoom Out
        act_zoom_out = QAction("Zoom Out", self)
        act_zoom_out.triggered.connect(self.svg_view.zoom_out)
        tb.addAction(act_zoom_out)

        # 100%
        act_zoom_100 = QAction("100%", self)
        act_zoom_100.triggered.connect(self.svg_view.set_zoom_100)
        tb.addAction(act_zoom_100)

        # Fit
        act_fit = QAction("Fit", self)
        act_fit.triggered.connect(lambda: self.svg_view.fit_to_view(0.05))
        tb.addAction(act_fit)

        tb.addSeparator()

        # Live Zoom % readout
        self.lbl_zoom = QLabel("Zoom: 100%")
        tb.addWidget(self.lbl_zoom)

        tb.addSeparator()

        # Export SVG
        act_export_svg = QAction("Export SVG…", self)
        act_export_svg.triggered.connect(self._on_export_svg)
        tb.addAction(act_export_svg)

        # Export Table to CSV
        act_export_csv = QAction("Export Table CSV…", self)
        act_export_csv.triggered.connect(self._on_export_table_csv)
        tb.addAction(act_export_csv)

    # ------------------------------ Presets ------------------------------

    def _build_presets_group(self) -> QGroupBox:
        """
        A compact presets manager row:
          - Preset chooser (combo)
          - Apply
          - Save/Update (writes current fields under the chosen name)
          - Delete
        Presets persist in ~/.fretboard_gui_presets.json
        """
        g = QGroupBox("Presets")
        grid = QGridLayout(g)
        r = 0

        self.cb_preset = QComboBox()
        # Load existing presets (if none, provide some useful defaults)
        if not self._presets:
            self._presets.update(self._default_presets())
            save_json_file(PRESETS_PATH, self._presets)
        self._refresh_preset_combo()

        btn_apply = QPushButton("Apply")
        btn_apply.setToolTip("Apply the selected preset values to the fields below.")
        btn_apply.clicked.connect(self._on_apply_preset)

        btn_save = QPushButton("Save/Update")
        btn_save.setToolTip(
            "Save current field values under this preset name (create or overwrite)."
        )
        btn_save.clicked.connect(self._on_save_preset)

        btn_delete = QPushButton("Delete")
        btn_delete.setToolTip("Delete the selected preset from disk.")
        btn_delete.clicked.connect(self._on_delete_preset)

        # Layout row
        grid.addWidget(QLabel("Preset:"), r, 0)
        grid.addWidget(self.cb_preset, r, 1, 1, 2)
        grid.addWidget(btn_apply, r, 3)
        grid.addWidget(btn_save, r, 4)
        grid.addWidget(btn_delete, r, 5)
        r += 1

        return g

    def _default_presets(self) -> Dict[str, Dict[str, str]]:
        """
        A handful of practical starter presets. Every key here maps to a GUI field name.
        You can extend freely.
        """
        return {
            "None (manual)": {},
            '6-String Strat (25.5")': {
                "frets": "22",
                "strings": "6",
                "scale": "25.5in",
                "nut_width": "1.650in",
                "bridge_width": "2.200in",
                "string_spacing": "0.350in",
                "neutral_fret": "0",
                "unit": "in",
                "decimals": "3",
                "draw_strings": "1",
                "datum_angle": "0",
                "stroke": "0.02",
                "board_margin": "0.10in",
            },
            '7-String Fan (26.5 → 25.5")': {
                "frets": "24",
                "strings": "7",
                "bass_scale": "26.5in",
                "treble_scale": "25.5in",
                "nut_width": "1.900in",
                "bridge_width": "2.450in",
                "string_spacing": "0.350in",
                "neutral_fret": "7",
                "unit": "in",
                "decimals": "3",
                "draw_strings": "1",
                "datum_angle": "0",
                "stroke": "0.02",
                "board_margin": "0.10in",
                "normalize": "0",
                "normalize_mode": "treble",
                "target_bridge_angle": "0.5",
            },
            '8-String Fan (27 → 25.5")': {
                "frets": "24",
                "strings": "8",
                "bass_scale": "27in",
                "treble_scale": "25.5in",
                "nut_width": "2.10in",
                "bridge_width": "2.60in",
                "string_spacing": "0.350in",
                "neutral_fret": "7",
                "unit": "in",
                "decimals": "3",
                "draw_strings": "1",
                "datum_angle": "0",
                "stroke": "0.02",
                "board_margin": "0.10in",
                "normalize": "1",
                "normalize_mode": "treble",
                "target_bridge_angle": "1.0",
            },
            'Baritone 6 (27")': {
                "frets": "24",
                "strings": "6",
                "scale": "27in",
                "nut_width": "1.700in",
                "bridge_width": "2.250in",
                "string_spacing": "0.355in",
                "neutral_fret": "0",
                "unit": "in",
                "decimals": "3",
                "draw_strings": "1",
                "datum_angle": "0",
                "stroke": "0.02",
                "board_margin": "0.10in",
            },
        }

    def _refresh_preset_combo(self):
        """
        Refresh the preset combo contents from self._presets (name -> dict).
        """
        self.cb_preset.blockSignals(True)
        self.cb_preset.clear()
        self.cb_preset.addItems(sorted(self._presets.keys()))
        # If config remembered a last selection, restore it
        last = self._config.get("last_preset", "None (manual)")
        idx = self.cb_preset.findText(last)
        if idx >= 0:
            self.cb_preset.setCurrentIndex(idx)
        self.cb_preset.blockSignals(False)

    # ------------------------------ Parameters ------------------------------

    def _build_parameters_group(self) -> QGroupBox:
        """
        Build the full parameters panel, **including every CLI argument** from your script.
        Organized into logical categories with labels, and heavily commented.
        """
        g = QGroupBox("Parameters (All CLI Arguments)")
        grid = QGridLayout(g)
        r = 0

        # --- Core counts ---
        self.in_frets = QLineEdit("24")
        self.in_frets.setToolTip("--frets")
        self.in_strings = QLineEdit("8")
        self.in_strings.setToolTip("--strings")
        grid.addWidget(QLabel("Frets:"), r, 0)
        grid.addWidget(self.in_frets, r, 1)
        grid.addWidget(QLabel("Strings:"), r, 2)
        grid.addWidget(self.in_strings, r, 3)
        r += 1

        # --- Scales: single or multi (favor multiscale if bass+treble provided) ---
        self.in_scale = QLineEdit("")
        self.in_scale.setToolTip('--scale (e.g., "25.5in")')
        self.in_bass_scale = QLineEdit("27in")
        self.in_bass_scale.setToolTip('--bass-scale (e.g., "27in")')
        self.in_treble_scale = QLineEdit("25.5in")
        self.in_treble_scale.setToolTip('--treble-scale (e.g., "25.5in")')
        grid.addWidget(QLabel("Single Scale (opt):"), r, 0)
        grid.addWidget(self.in_scale, r, 1)
        grid.addWidget(QLabel("Bass Scale:"), r, 2)
        grid.addWidget(self.in_bass_scale, r, 3)
        r += 1
        grid.addWidget(QLabel("Treble Scale:"), r, 2)
        grid.addWidget(self.in_treble_scale, r, 3)
        r += 1

        # --- Scale mapping (linear or exponential) ---
        self.cb_scale_map = QComboBox()
        self.cb_scale_map.addItems(["linear", "exp"])
        self.cb_scale_map.setToolTip("--scale-map {linear,exp}")
        self.in_scale_gamma = QLineEdit("1.0")
        self.in_scale_gamma.setToolTip("--scale-gamma (for exp)")
        grid.addWidget(QLabel("Scale Map:"), r, 0)
        grid.addWidget(self.cb_scale_map, r, 1)
        grid.addWidget(QLabel("Scale Gamma:"), r, 2)
        grid.addWidget(self.in_scale_gamma, r, 3)
        r += 1

        # --- Units / decimals / neutral fret ---
        self.cb_unit = QComboBox()
        self.cb_unit.addItems(["in", "mm"])
        self.cb_unit.setToolTip("--unit")
        self.in_decimals = QLineEdit("3")
        self.in_decimals.setToolTip("--decimals")
        self.in_neutral = QLineEdit("7")
        self.in_neutral.setToolTip("--neutral-fret")
        grid.addWidget(QLabel("Unit:"), r, 0)
        grid.addWidget(self.cb_unit, r, 1)
        grid.addWidget(QLabel("Decimals:"), r, 2)
        grid.addWidget(self.in_decimals, r, 3)
        r += 1
        grid.addWidget(QLabel("Neutral Fret:"), r, 0)
        grid.addWidget(self.in_neutral, r, 1)
        r += 1

        # --- String spacing (mutually exclusive: uniform OR list OR file) ---
        self.rb_spacing_uniform = QRadioButton("Uniform")
        self.rb_spacing_uniform.setChecked(True)
        self.rb_spacing_list = QRadioButton("List CSV")
        self.rb_spacing_file = QRadioButton("From File")
        grid.addWidget(self.rb_spacing_uniform, r, 0)
        grid.addWidget(self.rb_spacing_list, r, 1)
        grid.addWidget(self.rb_spacing_file, r, 2)
        r += 1

        self.in_spacing_uniform = QLineEdit("0.35in")
        self.in_spacing_uniform.setToolTip("--string-spacing")
        self.in_spacing_list = QLineEdit("")
        self.in_spacing_list.setToolTip("--string-spacing-list (CSV of S-1 gaps)")
        self.in_spacing_file = QLineEdit("")
        self.in_spacing_file.setToolTip("--string-spacing-file (path)")
        btn_spacing_browse = QPushButton("Browse…")
        btn_spacing_browse.setToolTip("Pick spacing list file (one gap per line)")
        btn_spacing_browse.clicked.connect(self._on_browse_spacing_file)

        grid.addWidget(QLabel("Uniform Gap:"), r, 0)
        grid.addWidget(self.in_spacing_uniform, r, 1)
        grid.addWidget(QLabel("List Gaps CSV:"), r, 2)
        grid.addWidget(self.in_spacing_list, r, 3)
        r += 1
        grid.addWidget(QLabel("From File:"), r, 0)
        grid.addWidget(self.in_spacing_file, r, 1, 1, 2)
        grid.addWidget(btn_spacing_browse, r, 3)
        r += 1

        # --- Board geometry: widths + margins, datum angle, stroke, draw-strings ---
        self.in_nut_width = QLineEdit("2.10in")
        self.in_nut_width.setToolTip("--nut-width")
        self.in_bridge_width = QLineEdit("2.60in")
        self.in_bridge_width.setToolTip("--bridge-width")
        self.in_board_margin = QLineEdit("0.10in")
        self.in_board_margin.setToolTip("--board-margin (Y)")
        self.in_board_margin_x = QLineEdit("")
        self.in_board_margin_x.setToolTip("--board-margin-x (X, optional)")
        grid.addWidget(QLabel("Nut Width:"), r, 0)
        grid.addWidget(self.in_nut_width, r, 1)
        grid.addWidget(QLabel("Bridge Width:"), r, 2)
        grid.addWidget(self.in_bridge_width, r, 3)
        r += 1
        grid.addWidget(QLabel("Board Margin Y:"), r, 0)
        grid.addWidget(self.in_board_margin, r, 1)
        grid.addWidget(QLabel("Board Margin X (opt):"), r, 2)
        grid.addWidget(self.in_board_margin_x, r, 3)
        r += 1

        self.in_datum_angle = QLineEdit("0")
        self.in_datum_angle.setToolTip("--datum-angle (deg)")
        self.in_stroke = QLineEdit("0.02")
        self.in_stroke.setToolTip("--stroke")
        self.chk_draw_strings = QCheckBox("Draw Strings")
        self.chk_draw_strings.setChecked(True)
        self.chk_draw_strings.setToolTip("--draw-strings")
        grid.addWidget(QLabel("Datum Angle (deg):"), r, 0)
        grid.addWidget(self.in_datum_angle, r, 1)
        grid.addWidget(QLabel("SVG Stroke Width:"), r, 2)
        grid.addWidget(self.in_stroke, r, 3)
        r += 1
        grid.addWidget(self.chk_draw_strings, r, 0, 1, 2)
        r += 1

        # --- Slot kerf + emit slot offsets ---
        self.in_slot_kerf = QLineEdit("")
        self.in_slot_kerf.setToolTip('--slot-kerf (e.g., "0.023in")')
        self.chk_emit_slot_offsets = QCheckBox("Emit Slot Offsets")
        self.chk_emit_slot_offsets.setToolTip(
            "--emit-slot-offsets (requires --slot-kerf)"
        )
        grid.addWidget(QLabel("Slot Kerf (opt):"), r, 0)
        grid.addWidget(self.in_slot_kerf, r, 1)
        grid.addWidget(self.chk_emit_slot_offsets, r, 2, 1, 2)
        r += 1

        # --- Normalization (relative bridge Δ angle) ---
        self.chk_normalize = QCheckBox("Normalize Bridge Δ Angle (relative)")
        self.chk_normalize.setToolTip("--normalize-scale")
        self.cb_normalize_mode = QComboBox()
        self.cb_normalize_mode.addItems(["treble", "both"])
        self.cb_normalize_mode.setToolTip("--normalize-mode")
        self.in_target_bridge_angle = QLineEdit("1.0")
        self.in_target_bridge_angle.setToolTip("--target-bridge-angle (deg)")
        grid.addWidget(self.chk_normalize, r, 0, 1, 2)
        grid.addWidget(QLabel("Normalize Mode:"), r, 2)
        grid.addWidget(self.cb_normalize_mode, r, 3)
        r += 1
        grid.addWidget(QLabel("Δ Bridge Angle (deg):"), r, 0)
        grid.addWidget(self.in_target_bridge_angle, r, 1)
        r += 1

        # ----------------------------------------------------------------------
        # Connect all relevant parameter widgets to the regeneration queue.
        # This allows the fretboard to auto-update whenever a parameter changes.
        # We check each widget type to connect the correct Qt signal:
        #   • QLineEdit → textChanged
        #   • QComboBox → currentIndexChanged
        #   • QCheckBox → stateChanged
        #   • QRadioButton → toggled(bool)
        # ----------------------------------------------------------------------
        for w in [
            self.in_frets,
            self.in_strings,
            self.in_scale,
            self.in_bass_scale,
            self.in_treble_scale,
            self.in_neutral,
            self.cb_unit,
            self.in_decimals,
            self.in_spacing_uniform,
            self.in_spacing_list,
            self.in_spacing_file,
            self.in_nut_width,
            self.in_bridge_width,
            self.in_board_margin,
            self.in_board_margin_x,
            self.in_target_bridge_angle,
            self.cb_normalize_mode,
            self.in_datum_angle,
            self.in_stroke,
            self.chk_draw_strings,
            self.chk_normalize,
            self.chk_emit_slot_offsets,
            self.cb_scale_map,
        ]:
            # Text inputs
            if isinstance(w, QLineEdit):
                w.textChanged.connect(self.queue_regeneration)
            # Drop-down menus
            elif isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self.queue_regeneration)
            # Checkboxes
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(self.queue_regeneration)
            # Radio buttons (e.g., spacing mode or others)
            elif hasattr(w, "toggled"):
                w.toggled.connect(self.queue_regeneration)

        return g

    # ------------------------------ Outputs Group ------------------------------

    def _build_outputs_group(self) -> QGroupBox:
        """
        Output settings:
          - Output Directory (for optional CSV/JSON/DXF exports)
          - Checkboxes to enable CSV/JSON/DXF
        The preview always writes a temp SVG + temp JSON for the GUI’s own usage.
        """
        g = QGroupBox("Outputs")
        grid = QGridLayout(g)
        r = 0

        self.in_out_dir = QLineEdit(str(Path.cwd()))
        self.in_out_dir.setToolTip("Destination for CSV/JSON/DXF when enabled below.")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._on_browse_dir)

        grid.addWidget(QLabel("Output Directory:"), r, 0)
        grid.addWidget(self.in_out_dir, r, 1, 1, 2)
        grid.addWidget(btn_browse, r, 3)
        r += 1

        self.chk_csv = QCheckBox("Write CSV")
        self.chk_csv.setChecked(False)
        self.chk_csv.setToolTip("--csv <path>")
        self.chk_json = QCheckBox("Write JSON")
        self.chk_json.setChecked(True)
        self.chk_json.setToolTip("--json <path>")
        self.chk_dxf = QCheckBox("Write DXF")
        self.chk_dxf.setChecked(False)
        self.chk_dxf.setToolTip("--dxf <path>")

        grid.addWidget(self.chk_csv, r, 0)
        grid.addWidget(self.chk_json, r, 1)
        grid.addWidget(self.chk_dxf, r, 2)
        r += 1

        # Any change here should also regenerate
        self.in_out_dir.textChanged.connect(self.queue_regeneration)
        self.chk_csv.stateChanged.connect(self.queue_regeneration)
        self.chk_json.stateChanged.connect(self.queue_regeneration)
        self.chk_dxf.stateChanged.connect(self.queue_regeneration)

        return g

    # ------------------------------ Actions Row ------------------------------

    def _build_actions_row(self) -> QWidget:
        """
        Bottom row of the left column: manual Generate, Auto Preview toggle, Fit, Clear Log.
        """
        row = QWidget(self)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)

        self.btn_generate = QPushButton("Generate Now")
        self.btn_generate.setToolTip(
            "Run the CLI immediately with current settings (threaded)."
        )
        self.btn_generate.clicked.connect(self._regenerate_now)

        self.btn_fit = QPushButton("Fit Preview")
        self.btn_fit.setToolTip("Fit the current SVG to the preview window.")
        self.btn_fit.clicked.connect(lambda: self.svg_view.fit_to_view(0.05))

        self.btn_auto = QPushButton("Auto Preview: ON")
        self.btn_auto.setCheckable(True)
        self.btn_auto.setChecked(True)
        self.btn_auto.setToolTip("Toggle automatic regeneration after changes.")
        self.btn_auto.toggled.connect(self._toggle_auto)

        self.btn_clear = QPushButton("Clear Log")
        self.btn_clear.clicked.connect(lambda: self.log.clear())

        h.addWidget(self.btn_generate)
        h.addStretch(1)
        h.addWidget(self.btn_fit)
        h.addWidget(self.btn_auto)
        h.addWidget(self.btn_clear)
        return row

    # ------------------------------ Event Wiring + Debounce ------------------------------

    def _toggle_auto(self, on: bool):
        self.btn_auto.setText(f"Auto Preview: {'ON' if on else 'OFF'}")
        self._config["auto_preview"] = bool(on)
        self._save_config()
        if on:
            self.queue_regeneration()

    def queue_regeneration(self):
        """
        Debounce user input. If Auto Preview is ON, queue a regeneration in ~300 ms.
        Each additional edit resets the timer, so we only run after user pauses.
        """
        if not self._config.get("auto_preview", True):
            return
        self._regen_timer.start(300)

    # ------------------------------ Command Construction ------------------------------

    def _build_cmd(self) -> Tuple[List[str], str, str]:
        """
        Construct the CLI command list and return:
            (cmd, temp_svg_path, temp_json_path)

        We always include:
          --svg <temp.svg>  (for preview)
          --json <temp.json> (for numeric table)
        """
        py = which_python()
        calc = str(self._calc_script)

        out_dir = Path(self.in_out_dir.text()).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [
            py,
            calc,
            "--frets",
            safe_text(self.in_frets.text()),
            "--strings",
            safe_text(self.in_strings.text()),
            "--unit",
            self.cb_unit.currentText(),
            "--decimals",
            safe_text(self.in_decimals.text()),
            "--neutral-fret",
            safe_text(self.in_neutral.text()),
            "--board-margin",
            safe_text(self.in_board_margin.text()),
            "--nut-width",
            safe_text(self.in_nut_width.text()),
            "--bridge-width",
            safe_text(self.in_bridge_width.text()),
            "--datum-angle",
            safe_text(self.in_datum_angle.text()),
            "--stroke",
            safe_text(self.in_stroke.text()),
            "--svg",
            self._tmp_svg,  # preview artifact for the view
            "--json",
            self._tmp_json,  # preview artifact for the table
            "--scale-map",
            self.cb_scale_map.currentText(),
            "--scale-gamma",
            safe_text(self.in_scale_gamma.text()),
        ]

        # --- Scale selection: multiscale (bass+treble) OR single scale ---
        scale = safe_text(self.in_scale.text())
        bass = safe_text(self.in_bass_scale.text())
        treb = safe_text(self.in_treble_scale.text())
        if bass and treb:
            cmd += ["--bass-scale", bass, "--treble-scale", treb]
        elif scale:
            cmd += ["--scale", scale]
        else:
            # Fallback to avoid CLI exit if fields are blank
            cmd += ["--bass-scale", "25.5in", "--treble-scale", "25.5in"]

        # --- String spacing mode (mutually exclusive) ---
        if self.rb_spacing_uniform.isChecked():
            val = safe_text(self.in_spacing_uniform.text())
            if val:
                cmd += ["--string-spacing", val]
        elif self.rb_spacing_list.isChecked():
            val = safe_text(self.in_spacing_list.text())
            if val:
                cmd += ["--string-spacing-list", val]
        elif self.rb_spacing_file.isChecked():
            val = safe_text(self.in_spacing_file.text())
            if val:
                cmd += ["--string-spacing-file", val]

        # Optional horizontal margin for strings (board-margin-x)
        if safe_text(self.in_board_margin_x.text()):
            cmd += ["--board-margin-x", safe_text(self.in_board_margin_x.text())]

        # Draw strings
        if self.chk_draw_strings.isChecked():
            cmd += ["--draw-strings"]

        # Slot kerf and SLOTS layer offsets
        if safe_text(self.in_slot_kerf.text()):
            cmd += ["--slot-kerf", safe_text(self.in_slot_kerf.text())]
        if self.chk_emit_slot_offsets.isChecked():
            cmd += ["--emit-slot-offsets"]

        # Normalization (relative bridge Δ angle)
        if self.chk_normalize.isChecked():
            cmd += [
                "--normalize-scale",
                "--normalize-mode",
                self.cb_normalize_mode.currentText(),
            ]
            targ = safe_text(self.in_target_bridge_angle.text())
            if targ:
                cmd += ["--target-bridge-angle", targ]

        # Optional file exports to OUT_DIR
        if self.chk_csv.isChecked():
            cmd += ["--csv", str(out_dir / "board.csv")]
        if self.chk_json.isChecked():
            cmd += ["--json", str(out_dir / "board.json")]
        if self.chk_dxf.isChecked():
            cmd += ["--dxf", str(out_dir / "board.dxf")]

        return cmd, self._tmp_svg, self._tmp_json

    # ------------------------------ Run (threaded) ------------------------------

    def _regenerate_now(self):
        """
        Start a background CLI run with current parameters. If a run is already
        active, we simply ignore (debounce will trigger another run soon).
        """
        if self._thread is not None:
            return

        cmd, tmp_svg, tmp_json = self._build_cmd()

        self.btn_generate.setEnabled(False)
        self.log.append(f"▶ Running: {' '.join(cmd)}\n")

        self._thread = QThread(self)
        self._worker = CalcWorker(cmd=cmd, svg_path=tmp_svg, json_path=tmp_json)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.started.connect(
            lambda _cmd: None
        )  # reserved for spinner/status hooks
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    # ------------------------------ Results handling ------------------------------

    def _on_worker_finished(
        self, svg_path: str, json_path: str, stdout: str, stderr: str
    ):
        """
        Called when the CLI completes. We:
          • Append stdout/stderr to log (includes normalization summaries).
          • Load SVG into the zoom/pan view.
          • Load JSON and rebuild the numeric table.
        """
        if stdout.strip():
            self.log.append(stdout.strip() + "\n")
        if stderr.strip():
            self.log.append(f"⚠️ stderr:\n{stderr.strip()}\n")

        # Load SVG
        if Path(svg_path).exists() and Path(svg_path).stat().st_size > 0:
            ok = self.svg_view.load_svg_file(svg_path)
            if not ok:
                self.log.append("⚠️ Failed to render SVG (invalid or unsupported).\n")
        else:
            self.svg_view.clear_svg()
            self.log.append("⚠️ No SVG generated (file missing or empty).\n")

        # Load JSON -> update table
        if Path(json_path).exists() and Path(json_path).stat().st_size > 0:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._populate_table(data)
            except Exception as e:
                self.log.append(f"⚠️ Failed to parse JSON: {e!r}\n")
        else:
            self.log.append("⚠️ Preview JSON missing or empty; table not updated.\n")

    def _on_thread_finished(self):
        """
        Clear thread references and re-enable UI controls after a run.
        """
        self._thread = None
        self._worker = None
        self.btn_generate.setEnabled(True)

    # ------------------------------ Table build + export ------------------------------

    def _populate_table(self, data: Dict[str, Any]):
        """
        Build a combined table:
            Columns = 1 (Fret) + S (Positions) + 1 (spacer) + S (Spacings)
            Rows    = Fret 0..F
        """
        positions = data.get("fret_positions", [])
        spacings = data.get("fret_spacings", [])
        unit = data.get("unit", "in")
        S = int(data.get("strings", len(positions)))
        F = int(data.get("frets", (len(positions[0]) - 1) if positions else 0))

        if not positions or not spacings or S == 0 or F <= 0:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.log.append("ℹ️ Table cleared (no valid data provided).\n")
            return

        col_count = 1 + S + 1 + S
        self.table.clear()
        self.table.setRowCount(F + 1)
        self.table.setColumnCount(col_count)

        headers = ["Fret"]
        headers += [f"S{i + 1} pos ({unit})" for i in range(S)]
        headers += [""]
        headers += [f"S{i + 1} gap ({unit})" for i in range(S)]
        self.table.setHorizontalHeaderLabels(headers)

        for fret in range(F + 1):
            self.table.setItem(fret, 0, QTableWidgetItem(str(fret)))
            # Positions block
            for s in range(S):
                val = positions[s][fret]
                self.table.setItem(fret, 1 + s, QTableWidgetItem(f"{val:.3f}"))
            # Spacer
            spacer_col = 1 + S
            self.table.setItem(fret, spacer_col, QTableWidgetItem(" "))
            self.table.setColumnWidth(spacer_col, 10)
            # Spacings block
            for s in range(S):
                val = spacings[s][fret]
                self.table.setItem(
                    fret, spacer_col + 1 + s, QTableWidgetItem(f"{val:.3f}")
                )

        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

    def _on_export_table_csv(self):
        """
        Export the *currently displayed* table to CSV without re-running the CLI.
        """
        if self.table.rowCount() == 0 or self.table.columnCount() == 0:
            self.log.append("ℹ️ Table is empty; nothing to export.\n")
            return
        # Ask the user where to save
        default_dir = self._config.get("last_output_dir", str(Path.cwd()))
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Export Table to CSV",
            str(Path(default_dir) / "fret_table.csv"),
            "CSV Files (*.csv)",
        )
        if not fname:
            return
        # Write CSV
        try:
            with open(fname, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Headers
                headers = [
                    self.table.horizontalHeaderItem(c).text()
                    if self.table.horizontalHeaderItem(c)
                    else ""
                    for c in range(self.table.columnCount())
                ]
                writer.writerow(headers)
                # Rows
                for r in range(self.table.rowCount()):
                    row = []
                    for c in range(self.table.columnCount()):
                        item = self.table.item(r, c)
                        row.append(item.text() if item else "")
                    writer.writerow(row)
            self.log.append(f"✅ Table exported to {fname}\n")
        except Exception as e:
            self.log.append(f"⚠️ Failed to export CSV: {e!r}\n")

    # ------------------------------ Toolbar actions ------------------------------

    def _on_zoom_changed(self, z: float):
        """
        Update the toolbar's zoom % readout whenever the view's zoom changes.
        """
        pct = round(z * 100)
        self.lbl_zoom.setText(f"Zoom: {pct}%")

    def _on_export_svg(self):
        """
        Copy the current preview SVG (temp) to a user-chosen file.
        """
        src = Path(self._tmp_svg)
        if not src.exists() or src.stat().st_size == 0:
            self.log.append("ℹ️ No preview SVG to export yet.\n")
            return
        default_dir = self._config.get("last_output_dir", str(Path.cwd()))
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Export SVG",
            str(Path(default_dir) / "fretboard.svg"),
            "SVG Files (*.svg)",
        )
        if not fname:
            return
        try:
            shutil.copyfile(src, fname)
            self.log.append(f"✅ SVG exported to {fname}\n")
        except Exception as e:
            self.log.append(f"⚠️ Failed to export SVG: {e!r}\n")

    # ------------------------------ File/Folder pickers ------------------------------

    def _on_browse_dir(self):
        """
        Pick the output directory for optional CSV/JSON/DXF exports (not the temp preview).
        """
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose Output Directory", self.in_out_dir.text()
        )
        if chosen:
            self.in_out_dir.setText(chosen)
            self._config["last_output_dir"] = chosen
            self._save_config()

    def _on_browse_spacing_file(self):
        """
        Choose a spacing list file (one gap per line). Only used if "From File" mode is selected.
        """
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Spacing List File",
            str(Path.cwd()),
            "Text Files (*.txt *.dat);;All Files (*)",
        )
        if fname:
            self.in_spacing_file.setText(fname)
            self.rb_spacing_file.setChecked(True)

    # ------------------------------ Preset handlers ------------------------------

    def _on_apply_preset(self):
        """
        Apply the selected preset's values to the GUI fields. Missing keys are ignored.
        """
        name = self.cb_preset.currentText()
        preset = self._presets.get(name, {})
        self._apply_preset_dict(preset)
        # Remember the last used preset in config
        self._config["last_preset"] = name
        self._save_config()
        self.queue_regeneration()

    def _apply_preset_dict(self, preset: Dict[str, str]):
        """
        Map preset keys to widget setters. Any missing keys are ignored.
        """
        # Simple mapping table: preset key -> widget
        mapping: Dict[str, Any] = {
            "frets": self.in_frets,
            "strings": self.in_strings,
            "scale": self.in_scale,
            "bass_scale": self.in_bass_scale,
            "treble_scale": self.in_treble_scale,
            "scale_map": self.cb_scale_map,
            "scale_gamma": self.in_scale_gamma,
            "unit": self.cb_unit,
            "decimals": self.in_decimals,
            "neutral_fret": self.in_neutral,
            "string_spacing": self.in_spacing_uniform,
            "string_spacing_list": self.in_spacing_list,
            "string_spacing_file": self.in_spacing_file,
            "nut_width": self.in_nut_width,
            "bridge_width": self.in_bridge_width,
            "board_margin": self.in_board_margin,
            "board_margin_x": self.in_board_margin_x,
            "datum_angle": self.in_datum_angle,
            "stroke": self.in_stroke,
            "slot_kerf": self.in_slot_kerf,
            "normalize_mode": self.cb_normalize_mode,
            "target_bridge_angle": self.in_target_bridge_angle,
        }

        for k, w in mapping.items():
            if k in preset:
                val = preset[k]
                if isinstance(w, QLineEdit):
                    w.setText(val)
                elif isinstance(w, QComboBox):
                    idx = w.findText(val)
                    if idx >= 0:
                        w.setCurrentIndex(idx)

        # Draw strings
        if "draw_strings" in preset:
            self.chk_draw_strings.setChecked(preset.get("draw_strings", "1") == "1")
        # Emit slot offsets
        if "emit_slot_offsets" in preset:
            self.chk_emit_slot_offsets.setChecked(
                preset.get("emit_slot_offsets", "0") == "1"
            )
        # Normalize flag
        if "normalize" in preset:
            self.chk_normalize.setChecked(preset.get("normalize", "0") == "1")

        # Spacing mode preference (set radio)
        if "string_spacing_file" in preset and preset["string_spacing_file"]:
            self.rb_spacing_file.setChecked(True)
        elif "string_spacing_list" in preset and preset["string_spacing_list"]:
            self.rb_spacing_list.setChecked(True)
        else:
            self.rb_spacing_uniform.setChecked(True)

    def _on_save_preset(self):
        """
        Save or update the currently selected preset name with *current UI values*.
        """
        name = self.cb_preset.currentText().strip()
        if not name:
            self.log.append(
                "⚠️ Please select or type a preset name in the Preset box.\n"
            )
            return
        # Collect all current UI values into a dict
        p: Dict[str, str] = {
            "frets": self.in_frets.text(),
            "strings": self.in_strings.text(),
            "scale": self.in_scale.text(),
            "bass_scale": self.in_bass_scale.text(),
            "treble_scale": self.in_treble_scale.text(),
            "scale_map": self.cb_scale_map.currentText(),
            "scale_gamma": self.in_scale_gamma.text(),
            "unit": self.cb_unit.currentText(),
            "decimals": self.in_decimals.text(),
            "neutral_fret": self.in_neutral.text(),
            "nut_width": self.in_nut_width.text(),
            "bridge_width": self.in_bridge_width.text(),
            "board_margin": self.in_board_margin.text(),
            "board_margin_x": self.in_board_margin_x.text(),
            "datum_angle": self.in_datum_angle.text(),
            "stroke": self.in_stroke.text(),
            "slot_kerf": self.in_slot_kerf.text(),
            "normalize_mode": self.cb_normalize_mode.currentText(),
            "target_bridge_angle": self.in_target_bridge_angle.text(),
            "draw_strings": "1" if self.chk_draw_strings.isChecked() else "0",
            "emit_slot_offsets": "1" if self.chk_emit_slot_offsets.isChecked() else "0",
            "normalize": "1" if self.chk_normalize.isChecked() else "0",
        }
        # Spacing mode
        if self.rb_spacing_uniform.isChecked():
            p["string_spacing"] = self.in_spacing_uniform.text()
            p["string_spacing_list"] = ""
            p["string_spacing_file"] = ""
        elif self.rb_spacing_list.isChecked():
            p["string_spacing"] = ""
            p["string_spacing_list"] = self.in_spacing_list.text()
            p["string_spacing_file"] = ""
        else:
            p["string_spacing"] = ""
            p["string_spacing_list"] = ""
            p["string_spacing_file"] = self.in_spacing_file.text()

        self._presets[name] = p
        save_json_file(PRESETS_PATH, self._presets)
        self._refresh_preset_combo()
        self.log.append(f"✅ Preset saved/updated: {name}\n")

    def _on_delete_preset(self):
        """
        Delete the selected preset from disk (if present).
        """
        name = self.cb_preset.currentText().strip()
        if not name:
            return
        if name in self._presets:
            del self._presets[name]
            save_json_file(PRESETS_PATH, self._presets)
            self._refresh_preset_combo()
            self.log.append(f"🗑️ Preset deleted: {name}\n")

    # ------------------------------ Config persistence ------------------------------

    def _load_presets(self) -> Dict[str, Dict[str, str]]:
        """
        Load preset file; if missing or invalid, return {}.
        """
        return load_json_file(PRESETS_PATH, {})

    def _load_config(self) -> Dict[str, Any]:
        """
        Load the user config file; initialize default keys if missing.
        """
        cfg = load_json_file(CONFIG_PATH, {})
        # Default toggles / remember last dirs
        cfg.setdefault("auto_preview", True)
        cfg.setdefault("last_output_dir", str(Path.cwd()))
        cfg.setdefault("last_preset", "None (manual)")
        # Geometry placeholders (set on first save)
        cfg.setdefault("win_geom", None)
        return cfg

    def _save_config(self):
        """
        Save the current config to disk. We update keys when they change.
        """
        save_json_file(CONFIG_PATH, self._config)

    def _restore_config(self):
        """
        Restore window geometry (if present), auto preview, last dir/preset.
        """
        # Geometry (window position/size)
        geom = self._config.get("win_geom")
        if isinstance(geom, list) and len(geom) == 4:
            try:
                self.setGeometry(geom[0], geom[1], geom[2], geom[3])
            except Exception:
                pass

        # Auto preview toggle
        ap = self._config.get("auto_preview", True)
        self.btn_auto.setChecked(ap)
        self.btn_auto.setText(f"Auto Preview: {'ON' if ap else 'OFF'}")

        # Last output dir is already bound to the line edit when user picks a dir
        last_dir = self._config.get("last_output_dir")
        if last_dir:
            self.in_out_dir.setText(last_dir)

        # Last preset is applied when refreshing the combo; nothing else needed here.

    def closeEvent(self, event):
        """
        Persist final config on window close: geometry, last output dir, last preset, auto preview.
        """
        # Window geometry
        g = self.geometry()
        self._config["win_geom"] = [g.x(), g.y(), g.width(), g.height()]
        # Last dir + preset already tracked
        # Auto preview state already tracked
        self._save_config()
        super().closeEvent(event)


# -------------------------------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------------------------------


def main():
    app = QApplication(sys.argv)
    gui = FretboardGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
