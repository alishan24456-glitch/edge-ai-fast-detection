#!/usr/bin/env python3
"""
Edge AI Fast Detection -- the Speed Race
=========================================

One YOLOv8s detector, two ways of running it, racing side by side on
the SAME footage:

* **LEFT  -- PyTorch FP32.** The standard way most people run YOLOv8:
  the ``.pt`` weights through PyTorch.
* **RIGHT -- ONNX Runtime FP32.** The same weights exported to ONNX and
  run through ONNX Runtime -- the format edge devices (Jetson, Raspberry
  Pi, industrial PCs) actually deploy.

Both panels show the live detections from their own backend, a live FPS
counter and the per-frame latency in milliseconds. The run ends with a
benchmark table (backend, precision, average latency, average FPS) that
also includes a third pass over the same frames with a **dynamically
quantised INT8** ONNX model -- the other standard edge trick. Every
number on screen is measured on the machine running the script; nothing
is copied from a datasheet.

Why this matters: on an edge device the model file rarely changes, the
runtime does. Exporting to ONNX and quantising is the difference between
a demo that crawls and a camera that keeps up -- and this script lets you
watch that difference instead of reading about it.

How the race is kept honest
---------------------------
* Both backends see exactly the same decoded frames, at the same input
  size (``--imgsz``, default 640) and confidence threshold.
* The order alternates every frame (PyTorch first on even frames, ONNX
  first on odd frames) so neither backend gets a systematic warm-cache
  advantage.
* The first ``--warmup`` frames (default 10) are processed and drawn but
  excluded from the averages -- cold-start costs are not hidden, they
  are just not mixed into the steady-state numbers.
* "Latency" is the time of the model's ``predict`` call (pre-processing,
  inference and NMS inside Ultralytics), excluding video decoding and
  drawing. FPS in the table is ``1000 / average latency``.
* Only vehicle classes are reported (COCO ids 1, 2, 3, 5, 7: bicycle,
  car, motorcycle, bus, truck), at the full frame resolution
  (``--imgsz``, default 1280) and a confidence threshold of 0.55 by
  default. On aerial footage the smaller YOLOv8n at 640 px and a low
  threshold labelled rooftops as trucks and buses; the larger model at
  full resolution with the vehicle-only filter keeps every label on a
  real vehicle.

First run downloads ``yolov8s.pt`` automatically, exports
``yolov8s.onnx`` if it is missing, and builds ``yolov8s_int8.onnx``
(dynamic quantisation) if that is missing too.

Examples
--------
    python edge_ai_fast_detection.py                 (opens a file picker)
    python edge_ai_fast_detection.py --source pick   (same picker, explicit)
    python edge_ai_fast_detection.py --source traffic.mp4
    python edge_ai_fast_detection.py --source 0      (webcam)
    python edge_ai_fast_detection.py --source traffic.mp4 --no-display
    python edge_ai_fast_detection.py --simulate --max-seconds 30
        (rendered street scene, no input file needed -- internal
        logic check only; the delivered demo is always real footage)

Press Q to quit. The race video is written to ``output.mp4`` and the
benchmark table to ``output_benchmark.txt`` next to it.

Simulation note: --simulate renders a stylised street scene (cars and
pedestrians drawn with OpenCV) and runs the SAME race on those rendered
pixels -- both backends see only the pixels, exactly as in real mode.
It exists to check the pipeline and the benchmark maths without footage;
it is not a demo. YOLO may or may not recognise the drawn shapes as
cars/people, so the detection counts in a simulation run are whatever
the model actually sees, and the latency comparison is the point.

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Canvas layout (the race screen is always rendered at 1280x720)
# ---------------------------------------------------------------------------
W, H = 1280, 720
TITLE_H = 46                      # top title bar
PANEL_Y = TITLE_H                 # video panels start here
PANEL_H = 360                     # each panel is 640x360
PANEL_W = W // 2
HUD_Y = PANEL_Y + PANEL_H         # per-backend meters: 406..560
HUD_H = 154
BOARD_Y = HUD_Y + HUD_H           # live running-average board: 560..720

# Colours are BGR (OpenCV order) for canvas work; the text helper converts
# when it draws through PIL.
BG = (16, 20, 32)
PANEL_BG = (10, 12, 18)
WHITE = (240, 244, 248)
GREY = (150, 160, 175)
ORANGE = (0, 140, 255)            # PyTorch side
TEAL = (210, 200, 40)             # ONNX side (cyan-ish in BGR)
GREEN = (90, 200, 90)
RED = (70, 70, 230)

BACKEND_TORCH = "torch"
BACKEND_ONNX = "onnx"
BACKEND_INT8 = "int8"

# Set in main() from --model, e.g. "YOLOv8s"; shown in the race title bar.
MODEL_LABEL = "YOLOv8s"
# COCO class ids reported by default: bicycle, car, motorcycle, bus, truck.
VEHICLE_CLASSES = [1, 2, 3, 5, 7]

LABELS = {
    BACKEND_TORCH: ("PyTorch", "FP32"),
    BACKEND_ONNX: ("ONNX Runtime", "FP32"),
    BACKEND_INT8: ("ONNX Runtime", "INT8 dynamic"),
}
ACCENTS = {BACKEND_TORCH: ORANGE, BACKEND_ONNX: TEAL, BACKEND_INT8: GREEN}


# ---------------------------------------------------------------------------
# Text drawing: PIL (DejaVu / Arial) when available, OpenCV Hershey fallback.
# The race HUD needs clean numbers, so PIL is preferred; the fallback keeps
# the script working on a bare Windows install with only requirements.txt.
# ---------------------------------------------------------------------------
class Text:
    def __init__(self):
        self.pil = None
        self.fonts = {}
        try:
            from PIL import Image, ImageDraw, ImageFont

            candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:\\Windows\\Fonts\\arial.ttf",
            ]
            bold_candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "C:\\Windows\\Fonts\\arialbd.ttf",
            ]
            regular = next((p for p in candidates if os.path.exists(p)), None)
            bold = next((p for p in bold_candidates if os.path.exists(p)), None)
            if regular:
                self._pil = (Image, ImageDraw, ImageFont)
                self._regular_path = regular
                self._bold_path = bold or regular
                self.pil = True
        except Exception:
            self.pil = None

    def _font(self, size, bold):
        key = (size, bold)
        if key not in self.fonts:
            _, _, ImageFont = self._pil
            path = self._bold_path if bold else self._regular_path
            self.fonts[key] = ImageFont.truetype(path, size)
        return self.fonts[key]

    def draw(self, img, xy, s, size=20, color=WHITE, bold=False, anchor="la"):
        """Draw text on a BGR numpy image at xy. anchor is a PIL anchor."""
        if self.pil:
            Image, ImageDraw, _ = self._pil
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            d = ImageDraw.Draw(pil_img)
            rgb_color = (color[2], color[1], color[0])
            d.text(xy, s, font=self._font(size, bold), fill=rgb_color,
                   anchor=anchor)
            img[:, :] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        else:
            scale = size / 24.0
            thickness = 2 if bold else 1
            x, y = xy
            if anchor.startswith("m") or anchor == "ma":
                (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX,
                                             scale, thickness)
                x = x - tw // 2 if anchor[0] == "m" else x
            cv2.putText(img, s, (int(x), int(y + size)), cv2.FONT_HERSHEY_DUPLEX,
                        scale, color, thickness, cv2.LINE_AA)

    def width(self, s, size=20, bold=False):
        if self.pil:
            _, ImageDraw, _ = self._pil
            from PIL import Image as PILImage
            d = ImageDraw.Draw(PILImage.new("RGB", (8, 8)))
            box = d.textbbox((0, 0), s, font=self._font(size, bold))
            return box[2] - box[0]
        (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, size / 24.0,
                                     2 if bold else 1)
        return tw


TEXT = Text()


def draw_rect(img, x0, y0, x1, y1, color, fill=True):
    cv2.rectangle(img, (x0, y0), (x1, y1), color, -1 if fill else 2)


# ---------------------------------------------------------------------------
# Model handling: download, ONNX export, INT8 quantisation (all automatic)
# ---------------------------------------------------------------------------
def ensure_models(model_name, imgsz=1280):
    """Return (torch_model_path, onnx_path, int8_path_or_None)."""
    from ultralytics import YOLO

    print(f"[setup] loading {model_name} (downloads on first run) ...")
    torch_model = YOLO(model_name)  # auto-downloads the .pt weights
    pt_path = model_name if os.path.exists(model_name) else torch_model.ckpt_path
    base = os.path.splitext(os.path.basename(str(pt_path)))[0]
    onnx_path = os.path.join(os.getcwd(), base + ".onnx")
    int8_path = os.path.join(os.getcwd(), base + "_int8.onnx")

    if not os.path.exists(onnx_path):
        print("[setup] exporting ONNX (first run only) ...")
        exported = torch_model.export(format="onnx", imgsz=imgsz,
                                      dynamic=False)
        if exported and os.path.exists(str(exported)):
            onnx_path = str(exported)
    print(f"[setup] ONNX model: {onnx_path}")

    if not os.path.exists(int8_path):
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            print("[setup] building INT8 dynamic-quantised ONNX ...")
            quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QUInt8)
        except Exception as exc:  # quantisation is a bonus row, never fatal
            print(f"[setup] INT8 quantisation unavailable: {exc}")
            int8_path = None
    if int8_path and os.path.exists(int8_path):
        print(f"[setup] INT8 model: {int8_path}")
    return str(pt_path), onnx_path, int8_path


class Runner:
    """One backend: wraps an Ultralytics model and times its predict calls."""

    def __init__(self, key, path, imgsz, conf, classes=None):
        from ultralytics import YOLO

        self.key = key
        self.model = YOLO(path)
        self.imgsz = imgsz
        self.conf = conf
        self.classes = classes
        self.latencies = []            # steady-state latencies (ms)
        self.recent = deque(maxlen=30)  # for the live FPS counter
        self.detections = 0

    def run(self, frame):
        t0 = time.perf_counter()
        results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                     classes=self.classes, verbose=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = results[0]
        self.detections += len(result.boxes)
        return result, latency_ms

    def record(self, latency_ms, steady):
        self.recent.append(latency_ms)
        if steady:
            self.latencies.append(latency_ms)

    @property
    def live_fps(self):
        if not self.recent:
            return 0.0
        return 1000.0 / (sum(self.recent) / len(self.recent))

    @property
    def avg_latency(self):
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def avg_fps(self):
        lat = self.avg_latency
        return 1000.0 / lat if lat > 0 else 0.0


# ---------------------------------------------------------------------------
# Race screen rendering
# ---------------------------------------------------------------------------
def annotate_panel(result, accent, label, det_count):
    """Annotated panel image (640x360) with a backend chip in the corner."""
    plotted = result.plot()  # BGR, same size as the input frame
    panel = cv2.resize(plotted, (PANEL_W, PANEL_H))
    draw_rect(panel, 0, 0, 250, 34, (0, 0, 0))
    draw_rect(panel, 0, 0, 6, 34, accent)
    TEXT.draw(panel, (14, 6), label, size=19, color=WHITE, bold=True)
    chip = f"{det_count} objects"
    w = TEXT.width(chip, size=16)
    draw_rect(panel, PANEL_W - w - 22, 0, PANEL_W, 30, (0, 0, 0))
    TEXT.draw(panel, (PANEL_W - w - 11, 5), chip, size=16, color=WHITE)
    return panel


def draw_meter(canvas, x, runner, accent):
    """One backend's HUD block: big live FPS, latency, speed bar."""
    name, precision = LABELS[runner.key]
    TEXT.draw(canvas, (x + 16, HUD_Y + 12), f"{name}  ·  {precision}",
              size=20, color=accent, bold=True)
    fps_text = f"{runner.live_fps:5.1f}"
    TEXT.draw(canvas, (x + 16, HUD_Y + 40), fps_text, size=54, color=WHITE,
              bold=True)
    fps_w = TEXT.width(fps_text, size=54, bold=True)
    TEXT.draw(canvas, (x + 26 + fps_w, HUD_Y + 62), "FPS", size=22,
              color=GREY, bold=True)
    lat = 1000.0 / runner.live_fps if runner.live_fps > 0 else 0.0
    TEXT.draw(canvas, (x + 300, HUD_Y + 46),
              f"{lat:5.1f} ms / frame", size=21, color=WHITE)
    TEXT.draw(canvas, (x + 300, HUD_Y + 76),
              f"{runner.detections} objects so far", size=17, color=GREY)
    # speed bar: width relative to a 60 FPS full scale
    bar_x, bar_y, bar_w, bar_h = x + 16, HUD_Y + 116, PANEL_W - 32, 16
    draw_rect(canvas, bar_x, bar_y, bar_x + bar_w, bar_y + bar_h, (45, 52, 66))
    fill = int(bar_w * min(runner.live_fps, 60.0) / 60.0)
    if fill > 0:
        draw_rect(canvas, bar_x, bar_y, bar_x + fill, bar_y + bar_h, accent)


def race_frame(frame, results, runners, frame_idx, elapsed_note):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:, :] = BG
    # title bar
    TEXT.draw(canvas, (16, 10), "EDGE AI SPEED RACE", size=23, color=WHITE,
              bold=True)
    TEXT.draw(canvas, (428, 14), f"{MODEL_LABEL} -- PyTorch vs ONNX Runtime, "
              "same footage, same frames", size=16, color=GREY)
    TEXT.draw(canvas, (W - 16, 12), elapsed_note, size=18, color=GREY,
              anchor="ra")
    # panels
    for i, key in enumerate((BACKEND_TORCH, BACKEND_ONNX)):
        result, det_count = results[key]
        label = f"{LABELS[key][0]} {LABELS[key][1]}"
        panel = annotate_panel(result, ACCENTS[key], label, det_count)
        x = i * PANEL_W
        canvas[PANEL_Y:PANEL_Y + PANEL_H, x:x + PANEL_W] = panel
    draw_rect(canvas, PANEL_W - 1, PANEL_Y, PANEL_W + 1, PANEL_Y + PANEL_H,
              (70, 78, 94))
    # HUD divider
    draw_rect(canvas, 0, HUD_Y, W, HUD_Y + 2, (45, 52, 66))
    draw_meter(canvas, 0, runners[BACKEND_TORCH], ACCENTS[BACKEND_TORCH])
    draw_meter(canvas, PANEL_W, runners[BACKEND_ONNX], ACCENTS[BACKEND_ONNX])
    draw_rect(canvas, PANEL_W - 1, HUD_Y, PANEL_W + 1, HUD_Y + HUD_H,
              (45, 52, 66))
    # live board: running averages
    draw_rect(canvas, 0, BOARD_Y, W, BOARD_Y + 2, (45, 52, 66))
    TEXT.draw(canvas, (16, BOARD_Y + 12),
              "RUNNING AVERAGE (steady state, warmup excluded)",
              size=16, color=GREY, bold=True)
    y = BOARD_Y + 42
    for key in (BACKEND_TORCH, BACKEND_ONNX):
        r = runners[key]
        name, precision = LABELS[key]
        line = (f"{name} {precision}:  {r.avg_fps:5.1f} FPS   ·   "
                f"{r.avg_latency:5.1f} ms/frame   ·   "
                f"{len(r.latencies)} frames timed")
        TEXT.draw(canvas, (16, y), line, size=19, color=ACCENTS[key])
        y += 30
    # leader callout on the right of the board
    t, o = runners[BACKEND_TORCH], runners[BACKEND_ONNX]
    if t.avg_fps > 0 and o.avg_fps > 0:
        if o.avg_fps >= t.avg_fps:
            leader, ratio = "ONNX Runtime", o.avg_fps / t.avg_fps
        else:
            leader, ratio = "PyTorch", t.avg_fps / o.avg_fps
        msg = f"LEADER: {leader}  x{ratio:.2f}"
        TEXT.draw(canvas, (W - 16, BOARD_Y + 52), msg, size=30, color=GREEN,
                  bold=True, anchor="ra")
    return canvas


# ---------------------------------------------------------------------------
# End-of-run benchmark: table text + full-screen summary
# ---------------------------------------------------------------------------
def benchmark_rows(runners):
    rows = []
    for key in (BACKEND_TORCH, BACKEND_ONNX, BACKEND_INT8):
        r = runners.get(key)
        if r is None or not r.latencies:
            continue
        name, precision = LABELS[key]
        rows.append({
            "backend": name,
            "precision": precision,
            "latency": r.avg_latency,
            "fps": r.avg_fps,
            "frames": len(r.latencies),
            "detections": r.detections,
        })
    return rows


def format_table(rows):
    lines = []
    lines.append(f"{'Backend':<16} {'Precision':<13} {'Avg latency':>12} "
                 f"{'Avg FPS':>9} {'Frames':>7}")
    lines.append("-" * 62)
    for r in rows:
        lines.append(f"{r['backend']:<16} {r['precision']:<13} "
                     f"{r['latency']:>9.1f} ms {r['fps']:>9.1f} "
                     f"{r['frames']:>7}")
    return "\n".join(lines)


def speedup_line(rows):
    by = {r["backend"] + r["precision"]: r for r in rows}
    torch = next((r for r in rows if r["backend"] == "PyTorch"), None)
    onnx = next((r for r in rows
                 if r["backend"] == "ONNX Runtime" and r["precision"] == "FP32"),
                None)
    if torch and onnx and torch["fps"] > 0:
        ratio = onnx["fps"] / torch["fps"]
        if ratio >= 1.0:
            return (f"ONNX Runtime FP32 ran at {ratio:.2f}x the frame rate "
                    f"of PyTorch FP32 on this machine.")
        return (f"On this machine PyTorch FP32 was faster: ONNX Runtime "
                f"FP32 reached {ratio:.2f}x its frame rate.")
    return ""


def summary_frame(rows, note_lines):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:, :] = BG
    TEXT.draw(canvas, (W // 2, 44), "BENCHMARK RESULTS", size=34, color=WHITE,
              bold=True, anchor="ma")
    TEXT.draw(canvas, (W // 2, 88),
              "Same footage, same frames, same settings -- measured live "
              "during the race", size=19, color=GREY, anchor="ma")
    # table
    tx, ty = 90, 170
    col_x = [tx, tx + 330, tx + 540, tx + 760, tx + 950]
    headers = ["Backend", "Precision", "Avg latency", "Avg FPS", "Frames"]
    draw_rect(canvas, tx - 20, ty - 14, W - 70, ty + 44, (30, 38, 54))
    for cx, htxt in zip(col_x, headers):
        TEXT.draw(canvas, (cx, ty), htxt, size=21, color=WHITE, bold=True)
    y = ty + 62
    accents = [ORANGE, TEAL, GREEN]
    for i, r in enumerate(rows):
        vals = [r["backend"], r["precision"], f"{r['latency']:.1f} ms",
                f"{r['fps']:.1f}", f"{r['frames']}"]
        for cx, v in zip(col_x, vals):
            TEXT.draw(canvas, (cx, y), v, size=22,
                      color=accents[min(i, 2)] if cx == col_x[0] else WHITE,
                      bold=(cx in (col_x[0], col_x[3])))
        y += 44
        draw_rect(canvas, tx - 20, y - 8, W - 70, y - 7, (45, 52, 66))
    msg = speedup_line(rows)
    if msg:
        TEXT.draw(canvas, (W // 2, y + 26), msg, size=23, color=GREEN,
                  bold=True, anchor="ma")
    ny = H - 118
    for note in note_lines:
        TEXT.draw(canvas, (90, ny), note, size=16, color=GREY)
        ny += 24
    return canvas


# ---------------------------------------------------------------------------
# Source handling: file picker / path / webcam index
# ---------------------------------------------------------------------------
def pick_file():
    """Tkinter video chooser. Returns a path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All files", "*.*")])
        root.destroy()
        return path or None
    except Exception as exc:
        print(f"File picker unavailable ({exc}). "
              "Pass a path instead: --source video.mp4")
        return None


def open_source(source_arg):
    """Resolve --source into (cv2.VideoCapture, is_webcam, label)."""
    if source_arg is None or str(source_arg).lower() == "pick":
        path = pick_file()
        if not path:
            print("No file chosen. Exiting.")
            sys.exit(0)
        source_arg = path
    if str(source_arg).isdigit():
        idx = int(source_arg)
        cap = cv2.VideoCapture(idx)
        return cap, True, f"webcam {idx}"
    cap = cv2.VideoCapture(str(source_arg))
    return cap, False, str(source_arg)


# ---------------------------------------------------------------------------
# Simulation mode: a rendered street scene, judged through the same
# detection/pixel stream as real footage -- no ground-truth peeking.
# Both backends run real YOLO on these rendered pixels; whatever they
# detect (or miss) is what the panels show. Motion is a pure function of
# the frame index, so the scene is identical on every run.
# ---------------------------------------------------------------------------
SIM_W, SIM_H = 1280, 720
SIM_FPS = 30.0

# (lane centre y, direction +1 right / -1 left, speed px/frame,
#  start x, length, colour BGR)
_SIM_CARS = [
    (360, +1, 4.0, 100, 190, (40, 40, 180)),    # red car, upper lane
    (360, +1, 3.0, 700, 170, (180, 130, 40)),   # blue car
    (455, +1, 5.0, 400, 150, (30, 170, 220)),   # yellow taxi
    (560, -1, 4.5, 1100, 185, (200, 200, 200)),  # white car, lower lane
    (560, -1, 3.5, 300, 160, (60, 60, 60)),     # dark car
    (650, -1, 5.5, 800, 175, (40, 120, 200)),   # orange car
]
# (pavement y, direction, speed px/frame, start x, shirt colour BGR)
_SIM_PEOPLE = [
    (295, +1, 1.6, 150, (200, 60, 60)),
    (295, -1, 1.4, 950, (60, 180, 60)),
    (700, +1, 1.8, 500, (60, 60, 200)),
    (700, -1, 1.5, 1100, (180, 60, 180)),
]


def _draw_car(img, cx, cy, length, color, direction):
    """A simple side-on car: body, cabin, windows, wheels."""
    h = 52
    x0, x1 = int(cx - length // 2), int(cx + length // 2)
    y0, y1 = int(cy - h // 2), int(cy + h // 2)
    cv2.rectangle(img, (x0, y0 + 12), (x1, y1), color, -1)
    cabin_x0 = x0 + length // 4 if direction > 0 else x0 + length // 5
    cabin_x1 = x1 - length // 5 if direction > 0 else x1 - length // 4
    cv2.rectangle(img, (cabin_x0, y0), (cabin_x1, y0 + 22), color, -1)
    cv2.rectangle(img, (cabin_x0 + 6, y0 + 4), (cabin_x1 - 6, y0 + 18),
                  (220, 230, 240), -1)
    for wx in (x0 + 28, x1 - 28):
        cv2.circle(img, (wx, y1), 11, (15, 15, 15), -1)
        cv2.circle(img, (wx, y1), 5, (90, 90, 90), -1)
    # headlight in the direction of travel
    hx = x1 - 4 if direction > 0 else x0 + 4
    cv2.circle(img, (hx, y0 + 22), 5, (180, 240, 255), -1)


def _draw_person(img, x, ground_y, shirt, frame_idx, direction):
    """A simple walking figure: head, torso, swinging legs."""
    bob = int(2 * abs(((frame_idx // 4) % 2) * 2 - 1))
    head_y = ground_y - 52 + bob
    cv2.circle(img, (int(x), head_y), 9, (200, 170, 140), -1)
    cv2.rectangle(img, (int(x) - 8, head_y + 9),
                  (int(x) + 8, ground_y - 14), shirt, -1)
    swing = 8 if (frame_idx // 6) % 2 == 0 else -8
    cv2.line(img, (int(x), ground_y - 14),
             (int(x) + swing * direction, ground_y), (30, 30, 30), 3)
    cv2.line(img, (int(x), ground_y - 14),
             (int(x) - swing * direction, ground_y), (30, 30, 30), 3)


def render_sim_frame(frame_idx):
    """Render one simulation frame (1280x720 BGR)."""
    img = np.zeros((SIM_H, SIM_W, 3), dtype=np.uint8)
    # sky + buildings
    img[0:280, :] = (235, 205, 160)
    for bx, bw, bh in ((0, 180, 190), (200, 130, 150), (360, 210, 210),
                       (600, 150, 170), (780, 190, 200), (1000, 150, 160),
                       (1180, 100, 190)):
        cv2.rectangle(img, (bx, 280 - bh), (bx + bw, 280), (120, 120, 130), -1)
        for wy in range(280 - bh + 14, 270, 34):
            for wx in range(bx + 14, bx + bw - 14, 34):
                cv2.rectangle(img, (wx, wy), (wx + 16, wy + 18),
                              (255, 240, 180), -1)
    # pavements and road
    cv2.rectangle(img, (0, 280), (SIM_W, 310), (150, 150, 150), -1)
    cv2.rectangle(img, (0, 310), (SIM_W, SIM_H - 30), (52, 52, 58), -1)
    cv2.rectangle(img, (0, SIM_H - 30), (SIM_W, SIM_H), (150, 150, 150), -1)
    # centre line + dashed lane lines
    cv2.rectangle(img, (0, 508), (SIM_W, 516), (0, 215, 255), -1)
    for y in (405, 605):
        for x in range(0, SIM_W, 90):
            cv2.rectangle(img, (x, y), (x + 48, y + 5), (230, 230, 230), -1)
    # cars, wrapping around the edges
    span = SIM_W + 400
    for lane_y, direction, speed, start_x, length, color in _SIM_CARS:
        x = ((start_x + direction * speed * frame_idx) % span) - 200
        if direction < 0:
            x = span - 200 - ((start_x + speed * frame_idx) % span)
        _draw_car(img, x, lane_y, length, color, direction)
    # pedestrians on the pavements
    for ground_y, direction, speed, start_x, shirt in _SIM_PEOPLE:
        x = ((start_x + direction * speed * frame_idx) % (SIM_W + 120)) - 60
        _draw_person(img, x, ground_y, shirt, frame_idx, direction)
    cv2.putText(img, "SIMULATION -- rendered street scene",
                (16, 36), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA)
    return img


class SimCapture:
    """A cv2.VideoCapture look-alike serving rendered simulation frames."""

    def __init__(self, total_frames, fps=SIM_FPS):
        self.total_frames = int(total_frames)
        self.fps = fps
        self.idx = 0

    def isOpened(self):
        return True

    def read(self):
        if self.idx >= self.total_frames:
            return False, None
        frame = render_sim_frame(self.idx)
        self.idx += 1
        return True, frame

    def release(self):
        pass

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0


# ---------------------------------------------------------------------------
# Main race
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Edge AI Fast Detection -- YOLOv8 speed race: "
                    "PyTorch FP32 vs ONNX Runtime on the same footage.")
    ap.add_argument("--source", default=None,
                    help="'pick' (default with no flag: file chooser), a "
                         "video path, or a webcam index such as 0")
    ap.add_argument("--model", default="yolov8s.pt",
                    help="YOLO weights to race (default yolov8s.pt)")
    ap.add_argument("--output", default="output.mp4",
                    help="race video to write (default output.mp4)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--classes", default="1,2,3,5,7",
                    help="comma-separated COCO class ids to report "
                         "(default: vehicles only -- bicycle, car, "
                         "motorcycle, bus, truck; empty string = all)")
    ap.add_argument("--warmup", type=int, default=10,
                    help="frames excluded from the averages (default 10)")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="stop after this many seconds of footage (0 = all)")
    ap.add_argument("--summary-seconds", type=float, default=4.0,
                    help="seconds of benchmark table at the end of the video")
    ap.add_argument("--no-display", action="store_true",
                    help="do not open a window (headless runs)")
    ap.add_argument("--simulate", action="store_true",
                    help="race on a rendered street scene instead of a "
                         "video/webcam (internal logic check only; the "
                         "delivered demo is always real footage)")
    args = ap.parse_args()

    global MODEL_LABEL
    stem = os.path.splitext(os.path.basename(args.model))[0]
    MODEL_LABEL = "YOLO" + stem[4:] if stem.lower().startswith("yolo") else stem
    classes = None
    if args.classes and args.classes.strip():
        classes = [int(x) for x in args.classes.split(",") if x.strip()]

    pt_path, onnx_path, int8_path = ensure_models(args.model, args.imgsz)
    runners = {
        BACKEND_TORCH: Runner(BACKEND_TORCH, pt_path, args.imgsz, args.conf,
                              classes),
        BACKEND_ONNX: Runner(BACKEND_ONNX, onnx_path, args.imgsz, args.conf,
                             classes),
    }
    if int8_path:
        runners[BACKEND_INT8] = Runner(BACKEND_INT8, int8_path, args.imgsz,
                                       args.conf, classes)

    def reopen_source():
        """A fresh frame source for the INT8 pass (same footage/scene)."""
        if args.simulate:
            total = int((args.max_seconds or 30.0) * SIM_FPS)
            return SimCapture(total), False, "simulation (rendered street scene)"
        return open_source(args.source)

    if args.simulate:
        total = int((args.max_seconds or 30.0) * SIM_FPS)
        cap = SimCapture(total)
        is_webcam, label = False, "simulation (rendered street scene)"
    else:
        cap, is_webcam, label = open_source(args.source)
    if not cap.isOpened():
        print(f"Could not open source: {label}")
        return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0 or src_fps > 120:
        src_fps = 30.0
    out_fps = src_fps
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, out_fps, (W, H))
        if not writer.isOpened():
            print(f"Could not open output writer for {args.output}")
            return 1

    print(f"[race] source: {label} @ {src_fps:.2f} fps -- "
          "PyTorch FP32 (left) vs ONNX Runtime FP32 (right)")
    frame_idx = 0
    max_frames = int(args.max_seconds * src_fps) if args.max_seconds else None
    quit_now = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break
        steady = frame_idx >= args.warmup
        order = ((BACKEND_TORCH, BACKEND_ONNX) if frame_idx % 2 == 0
                 else (BACKEND_ONNX, BACKEND_TORCH))
        results = {}
        for key in order:
            result, latency = runners[key].run(frame)
            runners[key].record(latency, steady)
            results[key] = (result, len(result.boxes))
        note = f"frame {frame_idx + 1}  ·  {frame_idx / src_fps:5.1f} s"
        canvas = race_frame(frame, results, runners, frame_idx, note)
        if writer:
            writer.write(canvas)
        if not args.no_display:
            cv2.imshow("Edge AI Fast Detection -- Speed Race", canvas)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                quit_now = True
                break
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[race] {frame_idx} frames processed")
    cap.release()
    timed_frames = len(runners[BACKEND_TORCH].latencies)
    print(f"[race] race pass done: {frame_idx} frames, "
          f"{timed_frames} timed per backend")

    # Third benchmark pass: INT8 over the same frames (no drawing -- the
    # numbers join the final table only, exactly as the video explains).
    if int8_path and not quit_now and not is_webcam:
        cap2, _, _ = reopen_source()
        idx = 0
        while True:
            ok, frame = cap2.read()
            if not ok or (max_frames and idx >= max_frames):
                break
            _, latency = runners[BACKEND_INT8].run(frame)
            runners[BACKEND_INT8].record(latency, idx >= args.warmup)
            idx += 1
        cap2.release()
        print(f"[race] INT8 pass done: {idx} frames")

    rows = benchmark_rows(runners)
    table = format_table(rows)
    print("\n=== BENCHMARK RESULTS (measured on this machine) ===")
    print(table)
    msg = speedup_line(rows)
    if msg:
        print(msg)

    notes = [
        "Latency = Ultralytics predict() call time (pre-processing + "
        "inference + NMS); video decode and drawing excluded.",
        f"Settings: model {args.model}, imgsz {args.imgsz}, conf "
        f"{args.conf}, vehicle classes only, first {args.warmup} frames "
        "excluded as warmup; order alternated.",
        "CPU-only run on the machine that produced this video. "
        "Numbers vary by hardware -- run it on yours.",
    ]
    if writer:
        summary = summary_frame(rows, notes)
        for _ in range(int(args.summary_seconds * out_fps)):
            writer.write(summary)
        writer.release()
        stem = os.path.splitext(args.output)[0]
        with open(stem + "_benchmark.txt", "w", encoding="utf-8") as fh:
            fh.write("Edge AI Fast Detection -- benchmark results\n")
            fh.write(f"Source: {label}\n\n{table}\n\n{msg}\n\n")
            fh.write("\n".join(notes) + "\n")
        print(f"[done] race video: {args.output}")
        print(f"[done] benchmark: {stem}_benchmark.txt")
    if not args.no_display:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
