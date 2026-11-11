# Edge AI Fast Detection — the Speed Race

One YOLOv8n detector, two ways of running it, racing side by side on the
**same footage, same frames**:

| Panel | Backend | What it is |
|-------|---------|------------|
| Left | **PyTorch FP32** | The standard way most people run YOLOv8: the `.pt` weights through PyTorch |
| Right | **ONNX Runtime FP32** | The same weights exported to ONNX and run through ONNX Runtime — the format edge devices (Jetson, Raspberry Pi, industrial PCs) actually deploy |

Both panels show the live detections from their own backend, a live FPS
counter and the per-frame latency in milliseconds. The run ends with a
benchmark table that also includes a third pass over the same frames with
a **dynamically quantised INT8** ONNX model — the other standard edge
trick. Every number on screen is measured on the machine running the
script; nothing is copied from a datasheet.

Why this matters: on an edge device the model file rarely changes, the
runtime does. Exporting to ONNX and quantising is the difference between
a demo that crawls and a camera that keeps up — and this script lets you
watch that difference instead of reading about it.

## How the race is kept honest

- Both backends see exactly the same decoded frames, at the same input
  size (`--imgsz`, default 640) and confidence threshold.
- The order alternates every frame (PyTorch first on even frames, ONNX
  first on odd frames) so neither backend gets a systematic warm-cache
  advantage.
- The first `--warmup` frames (default 10) are processed and drawn but
  excluded from the averages — cold-start costs are not hidden, they are
  just not mixed into the steady-state numbers.
- "Latency" is the time of the model's `predict` call (pre-processing,
  inference and NMS inside Ultralytics), excluding video decoding and
  drawing. FPS in the table is `1000 / average latency`.

## Demo

**Verified run (11 Nov 2026)** — real mode on aerial intersection
traffic footage (1280×720; the demo uses a 25-second segment at
20 fps taken from a 59.94 fps stock clip), 500 frames raced, the
first 10 excluded as warmup, 490 timed frames per backend, measured
on a shared CPU-only virtual machine (AMD EPYC 9D25):

| Backend | Precision | Avg latency | Avg FPS | Frames |
|---------|-----------|-------------|---------|--------|
| PyTorch | FP32 | 1116.4 ms | 0.9 | 490 |
| ONNX Runtime | FP32 | 450.6 ms | 2.2 | 490 |
| ONNX Runtime | INT8 dynamic | 339.5 ms | 2.9 | 490 |

ONNX Runtime FP32 ran at **2.48× the frame rate of PyTorch FP32**
in this run, and the dynamically quantised INT8 model was the
fastest of the three. The demo video (29 s, H.264, with the
benchmark table on screen at the end) is
`edge-ai-fast-detection-demo.mp4` in the project workspace's
`files/project-demos/linkedin-h264/` folder, with the full
benchmark text saved alongside it.

The demo video is the script's **real mode** running on free-license
stock footage (aerial view of a busy road intersection, 1280×720) —
real footage processed by this code, never a simulation render.

Simulation mode (`--simulate`) renders a stylised street scene and runs
the same race on those pixels as an internal logic check only. It is
not a demo: YOLO reads the drawn shapes literally — in the verification
run it labelled the drawn buildings as "chair" and only one drawn figure
as a person — so the simulation proves the pipeline and the benchmark
maths, not detection quality.

## What the numbers mean (read before quoting them)

The benchmark table is a measurement of *this* machine, not a property
of the models. The demo run above was measured on a shared, CPU-only
virtual machine, so the absolute latencies are far slower than a
desktop — or an edge board — running nothing else would show;
PyTorch in particular suffers when CPU cores are shared. The
*relative* result is the lesson: under identical conditions the ONNX
Runtime FP32 export ran at about 2.5× the frame rate of the PyTorch
original, and the dynamically quantised INT8 model was faster still.
On other hardware the gaps shrink, grow, or occasionally flip —
dynamic quantisation only pays off where the runtime executes the
quantised ops well. Run the script on your own machine (or on a
Jetson / Pi) and the table will look different. That is the point:
measure, don't assume.

## Run it

```bash
pip install -r requirements.txt

python edge_ai_fast_detection.py                 # opens a file picker
python edge_ai_fast_detection.py --source pick   # same picker, explicit
python edge_ai_fast_detection.py --source traffic.mp4
python edge_ai_fast_detection.py --source 0      # webcam
python edge_ai_fast_detection.py --source traffic.mp4 --no-display
python edge_ai_fast_detection.py --simulate --max-seconds 30 --no-display
```

First run downloads `yolov8n.pt` automatically, exports `yolov8n.onnx`
if it is missing, and builds `yolov8n_int8.onnx` (dynamic quantisation)
if that is missing too. Press Q to quit. The race video is written to
`output.mp4` and the benchmark table to `output_benchmark.txt` next to it.

Useful flags: `--imgsz` (default 640), `--conf` (default 0.25),
`--warmup` (default 10 frames), `--max-seconds` (stop early),
`--summary-seconds` (benchmark table hold at the end, default 4),
`--model` (race a different YOLOv8 `.pt`, e.g. `yolov8s.pt`).

## Files

- `edge_ai_fast_detection.py` — the whole project: race screen, both
  backends, INT8 pass, benchmark table, simulation scene
- `requirements.txt` — `ultralytics`, `onnxruntime`, `opencv-python`,
  `numpy`, `pillow`
- `output_benchmark.txt` (produced on each run) — the measured table

Author: Abdul Hanan — Computer Vision & AI Developer
