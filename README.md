# Edge AI Fast Detection — the Speed Race

One YOLOv8s detector, two ways of running it, racing side by side on the
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
  size (`--imgsz`, default 1280 — the full frame) and confidence
  threshold (`--conf`, default 0.55), and both report vehicle classes
  only (bicycle, car, motorcycle, bus, truck).
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

**Verified run (12 Nov 2026, rebuilt — see the detection note below)**
— real mode on aerial intersection traffic footage (1280×720; the
demo uses a 19-second segment at 20 fps taken from a 59.94 fps stock
clip), 380 frames raced, the first 10 excluded as warmup, 370 timed
frames per backend, measured on a shared CPU-only virtual machine
(AMD EPYC 9D25):

| Backend | Precision | Avg latency | Avg FPS | Frames |
|---------|-----------|-------------|---------|--------|
| PyTorch | FP32 | 691.6 ms | 1.4 | 370 |
| ONNX Runtime | FP32 | 1460.7 ms | 0.7 | 370 |
| ONNX Runtime | INT8 dynamic | 1681.6 ms | 0.6 | 370 |

On this machine **PyTorch FP32 was the fastest of the three**: ONNX
Runtime FP32 reached 0.47× its frame rate, and the dynamically
quantised INT8 model was the slowest. That is the opposite order
from the first run of this script (11 Nov 2026, YOLOv8n at 640 px on
the same VM: PyTorch 1116.4 ms · 0.9 FPS, ONNX FP32 450.6 ms ·
2.2 FPS — 2.48× — INT8 339.5 ms · 2.9 FPS). Same machine, same
method, different model size and input resolution, opposite winner —
which is exactly why the table is measured, not assumed. The demo
video (23 s, H.264, with the benchmark table on screen at the end) is
`edge-ai-fast-detection-demo.mp4` in the project workspace's
`files/project-demos/linkedin-h264/` folder, with the full
benchmark text saved alongside it.

**Detection note (why the demo was rebuilt):** the first demo ran
YOLOv8n at 640 px with a 0.25 threshold, and on this aerial footage
it labelled rooftops as "truck" and "bus". The rebuilt demo runs
YOLOv8s at the full 1280 px frame, a 0.55 threshold and vehicle
classes only, and every checked frame (2 s, 6 s, 10 s, 14 s, 18 s)
shows labels on real road vehicles only. The segment ends at 19 s
because the camera then swings to an angle where rooftops fill the
lower frame and even the larger model starts reading one long roof
as a vehicle — the demo keeps to the angles where the labels are
right.

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
desktop — or an edge board — running nothing else would show. The
*relative* result is the lesson, and this project has now produced
both versions of it: at 640 px input the ONNX export won by ~2.5×;
at the full 1280 px frame on the same VM, PyTorch won and the INT8
model came last. Under identical conditions the winner depends on
the model, the input size and the runtime's CPU path — dynamic
quantisation only pays off where the runtime executes the quantised
ops well. Run the script on your own machine (or on a Jetson / Pi)
and the table will look different. That is the point: measure,
don't assume.

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

First run downloads `yolov8s.pt` automatically, exports `yolov8s.onnx`
if it is missing, and builds `yolov8s_int8.onnx` (dynamic quantisation)
if that is missing too. Press Q to quit. The race video is written to
`output.mp4` and the benchmark table to `output_benchmark.txt` next to it.

Useful flags: `--imgsz` (default 1280), `--conf` (default 0.55),
`--classes` (default vehicle classes only: 1,2,3,5,7),
`--warmup` (default 10 frames), `--max-seconds` (stop early),
`--summary-seconds` (benchmark table hold at the end, default 4),
`--model` (race a different YOLOv8 `.pt`, e.g. `yolov8m.pt`).

## Files

- `edge_ai_fast_detection.py` — the whole project: race screen, both
  backends, INT8 pass, benchmark table, simulation scene
- `requirements.txt` — `ultralytics`, `onnxruntime`, `opencv-python`,
  `numpy`, `pillow`
- `output_benchmark.txt` (produced on each run) — the measured table

Author: Abdul Hanan — Computer Vision & AI Developer
