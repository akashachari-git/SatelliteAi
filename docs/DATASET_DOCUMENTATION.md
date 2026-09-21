# SatQuery AI — Dataset & Benchmark Integration

## 1. Supported Benchmarks
SatQuery AI incorporates standard benchmarks specified in the problem statement:

| Benchmark | Sensor / Modality | Primary Task | Metrics Reported |
| :--- | :--- | :--- | :--- |
| **BigEarthNet-S2 (BigEarthNet.txt)** | Sentinel-2 (B01-B12) | Multi-label Land Cover Classification | Micro-F1, Macro-F1, mAP |
| **RSVQA** | Sentinel-2 / Landsat | Single-Image Remote Sensing VQA | Top-1 Accuracy, Exact Match (EM) |
| **CDVQA** | Bi-temporal High-Res EO | Change Detection VQA | Directional Accuracy, Sector IoU |
| **VRSBench** | Multi-sensor Optical | Captioning & Text-guided Grounding | BLEU-4, CIDEr, Mean Grounding IoU |
| **Million-AID** | Optical Multi-task | Pretraining representation | *Reported as unmounted if directory absent* |

## 2. Scientific Honesty & Zero Metric Fabrication
In accordance with platform evaluation integrity guidelines:
- If a benchmark dataset directory is not mounted locally, the backend returns `"status": "NOT_CONFIGURED"`.
- The user interface displays a prominent warning: *"Benchmark dataset not configured locally"* rather than fabricated test results.
