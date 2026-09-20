# SatQuery AI — Step 16 CPU Performance Baseline

Measured on: 2026-09-20 14:26:50 UTC
Hardware Environment: Windows 11, Python 3.13, CPU-only runtime (No CUDA GPU)

## 1. Measured Latencies

| Component / Task | Measured Latency | Notes / Details |
|---|---:|---|
| **BigEarthNet ONNX Model Load** | `0.3487s` | ONNX Runtime CPUExecutionProvider initialization |
| **BigEarthNet Single-Tile Inference** | `0.0081s` | Mean forward pass on 10-band (10, 120, 120) tensor |
| **BigEarthNet Tiling (4 tiles sequential)** | `0.0279s` | Sequential forward passes without batching |
| **Florence-2 VLM Model Load** | `63.5719s` | Loading weights from local disk checkpoint |
| **Florence-2 Scene Captioning** | `13.1102s` | Open-ended captioning on 120×120 patch |
| **Florence-2 VQA** | `7.1777s` | Visual Question Answering on 512×512 raster |
| **Florence-2 Phrase Grounding** | `6.8498s` | Phrase grounding query on 120×120 patch |
| **Bi-Temporal Change Detection** | `1.2622s` | Classical spectral differencing on 512×512 pair |
| **Optical + SAR Fusion** | `0.0595s` | Cross-sensor corroboration on 512×512 pair |
| **Mission Intelligence Report Generation**| `0.0002s` | Structured Markdown dossier generation |

## 2. Memory Usage

- **Peak Process Memory Traced**: `243.78 MB`

## 3. Primary Bottlenecks Identified

1. **Florence-2 Initial Load Time**: `63.5719s` — Must be loaded once (lazy singleton) and retained in memory, never reloaded per request.
2. **Florence-2 CPU Inference Latency**: ~7.1777s–13.1102s per query — CPU-bound autoregressive decoding. Requires concurrency protection to prevent multi-request CPU exhaustion.
3. **Sequential Tile Forward Passes**: Running multiple tiles sequentially incurs N separate ONNX calls. Batching compatible tiles into a single forward pass `(N, 10, 120, 120)` will optimize throughput.
4. **ONNX Thread Configuration**: Defaults may not be optimized for CPU core topology.
