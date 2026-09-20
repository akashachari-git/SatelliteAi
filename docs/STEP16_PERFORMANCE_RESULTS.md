# SatQuery AI — Step 16 CPU Performance Results

Measured on: 2026-09-20 14:34:01 UTC
Hardware Environment: Windows 11, Python 3.13, CPU-only runtime (No CUDA GPU)

## 1. Before vs After Performance Comparison

| Component / Task | Baseline (Before) | Optimized (After) | Improvement / Notes |
|---|---:|---:|---|
| **BigEarthNet ONNX Model Load (First)** | `0.3487s` | `0.3155s` | Initial session initialization |
| **BigEarthNet ONNX Model Load (Reused)** | `0.3487s` | `1.2e-05s` | **Instantaneous** (~0.000s) via thread-safe session reuse |
| **BigEarthNet Single-Tile Inference** | `0.0081s` | `0.0068s` | Multi-threaded ONNX session (`intra_threads=4`) |
| **BigEarthNet 4-Tile Processing** | `0.0279s` (seq) | `0.0231s` (batch) | **Batched forward pass** `(4, 10, 120, 120)` in single call |
| **Florence-2 Model Load (First)** | `63.5719s` | `50.7103s` | Local checkpoint loading |
| **Florence-2 Model Load (Reused)** | `63.5719s` | `9e-06s` | **Instantaneous** (~0.000s) via lazy singleton retention |
| **Florence-2 Scene Captioning** | `13.1102s` | `8.841s` | Autoregressive decoding on CPU with inference lock |
| **Florence-2 VQA** | `7.1777s` | `4.4697s` | Visual Question Answering with CPU thread cap |
| **Florence-2 Phrase Grounding** | `6.8498s` | `4.9922s` | Phrase grounding with raster caching |
| **Bi-Temporal Change Detection** | `1.2622s` | `0.5984s` | Vectorized differencing + cached raster arrays |
| **Optical + SAR Fusion** | `0.0595s` | `0.0275s` | Vectorized indices + cached raster arrays |
| **Mission Intelligence Report** | `0.0002s` | `0.0001s` | Structured Markdown dossier generation |

## 2. Memory Observations

- **Baseline Peak Memory**: `243.78 MB`
- **Post-Optimization Peak Memory**: `245.98 MB`
- Memory remains bounded and stable thanks to:
  - Bounded raster cache (max 6 rasters in memory with LRU eviction).
  - Single model instance retained in memory without duplicate allocations.
  - Thread-safe inference locks preventing multi-request memory contention.

## 3. Summary of Key Wins

1. **Zero Repeated Model Loading**: Both Florence-2 and BigEarthNet are loaded once lazily and retained; subsequent requests execute without reload overhead.
2. **Batched Tile Forward Passes**: Multi-patch inputs are grouped into a single `(B, 10, 120, 120)` tensor call rather than B sequential round-trips.
3. **Bounded In-Memory Caching**: Bounded LRU caching prevents repeated TIFF decoding across validation, planning, and model execution.
4. **Safety Limits**: Huge or corrupted rasters (>250MB, >8192px, >32 bands) are rejected cleanly before allocating gigabytes of memory.
5. **Frontend Responsiveness**: Cursor HUD coordinates are throttled via `requestAnimationFrame` to eliminate UI stuttering.
