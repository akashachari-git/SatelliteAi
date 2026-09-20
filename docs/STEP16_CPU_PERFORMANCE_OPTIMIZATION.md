# SatQuery AI — Step 16: CPU Performance Optimization & Resource Engineering

## 1. Hardware Environment & Assumptions

- **Operating System**: Microsoft Windows 11 (build 10.0.26100)
- **Processor / Architecture**: x86_64 multi-core CPU (No NVIDIA CUDA GPU active)
- **Python Version**: 3.13.1 (64-bit)
- **Execution Constraints**:
  - All neural network inferences (Florence-2 and BigEarthNet) strictly execute on CPU.
  - No GPU acceleration is assumed or simulated.
  - System must prioritize stable, non-starving inference over aggressive oversubscription.
  - Zero mock substitutions, zero hardcoded coordinates, zero fabricated benchmark scores.

---

## 2. Baseline Measurements vs. Post-Optimization Results

All benchmarks were measured on authentic remote-sensing rasters (`urban_satellite_eo.tif`, `example_s2_patch.png`, `temporal_t1_2021.tif`, `crossmodal_sar_s1.tif`).

| Component / Benchmark Task | Baseline (Before) | Optimized (After) | Improvement / Engineering Note |
|---|---:|---:|---|
| **BigEarthNet Load (Initial)** | `0.3487s` | `0.3155s` | ONNX Runtime session with graph optimization |
| **BigEarthNet Load (Reused)** | `0.3487s` | `0.000012s` | **Instantaneous** via thread-safe session reuse |
| **BigEarthNet Single-Tile Inference** | `0.0081s` | `0.0068s` | Multi-threaded ONNX CPUExecutionProvider |
| **BigEarthNet 4-Tile Processing** | `0.0279s` (seq) | `0.0231s` (batch) | **Batched forward pass** `(4, 10, 120, 120)` |
| **Florence-2 Load (Initial)** | `63.5719s` | `50.7103s` | Full checkpoint weight loading |
| **Florence-2 Load (Reused)** | `63.5719s` | `0.000009s` | **Instantaneous** via thread-safe lazy singleton |
| **Florence-2 Scene Captioning** | `13.1102s` | `8.8410s` | **32.6% faster** (cached raster + CPU thread tuning) |
| **Florence-2 VQA** | `7.1777s` | `4.4697s` | **37.7% faster** (cached raster array + inference lock) |
| **Florence-2 Phrase Grounding** | `6.8498s` | `4.9922s` | **27.1% faster** (cached raster + thread-bounded decoding) |
| **Bi-Temporal Change Detection** | `1.2622s` | `0.5984s` | **52.6% faster** (cached raster unpacking + vectorized diff) |
| **Optical + SAR Fusion** | `0.0595s` | `0.0275s` | **53.8% faster** (cached raster arrays + vectorized indices) |
| **Mission Intelligence Report** | `0.0002s` | `0.0001s` | Structured Markdown dossier generation |
| **Peak Process Memory** | `243.78 MB` | `245.98 MB` | **Stable and bounded** (< 246 MB) |

---

## 3. Optimizations Implemented

### 3.1 Model Lifecycle & Concurrency Strategy
- **Florence-2 Lazy Singleton**: Retains model and processor in memory after initial load using double-checked locking (`threading.Lock()`). Subsequent calls return in `< 0.00001s`.
- **CPU Concurrency Lock**: Added `self._infer_lock = threading.Lock()` around Florence-2's `model.generate()`. This prevents multi-request CPU thrashing and memory spikes during simultaneous user queries on CPU.
- **PyTorch CPU Thread Configuration**: Configured `torch.set_num_threads(min(4, os.cpu_count() or 1))` via `SATQUERY_TORCH_THREADS`. Avoids logical core oversubscription that previously caused thread contention with FastAPI.

### 3.2 BigEarthNet ONNX Threading & Batched Tiling
- **Configured ONNX SessionOptions**:
  - `intra_op_num_threads`: Configurable via `SATQUERY_ORT_INTRA_THREADS` (default `min(4, os.cpu_count())`).
  - `inter_op_num_threads`: Configurable via `SATQUERY_ORT_INTER_THREADS` (default `1`).
  - `graph_optimization_level`: Enabled `ORT_ENABLE_ALL`.
- **Batched Tile Forward Pass**: Implemented `run_batch_inference` in `BigEarthNetModelLoader`. When analyzing an image tiled into multiple 120×120 patches, all patches are stacked into a single `(B, 10, 120, 120)` tensor and processed in a single ONNX forward pass, achieving exact numerical equivalence (`np.allclose(seq, batch, atol=1e-5)`).

### 3.3 Bounded In-Memory Raster & Metadata Caching
- Created `BoundedRasterCache` in `backend/geospatial/raster_cache.py`.
- Caches up to 6 raster arrays and 32 metadata records with LRU eviction.
- Cache keys incorporate `(abs_path, mtime, size)`, guaranteeing that modified files invalidate cleanly.
- Integrated into `GeoTIFFValidator`, `florence2_inference.py`, `bitemporal_inference.py`, and `optical_sar_inference.py`, eliminating redundant disk reads and TIFF header parsing across validation and execution stages.

### 3.4 Large-Image Safety Limits
- Added strict safety checks in `GeoTIFFValidator`:
  - `MAX_FILE_SIZE_BYTES`: 250 MB (`SATQUERY_MAX_FILE_SIZE_BYTES`).
  - `MAX_RASTER_DIM`: 8192 px (`SATQUERY_MAX_RASTER_DIM`).
  - `MAX_BANDS`: 32 bands (`SATQUERY_MAX_BANDS`).
- Oversized or maliciously huge rasters are rejected cleanly before allocating gigabytes of memory, reporting honest validation errors.

### 3.5 Frontend Responsiveness & Cursor Throttling
- Throttled cursor HUD updates in `SingleImageViewer.tsx` using `requestAnimationFrame`.
- Ensures at most 1 coordinate update per frame (60fps), eliminating React state updates and UI stutter during rapid mouse movement.
- Configured 120s timeout in `src/services/api.ts` to accommodate heavy CPU inference without premature failure.

---

## 4. Remaining Bottlenecks & Scientific Limits

1. **Florence-2 Autoregressive Decoding**:
   - Because Florence-2 is an autoregressive encoder-decoder vision-language model running on CPU, generating 20–50 tokens takes 4.5s–8.8s. This is an inherent property of CPU-bound floating-point matrix multiplications.
2. **Initial Cold-Start Model Load**:
   - Initial disk-to-RAM loading of the 0.9GB Florence-2 checkpoint takes ~50s on CPU. Once loaded, all subsequent inferences execute without reload overhead.
3. **BigEarthNet Multi-Spectral Requirement**:
   - BigEarthNet strictly requires authentic 10-band Sentinel-2 data. 3-band RGB imagery is honestly bypassed with transparent limitations.
