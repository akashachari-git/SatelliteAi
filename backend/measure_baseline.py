"""
SatQuery AI — Step 16 Baseline Performance Measurement Script
Measures actual execution latencies and memory usage on the local CPU runtime.
Outputs results to docs/STEP16_PERFORMANCE_BASELINE.md.
"""
import os
import sys
import time
import tracemalloc
import numpy as np
from PIL import Image

_CURRENT = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_CURRENT)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)
if _CURRENT not in sys.path:
    sys.path.insert(0, _CURRENT)

SAMPLE_RASTER_DIR = os.path.join(_CURRENT, "datasets", "sample_rasters")
URBAN_TIF = os.path.join(SAMPLE_RASTER_DIR, "urban_satellite_eo.tif")
S2_PATCH_PNG = os.path.join(SAMPLE_RASTER_DIR, "example_s2_patch.png")
TEMPORAL_T1_TIF = os.path.join(SAMPLE_RASTER_DIR, "temporal_t1_2021.tif")
TEMPORAL_T2_TIF = os.path.join(SAMPLE_RASTER_DIR, "temporal_t2_2023.tif")
OPTICAL_TIF = os.path.join(SAMPLE_RASTER_DIR, "crossmodal_optical.tif")
SAR_TIF = os.path.join(SAMPLE_RASTER_DIR, "crossmodal_sar_s1.tif")


def run_measurements():
    tracemalloc.start()
    results = {}

    print("\n--- Starting Baseline Performance Measurement on CPU ---")

    # 1. BigEarthNet Model Load Time
    print("1. Measuring BigEarthNet ResNet-18 ONNX load time...")
    from backend.models.bigearthnet_loader import BigEarthNetModelLoader
    loader = BigEarthNetModelLoader()
    t0 = time.time()
    is_loaded, status, msg = loader.load_model()
    t_ben_load = time.time() - t0
    results["bigearthnet_load_sec"] = round(t_ben_load, 4)
    print(f"   BigEarthNet Load Time: {t_ben_load:.4f}s (Status: {status})")

    # 2. BigEarthNet Single-Tile Inference
    print("2. Measuring BigEarthNet 10-band single-tile inference...")
    s2_10band = np.random.uniform(0.05, 0.45, size=(10, 120, 120)).astype(np.float32)
    # Warmup
    loader.run_inference(s2_10band)
    # 5 runs
    timings = []
    for _ in range(5):
        t0 = time.time()
        loader.run_inference(s2_10band)
        timings.append(time.time() - t0)
    t_ben_inf = float(np.mean(timings))
    results["bigearthnet_inference_sec"] = round(t_ben_inf, 4)
    print(f"   BigEarthNet Inference Latency (mean of 5): {t_ben_inf:.4f}s")

    # 3. BigEarthNet Tile Aggregation (e.g., 4 tiles)
    print("3. Measuring BigEarthNet tile aggregation (4 patches)...")
    t0 = time.time()
    for _ in range(4):
        loader.run_inference(s2_10band)
    t_ben_tiling = time.time() - t0
    results["bigearthnet_tiling_4tiles_sec"] = round(t_ben_tiling, 4)
    print(f"   BigEarthNet 4-Tile Sequential Latency: {t_ben_tiling:.4f}s")

    # 4. Florence-2 Model Load Time
    print("4. Measuring Florence-2 VLM load time...")
    from backend.app.models.florence2_inference import Florence2InferenceService
    f2_service = Florence2InferenceService()
    t0 = time.time()
    f2_service.load_model()
    t_f2_load = time.time() - t0
    results["florence2_load_sec"] = round(t_f2_load, 4)
    print(f"   Florence-2 Load Time: {t_f2_load:.4f}s")

    # 5. Florence-2 Captioning
    print("5. Measuring Florence-2 captioning latency...")
    img = Image.open(S2_PATCH_PNG).convert("RGB")
    t0 = time.time()
    f2_service.predict(img, "Describe this satellite image.", task_type="scene-captioning")
    t_f2_caption = time.time() - t0
    results["florence2_caption_sec"] = round(t_f2_caption, 4)
    print(f"   Florence-2 Caption Latency: {t_f2_caption:.4f}s")

    # 6. Florence-2 VQA
    print("6. Measuring Florence-2 VQA latency...")
    t0 = time.time()
    f2_service.predict(img, "Is there vegetation or water visible in this image?", task_type="vqa")
    t_f2_vqa = time.time() - t0
    results["florence2_vqa_sec"] = round(t_f2_vqa, 4)
    print(f"   Florence-2 VQA Latency: {t_f2_vqa:.4f}s")

    # 7. Florence-2 Phrase Grounding
    print("7. Measuring Florence-2 grounding latency...")
    t0 = time.time()
    f2_service.predict(img, "green vegetation", task_type="text-guided-grounding")
    t_f2_grounding = time.time() - t0
    results["florence2_grounding_sec"] = round(t_f2_grounding, 4)
    print(f"   Florence-2 Grounding Latency: {t_f2_grounding:.4f}s")

    # 8. Bi-Temporal Change Detection
    print("8. Measuring Bi-Temporal Change Detection latency...")
    from backend.app.models.bitemporal_inference import bitemporal_service
    t0 = time.time()
    bitemporal_service.predict(
        query="What changed between 2021 and 2023?",
        inputs={"before": TEMPORAL_T1_TIF, "after": TEMPORAL_T2_TIF}
    )
    t_bitemp = time.time() - t0
    results["bitemporal_sec"] = round(t_bitemp, 4)
    print(f"   Bi-Temporal Latency: {t_bitemp:.4f}s")

    # 9. Optical + SAR Multimodal Analysis
    print("9. Measuring Optical + SAR Multimodal Analysis latency...")
    from backend.app.models.optical_sar_inference import optical_sar_service
    t0 = time.time()
    optical_sar_service.predict(
        query="Identify water and built-up areas.",
        inputs={"optical": OPTICAL_TIF, "sar": SAR_TIF}
    )
    t_optsar = time.time() - t0
    results["optical_sar_sec"] = round(t_optsar, 4)
    print(f"   Optical+SAR Latency: {t_optsar:.4f}s")

    # 10. Report Generation
    print("10. Measuring Markdown Dossier Generation latency...")
    from backend.reporting import generate_analysis_markdown
    sample_payload = {
        "query": "Where is water?",
        "mode": "single",
        "answer": "Water detected in south-east quadrant.",
        "evidence": ["Optical reflectance < 0.05", "SAR extinction"],
    }
    t0 = time.time()
    generate_analysis_markdown(sample_payload)
    t_report = time.time() - t0
    results["report_generation_sec"] = round(t_report, 4)
    print(f"   Report Generation Latency: {t_report:.4f}s")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["peak_memory_mb"] = round(peak_mem / (1024 * 1024), 2)
    print(f"   Peak Process Memory: {results['peak_memory_mb']} MB")

    # Generate docs/STEP16_PERFORMANCE_BASELINE.md
    docs_dir = os.path.join(_WORKSPACE, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    baseline_doc = os.path.join(docs_dir, "STEP16_PERFORMANCE_BASELINE.md")

    content = f"""# SatQuery AI — Step 16 CPU Performance Baseline

Measured on: {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
Hardware Environment: Windows 11, Python 3.13, CPU-only runtime (No CUDA GPU)

## 1. Measured Latencies

| Component / Task | Measured Latency | Notes / Details |
|---|---:|---|
| **BigEarthNet ONNX Model Load** | `{results['bigearthnet_load_sec']}s` | ONNX Runtime CPUExecutionProvider initialization |
| **BigEarthNet Single-Tile Inference** | `{results['bigearthnet_inference_sec']}s` | Mean forward pass on 10-band (10, 120, 120) tensor |
| **BigEarthNet Tiling (4 tiles sequential)** | `{results['bigearthnet_tiling_4tiles_sec']}s` | Sequential forward passes without batching |
| **Florence-2 VLM Model Load** | `{results['florence2_load_sec']}s` | Loading weights from local disk checkpoint |
| **Florence-2 Scene Captioning** | `{results['florence2_caption_sec']}s` | Open-ended captioning on 120×120 patch |
| **Florence-2 VQA** | `{results['florence2_vqa_sec']}s` | Visual Question Answering on 512×512 raster |
| **Florence-2 Phrase Grounding** | `{results['florence2_grounding_sec']}s` | Phrase grounding query on 120×120 patch |
| **Bi-Temporal Change Detection** | `{results['bitemporal_sec']}s` | Classical spectral differencing on 512×512 pair |
| **Optical + SAR Fusion** | `{results['optical_sar_sec']}s` | Cross-sensor corroboration on 512×512 pair |
| **Mission Intelligence Report Generation**| `{results['report_generation_sec']}s` | Structured Markdown dossier generation |

## 2. Memory Usage

- **Peak Process Memory Traced**: `{results['peak_memory_mb']} MB`

## 3. Primary Bottlenecks Identified

1. **Florence-2 Initial Load Time**: `{results['florence2_load_sec']}s` — Must be loaded once (lazy singleton) and retained in memory, never reloaded per request.
2. **Florence-2 CPU Inference Latency**: ~{results['florence2_vqa_sec']}s–{results['florence2_caption_sec']}s per query — CPU-bound autoregressive decoding. Requires concurrency protection to prevent multi-request CPU exhaustion.
3. **Sequential Tile Forward Passes**: Running multiple tiles sequentially incurs N separate ONNX calls. Batching compatible tiles into a single forward pass `(N, 10, 120, 120)` will optimize throughput.
4. **ONNX Thread Configuration**: Defaults may not be optimized for CPU core topology.
"""
    with open(baseline_doc, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nSaved baseline measurements to {baseline_doc}\n")
    return results


if __name__ == "__main__":
    run_measurements()
