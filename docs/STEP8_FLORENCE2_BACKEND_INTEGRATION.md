# STEP 8 — Florence-2-base VLM Backend Integration Report

**Status:** Completed  
**Date:** September 20, 2026  
**Environment:** Windows, Python 3.13.3, PyTorch 2.14.0+cpu, Transformers 4.57.6, ONNX Runtime 1.24.3  
**Model:** `microsoft/Florence-2-base` (231.4M parameters)  
**Execution Runtime:** Local CPU (`torch.float32`)  

---

## 1. Executive Summary

In Step 8, the locally verified **Microsoft Florence-2-base** Vision-Language Model was integrated directly into the production SatQuery AI backend pipeline.

Key achievements:
1. **Dedicated Specialist Service**: Implemented [florence2_inference.py](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/backend/app/models/florence2_inference.py) providing lazy loading, CPU execution, local checkpoint loading, and clean task execution methods for captioning, VQA, and phrase grounding.
2. **Specialist Registry Registration**: Registered `florence2-vlm` in [tools.py](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/backend/registry/tools.py) with explicit capabilities (`image_captioning`, `visual_question_answering`, `phrase_grounding`).
3. **Autonomous Agent Routing**: Updated [agent.py](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/backend/orchestrator/agent.py) to route scene captioning, visual question answering, and phrase grounding queries to Florence-2, while strictly routing land-cover classification queries to BigEarthNet ResNet-18.
4. **Scientifically Honest Input Handling**: Native RGB images are ingested directly. For 10-band Sentinel-2 rasters, an authentic true-color RGB composite (B04-B03-B02) is derived with percentile contrast normalization, with explicit provenance in the audit trail distinguishing the optical visualization from raw 10-band multispectral data.
5. **No Fabricated Outputs**: Open-ended VLM responses omit fake confidence numbers (`confidence = None`), and bounding box coordinates remain strictly in image pixel space (no guessed geographic coordinates).
6. **Zero Regressions**: 53/53 backend tests pass (41 existing BigEarthNet/pipeline tests + 12 new Florence-2 integration tests).

---

## 2. Files Changed & Created

| File | Status | Description |
| :--- | :---: | :--- |
| `backend/app/models/florence2_inference.py` | **NEW** | Dedicated Florence-2 specialist service with lazy loading, image preparation, task methods, and error handling. |
| `backend/tests/test_florence2_pipeline.py` | **NEW** | Focused integration tests covering lazy loading, registry selection, intent classification, image preparation, task execution, BigEarthNet isolation, and error handling. |
| `backend/schemas.py` | **MODIFIED** | Updated `confidence` in `BoundingBox` and `AnalyzeResponse` to `Optional[float] = None`. |
| `backend/registry/tools.py` | **MODIFIED** | Added `florence2-vlm` to `TOOL_REGISTRY` and updated `select_tool_for_task` to route captioning, VQA, and phrase grounding to Florence-2. |
| `backend/orchestrator/agent.py` | **MODIFIED** | Integrated `florence2_service`, updated `classify_task` keywords, added Florence execution with audit traces, and ensured no fake confidence or geocoordinates. |
| `docs/STEP8_FLORENCE2_BACKEND_INTEGRATION.md` | **NEW** | This integration report. |

---

## 3. BigEarthNet vs. Florence-2 Responsibilities

| Dimension | BigEarthNet ResNet-18 Specialist | Florence-2 Vision-Language Specialist |
| :--- | :--- | :--- |
| **Model Type** | 10-Band Multispectral Classifier (ONNX) | Vision-Language Foundation Model (PyTorch CPU) |
| **Target Queries** | "What land-cover classes are present?", "Classify this Sentinel-2 image", "What type of land cover is this?" | "Describe this image", "Is there water?", "Where are the buildings?", "Locate roads" |
| **Tasks** | `land-cover-classification`, `vegetation-classification`, `corine-clc-mapping` | `scene-captioning` (`image_captioning`), `vqa` (`visual_question_answering`), `text-guided-grounding` (`phrase_grounding`) |
| **Input Modality** | Authentic 10-band Sentinel-2 raster (B02-B12) | Native Optical RGB or Derived B04-B03-B02 RGB Composite |
| **Output Contract** | 19 CORINE multi-label probabilities, spatial tile aggregation, dominant classes | Natural language caption, conversational answer, or pixel bounding boxes `[x, y, w, h]` |
| **Confidence** | Calibrated Sigmoid probability (e.g. `84.2%`) | `None` / Not applicable (generative sequence) |
| **Spatial Output** | Tile spatial window indices, coverage fractions | Pixel bounding boxes in image pixel coordinate space |

---

## 4. Routing Logic

The agent router in [agent.py](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/backend/orchestrator/agent.py) operates autonomously across query semantics and input modalities:

```
User Query + Uploaded Image
             │
             ├── Mode == "bi-temporal" or 2 images ──> ChangeFormer-V2 ("change-analysis" / "change-based-vqa")
             │
             ├── Mode == "optical-sar" or SAR band ──> CrossSens-Fusion ("optical-sar-analysis")
             │
             └── Single Image Mode:
                   │
                   ├── Keywords: "land cover", "classify", "corine", "clc", "bigearthnet"
                   │     └──> BigEarthNet ResNet-18 (Requires 10-band Sentinel-2; else deterministic fallback)
                   │
                   ├── Keywords: "describe", "caption", "what is shown", "scene description", "overview"
                   │     └──> Florence-2 Captioning (<MORE_DETAILED_CAPTION>)
                   │
                   ├── Keywords: "where are", "where is", "locate", "find", "show me", "highlight"
                   │     └──> Florence-2 Phrase Grounding (<CAPTION_TO_PHRASE_GROUNDING> / <OPEN_VOCABULARY_DETECTION>)
                   │
                   └── All other visual questions (e.g. "Is there water?", "Are there buildings?")
                         └──> Florence-2 VQA (<VQA>)
```

---

## 5. Input Preprocessing & Scientific Honesty

In [florence2_inference.py](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/backend/app/models/florence2_inference.py), `Florence2InferenceService.prepare_image()` enforces strict scientific honesty:

1. **Native Optical RGB (PNG, JPEG, RGB TIFF)**:
   - Ingested directly into PIL `Image.Image` in `RGB` mode.
   - Modality labeled: `"Optical RGB"`.
   - `is_multispectral_derived = False`.
2. **10-Band Sentinel-2 Rasters**:
   - Bands in standard order: `B02` (idx 0), `B03` (idx 1), `B04` (idx 2).
   - Authentic True-Color RGB composite constructed as `[B04 (Red), B03 (Green), B02 (Blue)]`.
   - Scaled using 2nd to 98th percentile contrast stretching into `uint8`.
   - Modality labeled: `"Sentinel-2 Derived True-Color RGB Composite (B04-B03-B02)"`.
   - `is_multispectral_derived = True`.
   - Audit trail records: *"Visual interpretation operated on derived B04-B03-B02 true-color composite; full 10-band multispectral data is preserved for BigEarthNet specialist."*
3. **Invalid / Incompatible Input**:
   - If input cannot be decoded into a valid image, raises an explicit `ValueError`.
   - **Never silently falls back to fake or static template outputs.**

---

## 6. Execution Trace & Audit System Integration

Every Florence-2 execution produces comprehensive `ExecutionTraceStep` entries in the response:
- **Selected Specialist**: `Florence-2 Vision-Language Specialist`
- **Model Architecture**: `microsoft/Florence-2-base (231.4M params)`
- **Task Classification**: `Image Captioning`, `Visual Question Answering`, or `Text-Guided Phrase Grounding`
- **Input Modality**: Details whether native RGB or Sentinel-2 derived RGB composite
- **Execution Status**: `Completed (Local CPU PyTorch float32)`
- **Inference Latency**: Precise measured duration in ms
- **Multispectral Provenance**: Tagged as `Derived B04-B03-B02 RGB Composite` or `Native RGB`
- **Coordinate System**: Explicitly recorded as `Image Pixel Coordinates (No synthetic georeferencing)`
- **Confidence Status**: Explicitly recorded as `Not applicable (Open-ended VLM generation)`

---

## 7. Performance & Latency Observations

Measurements from standalone verification and live test execution on local CPU:

| Task | Prompt / Method | Measured Latency | Memory (Delta) |
| :--- | :--- | :--- | :--- |
| **Model Load** | Local checkpoint cold load | **3.73 s** | +1,378.9 MB RAM |
| **Detailed Caption** | `<DETAILED_CAPTION>` | **11.60 s** | Peak ~1.4 GB |
| **More Detailed Caption**| `<MORE_DETAILED_CAPTION>` | **15.58 s** | Peak ~1.4 GB |
| **VQA** | `<VQA>` | **9.40 s – 9.43 s** | Peak ~1.4 GB |
| **Phrase Grounding** | `<CAPTION_TO_PHRASE_GROUNDING>` | **8.90 s – 9.01 s** | Peak ~1.4 GB |

---

## 8. Test Verification Results

### A. Dedicated Florence-2 Test Suite (`test_florence2_pipeline.py`)
```
Ran 12 tests in 0.423s
OK
```
- Lazy loading & availability: **Passed**
- Tool registry selection: **Passed**
- Query intent classification: **Passed**
- Image preparation (Native RGB): **Passed**
- Image preparation (Sentinel-2 10-band composite): **Passed**
- Caption output schema: **Passed**
- VQA output schema & sanitization: **Passed**
- Grounding pixel coordinates: **Passed**
- Agent pipeline routing & trace: **Passed**
- BigEarthNet preservation for land-cover: **Passed**
- Invalid input error handling: **Passed**
- Missing model error handling: **Passed**

### B. Complete Backend Test Suite
```
Ran 53 tests in 19.867s
OK
```
All 41 existing BigEarthNet/pipeline tests and all 12 new Florence-2 integration tests pass without a single regression or failure.

---

## 9. Known Limitations

1. **CPU Latency (~9–15s)**:
   - On a CPU-only host without CUDA acceleration, generative beam search takes ~9–15 seconds per query. This is acceptable for asynchronous Earth observation analysis but should be taken into account when designing UI loading states.
2. **Domain Nuances**:
   - Florence-2 is a general-purpose vision-language foundation model. While it recognizes aerial and top-down perspectives effectively, it is not specialized or fine-tuned on multispectral satellite bands.
   - For quantitative land-cover analysis, BigEarthNet remains the authoritative, scientifically verified 10-band specialist.
3. **RGB-Only Vision Encoder**:
   - Florence-2 cannot ingest non-optical bands (SWIR, Red Edge, SAR) directly. Any non-RGB inputs must be converted to scientifically honest true-color or false-color composites with explicit provenance.
