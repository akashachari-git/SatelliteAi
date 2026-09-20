# SatQuery AI — Step 7A: Remote-Sensing VLM / VQA / Captioning / Grounding Layer Audit

**Audit Date**: September 2026  
**Auditor**: Antigravity AI  
**Scope**: Complete repository inspection of remote-sensing Vision-Language Models (VLMs), Visual Question Answering (VQA), Scene Captioning, Text-Guided Grounding, and Multimodal Analytics.

---

## 1. Executive Summary

This audit establishes the ground truth regarding all remote-sensing AI models, vision-language capabilities, and execution paths within the SatQuery AI repository.

### Key Findings:
1. **Single Genuine Local Model**: The repository possesses **exactly one** real, operational deep learning model: the verified **BigEarthNet ResNet-18 (ONNX & PyTorch)** 10-band Sentinel-2 land-cover classifier (`backend/models/bigearthnet_resnet18_10band.onnx` and `backend/models/checkpoints/resnet18-s2-v0.2.0/model.safetensors`).
2. **Zero Genuine Local VLMs**: There are **no** real local Vision-Language Model weights, checkpoints, or active neural inference pipelines for VQA, Scene Captioning, or Text-Guided Grounding.
3. **Pervasive Mocking in Python Specialist Layer**: The classes `RSVHAModel` (`backend/models/rs_vqa.py`), `SceneCaptioningModel` (`backend/models/captioning.py`), `TextGuidedGroundingModel` (`backend/models/grounding.py`), `BiTemporalChangeModel` (`backend/models/change_detection.py`), and `OpticalSARFusionModel` (`backend/models/optical_sar_fusion.py`) rely on simple keyword regex/string matching that returns hardcoded text templates and fabricated static bounding boxes.
4. **Orphaned VLM Interface**: The modular classes in `backend/vlm/` (`BaseRemoteSensingVLM`, `RemoteSensingVLM`, `EvidenceExtractor`, `ConfidenceEstimator`) are architectural stubs; they load no model weights and are completely uncalled by the production orchestrator (`backend/orchestrator/agent.py`).
5. **External Cloud Dependency in Node.js Layer**: The Node.js Express server (`server.ts`) handles `/api/analyze` by delegating to Google's commercial **Gemini 2.5 Flash** cloud API (`@google/genai`) when `GEMINI_API_KEY` is present. When absent, it calls `backend.geospatial.local_engine`, which executes classical computer vision heuristics (Sobel edge filters, color thresholds, connected components).

---

## 2. Current VQA Capability

### A. Python Backend Implementation (`RSVHAModel` in `backend/models/rs_vqa.py`)
- **Status**: Mockup / Keyword Heuristic (Classification: **D. PLACEHOLDER/MOCK**).
- **Execution Mechanism**:
  - Checks query substring:
    - `"water"`, `"river"`, `"lake"`, `"reservoir"` $\to$ Returns hardcoded text describing an inland water body of 1.42 km², NDWI +0.48, and hardcoded bounding boxes `box-1` `[22.0, 45.0, 38.0, 28.0]` and `box-2` `[64.0, 18.0, 26.0, 34.0]`.
    - `"building"`, `"built-up"`, `"urban"` $\to$ Returns hardcoded text describing 68% dense commercial blocks.
    - Default $\to$ Returns hardcoded text describing a peri-urban landscape (52% built-up, 28% canopy).
- **Model Weights**: None. `self._is_loaded` defaults to `False`. `load_weights()` merely sets a boolean flag.
- **Image Processing**: Zero image tensor processing. Pixel values are never read or passed to any neural network.

### B. Stub Implementation (`RemoteSensingVLM` in `backend/vlm/rs_vqa.py`)
- **Status**: Unused Architectural Stub / Keyword Heuristic (Classification: **D. PLACEHOLDER/MOCK**).
- **Execution Mechanism**:
  - `classify_question_type()` matches 8 query categories (`water_bodies`, `built_up`, `vegetation`, `agricultural`, `roads`, `objects`, `spatial_relationships`, `land_cover`).
  - Returns static template paragraphs with hardcoded bounding boxes.
  - `load_checkpoint()` takes a path but loads no tensors or PyTorch/ONNX sessions.
- **Production Status**: Completely disconnected from `backend/orchestrator/agent.py` and `backend/main.py`.

### C. Node.js Layer Implementation (`server.ts`)
- **Status**: Dual Path — External Cloud API or Classical CV (Classification: **B. REAL EXTERNAL API** or **C. HEURISTIC/CLASSICAL**).
- **Execution Mechanism**:
  - If `GEMINI_API_KEY` is set: Calls `analyzeSingleWithGemini()` using `gemini-2.5-flash` with a system prompt instructing it to act as a satellite analyst and output JSON.
  - If `GEMINI_API_KEY` is missing: Calls `runLocalEngine({ action: 'analyze_single' })`, which invokes `backend/geospatial/local_engine.py`.

---

## 3. Current Captioning Capability

### A. Python Backend Implementation (`SceneCaptioningModel` in `backend/models/captioning.py`)
- **Status**: Hardcoded Static Mockup (Classification: **D. PLACEHOLDER/MOCK**).
- **Model Name**: Registered as `GeoCaption-Net` / `rs-captioning-engine`.
- **Claimed Architecture**: `CLIP-ViT-Large/14 + Autoregressive Remote Sensing Decoder` (450M params).
- **Actual Code**:
  ```python
  answer = (
      "A high-resolution synoptic overview revealing an engineered coastal metropolitan zone. "
      "The southern half exhibits a major sheltered marine harbor with 6 industrial shipping berths..."
  )
  ```
  Returns the exact same port facility description regardless of the uploaded image.
- **Model Weights**: None exist on disk.
- **Image Processing**: Zero image tensor inspection.

### B. Node.js Layer Implementation (`server.ts`)
- If Gemini is active: Prompts Gemini to describe the scene.
- If Gemini is inactive: `local_engine.py` generates a summary string listing detected surface percentages (e.g., `"water bodies (14.2%), vegetation/canopy (35.1%)"`).

---

## 4. Current Grounding Capability

### A. Python Backend Implementation (`TextGuidedGroundingModel` in `backend/models/grounding.py`)
- **Status**: Hardcoded Static Mockup (Classification: **D. PLACEHOLDER/MOCK**).
- **Model Name**: Registered as `SatGround-DETR` / `spatial-grounding-det`.
- **Claimed Architecture**: `Deformable DETR with Linguistic Feature Grounding` (285M params).
- **Actual Code**:
  - If `"water"` in query: Returns `[x: 20.0, y: 38.0, width: 42.0, height: 34.0]`.
  - Else: Returns `[x: 35.0, y: 25.0, width: 30.0, height: 30.0]`.
- **Model Weights**: None exist on disk.
- **Image Processing**: Zero image tensor inspection.

### B. Local Computer Vision Engine (`backend/geospatial/local_engine.py`)
- **Status**: Genuine Classical Computer Vision (Classification: **C. HEURISTIC/CLASSICAL**).
- **Execution Mechanism**:
  - Resizes RGB image to $512 \times 512$.
  - Computes channel proxies:
    - NDWI proxy: $(G - R) / (G + R + \epsilon)$ with darkness and blue/red ratio thresholding.
    - ExG (Excess Green): $(2G - R - B) / (R + G + B + \epsilon)$.
    - Built-up: Sobel edge magnitude filter ($>45.0$) + spatial density smoothing.
    - Roads: Morphological line structuring elements (`np.ones((1, 9))` and `np.ones((9, 1))`).
  - Calls `scipy.ndimage.label()` and `scipy.ndimage.find_objects()` on threshold masks to extract genuine bounding boxes around contiguous pixel clusters.
- **Limitation**: Not text-guided. Does not parse natural language phrases; merely selects pre-extracted masks based on query keywords.

---

## 5. Existing Remote-Sensing VLMs

A comprehensive survey of current remote-sensing vision-language models in the open-source community:

| Model | Architecture | Size / Weights | RS Adaptation / Dataset | Supported Tasks | CPU Feasibility |
|---|---|---|---|---|---|
| **RemoteCLIP** (Chanzh66) | ViT-B/32 & ViT-L/14 Dual-Encoder | 350 MB – 1.2 GB | Pretrained on RS5M (5M remote-sensing image-text pairs) | Zero-shot classification, image-text retrieval, visual grounding via similarity maps | **High** (ViT-B runs in ~150ms on CPU) |
| **GeoRSCLIP** | ViT-B/32 Dual-Encoder | 350 MB | Pretrained on RSICD, UCM, Sydney, RS5M | Cross-modal retrieval, zero-shot land-cover mapping | **High** (CPU-compatible) |
| **Microsoft Florence-2** (base / large) | Unified Sequence-to-Sequence Vision Foundation Model | 0.23B (500 MB) / 0.77B (1.5 GB) | Pretrained on 5.4B visual annotations (FLDA-5B); supports aerial/satellite imagery | **VQA, Detailed Captioning, Dense Region Captioning, Text-Guided Grounding (bounding boxes)** | **Very High** (base model runs in ~400–800ms on CPU) |
| **Qwen2-VL-2B-Instruct** | Multimodal LLM (Vision Transformer + Qwen2) | 2.2B (~4.5 GB) | General multimodal with high-resolution visual tokens | Conversational VQA, Detailed Captioning, Spatial Coordinate tokens `[ymin, xmin, ymax, xmax]` | **Medium-Low** (4–8s latency on CPU, 8GB+ RAM) |
| **GeoChat** | LLaVA-1.5 7B fine-tuned for RS | 7B (~14 GB) | Fine-tuned on RS multimodal instruction data | Conversational RS-VQA, grounding | **Infeasible on CPU** (Requires 16GB+ VRAM GPU) |
| **EarthDial / EarthVLM** | 7B Multimodal LLM | 7B (~14 GB) | RS instruction datasets | Conversational VQA, multi-temporal reasoning | **Infeasible on CPU** (Requires GPU) |

---

## 6. Model Checkpoints Actually Available

A full recursive scan of the repository was executed for all standard model weight extensions (`.pt`, `.pth`, `.onnx`, `.safetensors`, `.bin`, `.h5`):

```
FullName                                                                                       Length (Bytes)
--------                                                                                       --------------
backend/models/checkpoints/resnet18-s2-v0.2.0/model.safetensors                                44,884,748
backend/models/bigearthnet_resnet18_10band.onnx                                                44,949,860
```

### Claimed vs. Actual Checkpoint Table:

| Registry ID | Claimed Checkpoint Name | Status in Code | File Exists on Disk? | Actual File Size |
|---|---|---|---|---|
| `bigearthnet-classifier` | `bigearthnet_resnet18_10band.onnx` | "Connected" | **YES** | 44.95 MB (Verified) |
| `bigearthnet-classifier` | `resnet18-s2-v0.2.0/model.safetensors` | "Mounted" | **YES** | 44.88 MB (Verified) |
| `rs-vqa-transformer` | `swin_l_roberta_rs_vqa_v2.4.pt` | "Connected" | **NO** | 0 bytes (Missing) |
| `rs-captioning-engine` | `rs_captioner_cross_attn_v2.4.onnx` | "Available" | **NO** | 0 bytes (Missing) |
| `spatial-grounding-det` | `rs_grounder_detr_r50.pt` | "Connected" | **NO** | 0 bytes (Missing) |
| `bitemporal-diff-net` | `changeformer_v2_siamese.pt` | "Demo" | **NO** | 0 bytes (Missing) |
| `change-vqa-specialist` | `cdvqa_temporal_reasoner.pt` | "Available" | **NO** | 0 bytes (Missing) |
| `optical-sar-fusion-net` | `crosssens_fusion_resnet_unet.pt` | "Connected" | **NO** | 0 bytes (Missing) |
| `rs-foundational-vlm` | `EarthVLM-7B` | "Planned" | **NO** | 0 bytes (Missing) |

---

## 7. Real vs Heuristic vs Mock Classification

Every candidate model, specialist, and engine in the repository is classified strictly into one of the four mandated categories:

- **A. REAL LOCAL MODEL**: Genuine local weights loaded into memory executing real tensor forward passes.
- **B. REAL EXTERNAL MODEL/API**: External neural model or API called via network.
- **C. HEURISTIC/CLASSICAL**: Classical image processing, spectral indices, thresholds, connected components, or rule engines.
- **D. PLACEHOLDER/MOCK**: Static, synthetic, hardcoded, or demo-only results.

| Candidate Name | File / Location | Classification | Justification |
|---|---|---|---|
| **BigEarthNet ResNet-18 (ONNX)** | `backend/models/bigearthnet_loader.py` | **A. REAL LOCAL MODEL** | Genuine 11.2M parameter ResNet-18 ONNX model loaded into ONNX Runtime CPU session. Real 19-class logits and sigmoid probabilities. |
| **BigEarthNet ResNet-18 (PyTorch)** | `backend/verify_checkpoint.py` | **A. REAL LOCAL MODEL** | Genuine PyTorch checkpoint loaded via `ConfigILM`/`timm`. Numerically verified against ONNX. |
| **BigEarthNet Spatial Tiler** | `backend/app/models/bigearthnet_tiler.py` | **A. REAL LOCAL MODEL** | Partitions large 10-band rasters into $120 \times 120$ windows, batches into real ONNX forward pass. |
| **Gemini 2.5 Flash API** | `server.ts` (lines 408–795) | **B. REAL EXTERNAL MODEL/API** | Commercial cloud API call to Google GenAI (`gemini-2.5-flash`). General-purpose multimodal LLM, not a local or RS-specialized model. |
| **Local CV & Spectral Engine** | `backend/geospatial/local_engine.py` | **C. HEURISTIC/CLASSICAL** | Genuine classical computer vision using Pillow, NumPy, and SciPy (Sobel edge detection, NDWI/GLI proxies, morphological operators, connected components). Zero neural weights. |
| **Deterministic Fallback Adapter** | `backend/app/models/adaptation.py` | **C. HEURISTIC/CLASSICAL** | Classical heuristic fallback calculating Corine Land Cover prior probabilities with basic channel ratios. |
| **RSVHAModel (VQA)** | `backend/models/rs_vqa.py` | **D. PLACEHOLDER/MOCK** | Keyword regex matching returning static hardcoded text and fixed bounding box coordinates `[box-1, box-2]`. |
| **RemoteSensingVLM (Stub)** | `backend/vlm/rs_vqa.py` | **D. PLACEHOLDER/MOCK** | Uncalled stub returning static template text and hardcoded bounding boxes. |
| **SceneCaptioningModel** | `backend/models/captioning.py` | **D. PLACEHOLDER/MOCK** | Completely static hardcoded paragraph about a port facility (`cap-box-1`). Ignores image. |
| **TextGuidedGroundingModel** | `backend/models/grounding.py` | **D. PLACEHOLDER/MOCK** | Hardcoded bounding boxes `[20, 38, 42, 34]` or `[35, 25, 30, 30]`. Ignores image. |
| **BiTemporalChangeModel** | `backend/models/change_detection.py` | **D. PLACEHOLDER/MOCK** | Keyword matching returning hardcoded change statistics (`+32.4%`, `2.85 km²`) and static change masks. |
| **OpticalSARFusionModel** | `backend/models/optical_sar_fusion.py` | **D. PLACEHOLDER/MOCK** | Keyword matching returning hardcoded text citations and static bounding boxes. |
| **EarthVLM-7B** | `backend/registry/tools.py` | **D. PLACEHOLDER/MOCK** | Purely aspirational catalog metadata entry. Zero implementation code or weights. |

---

## 8. Current `/api/analyze` Execution Path

The repository currently exhibits **two completely independent backend server architectures**:

```mermaid
graph TD
    User([User Natural Language Query + Image Upload]) --> Client[Frontend UI]
    
    subgraph Path A: Node.js Express Server (server.ts on Port 3000)
        Client -->|POST /api/analyze| Express[server.ts Router]
        Express --> CheckKey{GEMINI_API_KEY Configured?}
        CheckKey -->|Yes| GeminiCall[Gemini 2.5 Flash Cloud API]
        CheckKey -->|No| LocalEngine[python -m backend.geospatial.local_engine]
        LocalEngine --> ClassicalCV[Sobel Edges + NDWI/GLI Proxies + Connected Components]
        GeminiCall --> ExpressResponse[Express Formatted Dossier + Mock Trace]
        ClassicalCV --> ExpressResponse
    end

    subgraph Path B: Python FastAPI Backend (backend/main.py on Port 8000)
        Client -.->|POST /api/analyze| FastAPI[FastAPI Router]
        FastAPI --> Agent[SatQueryAgent.execute_pipeline]
        Agent --> Classify[Task Classifier]
        Classify -->|land-cover-classification| CheckS2{10-Band Sentinel-2?}
        CheckS2 -->|Yes| ONNX[Real BigEarthNet ResNet-18 ONNX Tiler Engine]
        CheckS2 -->|No| Fallback[Deterministic Fallback Heuristic]
        Classify -->|vqa| MockVQA[RSVHAModel - Keyword Mock]
        Classify -->|scene-captioning| MockCap[SceneCaptioningModel - Static Mock]
        Classify -->|text-guided-grounding| MockGrd[TextGuidedGroundingModel - Static Mock]
        Classify -->|change-analysis| MockChg[BiTemporalChangeModel - Keyword Mock]
        Classify -->|optical-sar-analysis| MockFus[OpticalSARFusionModel - Keyword Mock]
        ONNX --> FastAPIResponse[FastAPI AnalyzeResponse with Real Trace]
        Fallback --> FastAPIResponse
        MockVQA --> FastAPIResponse
        MockCap --> FastAPIResponse
        MockGrd --> FastAPIResponse
        MockChg --> FastAPIResponse
        MockFus --> FastAPIResponse
    end
```

### Critical Architectural Observation:
- When running through `server.ts` (the default dev server for the web app), requests **never reach** the FastAPI backend or `backend/orchestrator/agent.py`.
- In `server.ts`, if `GEMINI_API_KEY` is present, it uses Gemini 2.5 Flash for VQA, captioning, and grounding. If absent, it uses `backend.geospatial.local_engine` (classical CV heuristics).
- In FastAPI (`backend/main.py`), only `land-cover-classification` uses a real neural network (BigEarthNet ONNX). All other tasks execute mockups.

---

## 9. SIH Requirement Coverage

Assessment of repository capabilities against the mandatory Smart India Hackathon (SIH) Problem Statement 26167 requirements:

| Requirement | Current Implementation | Genuine? | Evidence | Gap |
|---|---|---|---|---|
| **1. Single-Image VQA** | `RSVHAModel` in Python (`backend/models/rs_vqa.py`) & Gemini in `server.ts` | **NO (Local)** / **YES (Cloud API)** | Python returns hardcoded strings for "water"/"building". `server.ts` delegates to external Gemini 2.5 Flash. | No local VLM exists. Needs a local open-source vision-language model that inspects actual image pixels and generates query-specific answers. |
| **2. Captioning / Scene Description** | `SceneCaptioningModel` in Python (`backend/models/captioning.py`) & `local_engine.py` | **NO** | Python returns static string describing a port facility regardless of image. `local_engine.py` lists surface percentages. | No local image-to-text generative vision-language model exists. |
| **3. Text-Guided Region Grounding** | `TextGuidedGroundingModel` (`backend/models/grounding.py`) & `local_engine.py` | **NO (Linguistic)** / **YES (Spectral Blobs)** | Python returns static box coordinates. `local_engine.py` generates real bounding boxes from connected components, but is NOT text-guided. | No model exists that maps referring natural-language expressions to localized bounding boxes $[x, y, w, h]$. |
| **4. Bi-Temporal Change Detection** | `BiTemporalChangeModel` (`backend/models/change_detection.py`) & `local_engine.py` | **PARTIAL** | Python returns keyword templates (`+32.4%`). `local_engine.py` computes real pixel difference $\Delta(T2, T1)$ via NumPy array subtraction. | No neural Siamese difference network or change-captioning VLM. |
| **5. Optical + SAR Analysis** | `OpticalSARFusionModel` (`backend/models/optical_sar_fusion.py`) | **NO** | Returns keyword text templates mentioning dB values. No joint cross-attention fusion tensor network exists. | True dual-stream optical reflectance + SAR backscatter fusion model is missing. |
| **6. Agentic Model / Tool Selection** | `SatQueryAgent.classify_task` & `ToolRegistry.select_tool_for_task` | **YES** | Real linguistic intent parser routes queries across tasks, modes, and image counts; emits 8-stage execution trace. | Fully operational and connected; currently routes to mocks for tasks other than land cover. |
| **7. Land-Cover Classification** | `BigEarthNetTiler` + `BigEarthNetInferenceService` | **YES** | Verified 10-band Sentinel-2 ResNet-18 ONNX model running genuine spatial window tile inference across 19 CLC classes. | Fully satisfied and verified by 41 passing tests. |

---

## 10. Recommended Model Integration Candidates

To satisfy the SIH requirements for VQA, Scene Captioning, and Text-Guided Grounding on a development machine **without an NVIDIA CUDA GPU (CPU execution only)**:

### Candidate 1: Microsoft Florence-2-base (RECOMMENDED PRIMARY CANDIDATE)
- **HuggingFace ID**: `microsoft/Florence-2-base` (or `microsoft/Florence-2-large`)
- **Architecture**: Unified sequence-to-sequence Vision Foundation Model (DaViT vision backbone + standard text encoder-decoder).
- **Model Size**: 0.23B parameters (~500 MB weights in float32 / float16).
- **Remote-Sensing Suitability**: Trained on 5.4 billion visual annotations; excels at aerial and nadir views.
- **Supported Tasks**:
  - `<CAPTION>` / `<DETAILED_CAPTION>` / `<MORE_DETAILED_CAPTION>` $\to$ Solves **Scene Captioning** natively.
  - `<VQA>` $\to$ Solves **Visual Question Answering** natively.
  - `<CAPTION_TO_PHRASE_GROUNDING>` / `<OPEN_VOCABULARY_DETECTION>` $\to$ Solves **Text-Guided Region Grounding** natively, outputting real bounding boxes $[x1, y1, x2, y2]$ for referenced text phrases!
  - `<DENSE_REGION_CAPTION>` $\to$ Simultaneously grounds and describes all major visual regions.
- **CPU Feasibility**: **Exceptional**. At 230M parameters, it runs in ~400–800ms on a modern CPU and consumes <1.5 GB RAM. Can be exported to ONNX Runtime for additional speedup.
- **Integration Difficulty**: **Low**. Standard Hugging Face `AutoModelForCausalLM` and `AutoProcessor`.

### Candidate 2: RemoteCLIP (ViT-B/32)
- **HuggingFace ID**: `chendelong/RemoteCLIP`
- **Architecture**: Contrastive Vision-Language dual-encoder (ViT-B/32).
- **Model Size**: ~350 MB weights.
- **Remote-Sensing Suitability**: Specifically pretrained on RS5M (5 million remote-sensing image-text pairs). Unmatched domain vocabulary for satellite terminology (LULC, sensors, infrastructure).
- **Supported Tasks**: Zero-shot land-cover identification, text-to-image / image-to-text retrieval, visual feature similarity ranking.
- **Limitation**: Dual-encoder, not an autoregressive text generator. Cannot generate conversational free-form answers or native bounding boxes without auxiliary heads.
- **CPU Feasibility**: **High** (~150ms per forward pass).
- **Integration Difficulty**: **Low**.

### Candidate 3: Qwen2-VL-2B-Instruct
- **HuggingFace ID**: `Qwen/Qwen2-VL-2B-Instruct`
- **Architecture**: Full Multimodal Large Language Model (2.2B parameters).
- **Supported Tasks**: Conversational VQA, rich scene captioning, spatial grounding via coordinate tokens (`<|box_start|>(y1,x1),(y2,x2)<|box_end|>`).
- **Limitation**: Large memory footprint (~4.5 GB weights). Latency on CPU is 4–10 seconds per query, requiring 8–12 GB system RAM.
- **CPU Feasibility**: **Borderline**.
- **Integration Difficulty**: **Medium**.

---

## 11. Recommended Next Implementation Step

### Recommended Action for Step 7B:
Integrate **Microsoft Florence-2-base** (or an ONNX/PyTorch CPU-optimized RS-VLM) into `backend/models/` to replace the keyword-matching mocks in:
1. `backend/models/rs_vqa.py` (Real conversational VQA)
2. `backend/models/captioning.py` (Real detailed scene captioning)
3. `backend/models/grounding.py` (Real text-guided spatial region bounding box detection)

### Why Florence-2-base:
1. **Solves Three Mandatory SIH Requirements in a Single Model**: Captioning, VQA, and phrase grounding are all supported natively by Florence-2's task prompts without training three separate networks.
2. **True Grounding**: Directly outputs pixel bounding boxes for any phrase in the query, eliminating fabricated coordinates forever.
3. **Strict CPU Feasibility**: At 230M parameters (~500 MB), it runs reliably and smoothly on the user's non-CUDA development machine without OOM risks or intolerable latency.
4. **Preserves BigEarthNet**: BigEarthNet ResNet-18 remains the specialist for multi-spectral 10-band Corine land-cover classification, while Florence-2 acts as the vision-language specialist for high-resolution visual reasoning, captioning, and referring expression grounding.

---

## 12. Verification & Test Integrity

- **Existing Tests Checked**: 41 backend tests in `backend/tests/`.
- **Test Command**: `python -m unittest discover -s backend/tests -v`
- **Test Results**: **41 / 41 PASSED (100%)**
  - `TestBigEarthNetCheckpointLoading`: 5/5
  - `TestBigEarthNetIntegration`: 7/7
  - `TestBigEarthNetONNX`: 9/9
  - `TestBigEarthNetPipeline`: 9/9
  - `TestBigEarthNetTiler`: 11/11
- **Production Code Status**: Strictly unmodified in Step 7A.
