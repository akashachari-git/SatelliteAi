# STEP 11: Agentic Orchestration & Multi-Specialist Controller

## 1. Executive Summary & Orchestration Audit

### Prior Architecture & Audit Findings
Before Step 11, SatQuery AI operated as a single-specialist router:
1. **Single-Task Mapping**: `classify_task` mapped a query to exactly one task string (e.g. `land-cover-classification` or `scene-captioning`).
2. **Single Specialist Execution**: Even if a query asked compound questions (such as *"What land cover is present and what does the image show?"*), only one specialist was invoked, ignoring the other half of the query.
3. **No Explicit Planning**: The system lacked an explicit, structured planning representation capturing detected intents, input requirements, tool dependencies, and known limitations prior to execution.
4. **No Disagreement Handling**: If two potential sources of evidence conflicted, there was no mechanism to detect, structure, or report the disagreement.
5. **No Evidence Hierarchy**: Direct measurements (e.g. bounding boxes, detected classes, spectral indices) were mixed indiscriminately with generic strings.

### Upgrades Implemented in Step 11
- **Explicit Planning Representation (`AgentPlan`)**: Introduced an internal plan capturing intents, required modalities, required image count, selected tools, execution order, dependencies, expected evidence, and operational limitations.
- **Multi-Specialist Orchestration**: Enabled compound queries to trigger multiple specialists (e.g. Florence-2 Vision-Language Specialist + BigEarthNet ResNet-18 Land-Cover Specialist) and sequentially execute them with dependency tracking.
- **Deterministic Evidence Hierarchy (`EvidenceHierarchy`)**: Separates direct specialist evidence, supporting context (e.g. land-cover priors), operational limitations, and specialist disagreements.
- **Specialist Disagreement Resolution**: Automatically detects semantic contradictions between vision-language descriptions and land-cover class predictions (e.g. water vs. urban), explicitly presenting both perspectives rather than silently discarding one.
- **Input-Aware Pre-Flight Gatekeeping**: Validates raster inputs (dimensions, band count, CRS, modality count) against the plan before executing specialists, rejecting impossible requests (e.g. 1-image bi-temporal or optical-SAR queries) with informative validation errors.
- **No Fabricated Metrics**: All confidence scores remain strictly `None` for open-ended VLM and multi-specialist synthesis.

> [!IMPORTANT]
> **Deterministic Rule-Based Controller**: SatQuery AI's orchestration is implemented as a deterministic, evidence-aware controller, **not** an unconstrained or non-deterministic large language model agent. Query understanding and tool selection use structured semantic matching, rule-based intent analysis, and strict schema validation.

---

## 2. Planning Representation (`AgentPlan`)

The internal planning representation is defined in [`backend/orchestrator/planner.py`](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/backend/orchestrator/planner.py):

```python
@dataclass
class AgentPlan:
    query: str
    intents: List[str]                  # e.g. ['caption', 'land-cover']
    primary_intent: str                 # e.g. 'scene-captioning'
    required_modalities: List[str]      # e.g. ['Optical RGB', 'Sentinel-2 (10-Band)']
    required_images: int                # e.g. 1 or 2
    selected_tools: List[str]           # e.g. ['florence2-vlm', 'bigearthnet-classifier']
    execution_order: List[str]          # Ordered tool sequence
    dependencies: Dict[str, List[str]]  # Tool execution dependencies
    expected_evidence: List[str]        # Expected evidence types
    limitations: List[str]              # Known constraints (e.g. RGB-only, unprojected)
    is_multi_specialist: bool           # True if >= 2 specialists required
```

---

## 3. Query Intent Analysis & Tool Routing

The controller analyzes queries across six canonical remote-sensing intents:

| Canonical Intent | Triggers & Keywords | Selected Tool(s) | Primary Capability |
| :--- | :--- | :--- | :--- |
| **`caption`** | "describe", "caption", "overview", "what does the image show" | `florence2-vlm` | Scene-level captioning |
| **`land-cover`** | "land cover", "corine", "clc", "vegetation class", "classify" | `bigearthnet-classifier` | 19 Corine Land Cover classes |
| **`grounding`** | "locate", "where is", "where are", "find", "bounding box" | `florence2-vlm` | Phrase-guided spatial grounding |
| **`vqa`** | "what", "how many", "is there", "are there", "why" | `florence2-vlm` | Visual question answering |
| **`bi-temporal`** | "compare", "what changed", "temporal", "increase/decrease" | `bitemporal-diff-net` | Normalized differencing + dual-state prior |
| **`optical-sar`** | "optical and sar", "optical and radar", "what does sar reveal" | `optical-sar-fusion-net` | Cross-sensor physical corroboration |

### Multi-Specialist Combinations
1. **Caption + Land-Cover**:
   - Query: *"What land cover is present and what does the image show?"*
   - Query: *"Describe this image and identify the land-cover classes."*
   - Execution: Florence-2 captioning $\rightarrow$ BigEarthNet classification $\rightarrow$ Evidence Combiner.
2. **Grounding + Land-Cover Support**:
   - Query: *"Where are the buildings?"* (on 10-band Sentinel-2)
   - Execution: Florence-2 grounding $\rightarrow$ BigEarthNet urban class corroboration.
3. **Bi-Temporal + BigEarthNet Dual-State Prior**:
   - Query: *"Compare these two images and tell me what changed."* (on 10-band Sentinel-2 pair)
   - Execution: Normalized differencing $\rightarrow$ BigEarthNet T1 and T2 classification $\rightarrow$ Class probability shifts.

---

## 4. Evidence Hierarchy & Disagreement Resolution

Defined in [`backend/orchestrator/evidence_combiner.py`](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/backend/orchestrator/evidence_combiner.py):

```
                        Evidence Hierarchy
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
[Direct Evidence]    [Supporting Evidence]       [Limitations]
- Florence-2 caption - BigEarthNet land-cover   - Missing bands (RGB only)
- BigEarthNet top-3    context for grounding   - Unprojected pixel space
- Spatial bounding   - Dual-state CLC shifts   - Approximate resampling
  boxes & masks        for bi-temporal change  - Out-of-domain query
- Spectral indices
                               │
                               ▼
               [Disagreement Detection Engine]
  Checks for semantic contradictions between VLM and CLC predictions:
  ├── Case 1: Florence observes water, but BigEarthNet predicts Urban (>50%)
  ├── Case 2: Florence observes no buildings, but BigEarthNet predicts Urban (>50%)
  └── Case 3: Florence observes urban city, but BigEarthNet predicts Forest/Water (>50%)
                               │
                               ▼
                   [Synthesized Final Answer]
  If conflict: explicitly reports both perspectives without picking a side.
  If consistent: combines direct and supporting evidence seamlessly.
```

---

## 5. Input Validation & Guardrails

The controller validates all inputs before executing any neural network or classical pipeline:
1. **Image Count Guardrail**:
   - Bi-temporal change queries (`bi-temporal`) require exactly 2 temporal observations. Providing 1 image raises `ValueError("Input Validation Error: Bi-temporal change analysis requires 2 observations...")`.
   - Optical+SAR queries (`optical-sar`) require 1 Optical and 1 SAR observation. Providing 1 image raises `ValueError("Input Validation Error: Optical + SAR analysis requires 2 observations...")`.
2. **Band Integrity Guardrail**:
   - BigEarthNet ResNet-18 requires a 10-band Sentinel-2 array ($B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12$).
   - If an RGB image is submitted, the system does not pretend deep learning occurred; it notes the limitation and uses the classical statistical baseline with clear provenance.

---

## 6. Verification & Test Results

### Focused Orchestrator Planning Tests: `backend/tests/test_orchestrator_planning.py`
1. `test_01_single_specialist_routing`: Single intents route to designated specialists.
2. `test_02_multi_specialist_planning`: Compound queries produce multi-specialist plans.
3. `test_03_input_validation_before_execution`: Feasibility checks reject impossible plans early.
4. `test_04_caption_and_land_cover_combination`: Multi-specialist caption + land cover execution and evidence synthesis.
5. `test_05_temporal_and_bigearthnet_combination`: Bi-temporal change incorporates dual-state BigEarthNet prior.
6. `test_06_optical_sar_routing`: Optical+SAR queries route exclusively to optical-sar specialist.
7. `test_07_missing_image_validation`: 1-image comparison or fusion queries rejected with `ValueError`.
8. `test_08_insufficient_band_validation`: RGB inputs reject BigEarthNet deep learning with transparent limitation.
9. `test_09_disagreement_detection_and_reporting`: Conflicting specialist findings are detected and reported.
10. `test_10_structured_execution_trace`: Trace contains genuine step metrics, durations, and statuses.
11. `test_11_no_fabricated_confidence`: Confidence is strictly `None`.
12. `test_12_unsupported_query_handling`: Out-of-domain queries record clear limitations.

### Complete Backend Test Suite:
```
python -m unittest discover -s backend/tests -v
Ran 90 tests in 28.4s — OK (0 failures, 0 errors)
```
- **41/41** BigEarthNet ResNet-18 tests passing
- **12/12** Florence-2 VLM tests passing
- **12/12** Bi-temporal change analysis tests passing
- **13/13** Optical+SAR multimodal tests passing
- **12/12** Orchestrator planning & multi-specialist tests passing
- **Total: 90/90 passing tests**

---

## 7. Known Limitations

1. **Rule-Based Routing**: Intent classification is deterministic and rule-based; novel vocabulary outside remote-sensing taxonomy defaults to general VQA.
2. **Disagreement Arbitration**: When specialists disagree, the controller reports the conflict transparently rather than applying an automated probabilistic arbiter, ensuring human experts make final decisions.
