# SatQuery AI — Step 13: Real Remote-Sensing Benchmark & Evaluation Framework

## 1. Overview & Core Principles

SatQuery AI's Step 13 introduces a reproducible, scientifically honest evaluation framework for remote sensing artificial intelligence. The framework provides a strict separation between **model capability** and **measured benchmark performance**.

### Non-Fabrication Guarantee
1. **Zero Fabricated Scores**: Historical placeholder numbers (e.g. 88.3%, 86.4%, 89.4%) have been completely eliminated from all production paths and evaluation reporting.
2. **Four Distinct Dataset States**:
   - `evaluated`: Dataset is locally present, formatted, and evaluated through genuine model inference.
   - `partially_evaluated`: Dataset is present, but certain samples were skipped or failed (with exact reasons recorded).
   - `dataset_unavailable`: Required dataset directory or split annotations are unconfigured or not found on disk. Zero metrics are reported.
   - `not_configured`: Benchmark ID is unrecognized.
3. **Auditability**: Every evaluation run records a deterministic seed, timestamp, run ID, model checkpoint, preprocessing version, decision threshold, per-sample predictions, ground-truth answers, and runtime latency.

---

## 2. Benchmark Registry & Supported Datasets

The central registry ([`backend/evaluation/registry.py`](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/backend/evaluation/registry.py)) defines the contracts for four primary remote sensing benchmarks:

| Benchmark ID | Benchmark Name | Task | Modality | Model Architecture | Adaptation Status | Environment Variable |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bigearthnet-s2` | BigEarthNet-S2 | Multilabel Classification | Sentinel-2 (10 bands) | ResNet-18 (`resnet18-s2-v0.2.0`) | `pretrained_specialist` | `BIGEARTHNET_ROOT` |
| `rsvqa-hr` | RSVQA-HR | Visual Question Answering | High-Resolution Aerial (RGB) | `microsoft/Florence-2-base` | `not_remote_sensing_finetuned` | `RSVQA_ROOT` |
| `rsvqa-lr` | RSVQA-LR | Visual Question Answering | Sentinel-2 (RGB composite) | `microsoft/Florence-2-base` | `not_remote_sensing_finetuned` | `RSVQA_ROOT` |
| `vrsbench` | VRSBench | Scene Captioning / Description | Optical Aerial / Satellite | `microsoft/Florence-2-base` | `not_remote_sensing_finetuned` | `VRSBENCH_ROOT` |
| `cdvqa` | CDVQA | Bi-Temporal Change VQA | Bi-Temporal Image Pairs | `BiTemporalChangeService` + Florence-2 | `algorithmic_differencing_with_vlm` | `CDVQA_ROOT` |

---

## 3. Dataset Configuration & Discovery

Dataset roots are configured exclusively via environment variables. **No hardcoded local machine paths are permitted.**

```bash
# Environment Variable Configuration
export BIGEARTHNET_ROOT="/path/to/BigEarthNet-S2"
export RSVQA_ROOT="/path/to/RSVQA"
export VRSBENCH_ROOT="/path/to/VRSBench"
export CDVQA_ROOT="/path/to/CDVQA"
```

### Dataset Structure Requirements

#### BigEarthNet-S2 (`$BIGEARTHNET_ROOT`)
- `splits/test.json` or `test.json`: Array of `{ "patch_name": "...", "labels": [...] }`
- Patch directory containing 10-band Sentinel-2 GeoTIFFs or `.npy` arrays with exact spectral band ordering:
  `B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12`.
- 19 CORINE Land Cover classes.

#### RSVQA (`$RSVQA_ROOT`)
- `test_questions.json` or `questions.json`: Array/object containing `{ "id": ..., "question": "...", "answer": "...", "image": "...", "category": "..." }`
- `images/`: Directory containing source optical imagery.

#### VRSBench (`$VRSBENCH_ROOT`)
- `captions.json` or `vrsbench_test_captions.json`: Contains `{ "id": ..., "caption": "...", "image": "...", "task": "captioning" }`
- Only scene description / captioning tasks are evaluated. Unsupported tasks (e.g. 3D reconstruction) are marked as `unsupported_task` and skipped.

#### CDVQA (`$CDVQA_ROOT`)
- `cdvqa.json` or `cdvqa_test.json`: Contains `{ "id": ..., "image_t1": "...", "image_t2": "...", "question": "...", "answer": "...", "change_type": "..." }`
- Pairs of bi-temporal images.

---

## 4. Evaluation Metrics & Mathematical Formulations

All evaluation metrics are implemented in [`backend/evaluation/metrics.py`](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/backend/evaluation/metrics.py) without hidden external dependencies:

### Multilabel Classification Metrics
- **Micro Precision**: $\frac{\sum \text{TP}}{\sum \text{TP} + \sum \text{FP}}$
- **Micro Recall**: $\frac{\sum \text{TP}}{\sum \text{TP} + \sum \text{FN}}$
- **Micro F1-Score**: $2 \cdot \frac{\text{MicroP} \cdot \text{MicroR}}{\text{MicroP} + \text{MicroR}}$
- **Macro F1-Score**: $\frac{1}{C} \sum_{c=1}^C \text{F1}_c$ (unweighted mean across all 19 CORINE classes)
- **Per-Class Metrics**: Precision, Recall, F1, and Support per class.
- **Exact Match Ratio**: Percentage of samples where predicted multi-label set exactly matches ground truth across all classes.
- **Hamming Loss**: $\frac{1}{N \cdot C} \sum_{i=1}^N \sum_{c=1}^C \mathbb{I}(y_{i,c} \neq \hat{y}_{i,c})$ (fraction of incorrect label predictions)

### VQA Metrics
- **Exact Match (EM)**: Strict string equality between prediction and reference answer after trimming whitespace.
- **Normalized Exact Match (N-EM)**: String equality after case-folding, removing punctuation, and stripping English articles (`a`, `an`, `the`).
- **Category-Stratified Accuracy**: Accuracy computed independently across question categories (presence, count, comparison, etc.).

### Captioning Metrics
- **ROUGE-L**: F-measure based on the Longest Common Subsequence (LCS) of word tokens between prediction and reference.
- **Token Overlap F1**: Harmonic mean of token precision and token recall.

---

## 5. Running Evaluations

### Running via Python API
```python
from backend.evaluation import EvaluationRunner, ReportGenerator

runner = EvaluationRunner()

# Evaluate BigEarthNet-S2 (if configured)
result = runner.run_evaluation(
    benchmark_id="bigearthnet-s2",
    split="test",
    threshold=0.5,
    seed=42,
    save_audit=True,
)

# Generate JSON and Markdown reports
json_path = ReportGenerator.generate_json_report(result)
md_path = ReportGenerator.generate_markdown_report(result)

print(f"Status: {result.status}")
print(f"Report: {md_path}")
```

### Running Unit Tests
```bash
python -m unittest backend/tests/test_evaluation_framework.py -v
```

---

## 6. Output Reports & Sample-Level Audit Trail

Each evaluation run creates two files:
1. `reports/<run_id>.json`: Complete machine-readable result, containing dataset metadata, metrics, metric definitions, and sample-level audit records.
2. `reports/<run_id>.md`: Human-readable summary table detailing dataset, task, model, metrics, and limitations.

### Sample-Level Audit Record Structure
```json
{
  "sample_id": "S2A_MSIL2A_20170717T113321_N0205_R080_T30UVU_24_57",
  "prediction": ["Arable land", "Complex cultivation patterns"],
  "ground_truth": ["Arable land", "Complex cultivation patterns"],
  "metrics": {
    "exact_match": 1
  },
  "status": "evaluated",
  "reason": "OK",
  "runtime_ms": 34.2
}
```

---

## 7. Current Evaluation State

- **Locally Available Datasets**: 0 (No remote sensing benchmark roots are currently mounted on this development machine).
- **Evaluation Framework**: Fully implemented and validated across 18 specialized unit tests.
- **Current Benchmark Execution Status**: `dataset_unavailable` pending configuration of `BIGEARTHNET_ROOT`, `RSVQA_ROOT`, `VRSBENCH_ROOT`, or `CDVQA_ROOT`.
- **Zero Fabricated Scores**: No benchmark accuracy or F1 score is reported.
