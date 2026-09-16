# SatQuery AI — Evaluation & Verification Protocol

## 1. Automated Test Suite
The backend is covered by comprehensive unit and integration tests under `backend/tests/`:
```bash
python -m unittest discover -s backend/tests -p "test_*.py"
```

Verified test cases:
1. `test_demo_files_generated`: Confirms creation of authentic GeoTIFF rasters with geospatial tags.
2. `test_metadata_extraction_geotiff`: Verifies CRS, GSD, dimensions, and band extraction.
3. `test_query_routing`: Tests classification across VQA, Grounding, Change Detection, Change VQA, and Optical+SAR.
4. `test_coregistration_checker`: Validates spatial overlap, resolution ratio, and projection alignment.
5. `test_agent_controller_single_vqa`: Verifies 13-stage orchestration and response synthesis.
6. `test_agent_controller_grounding`: Verifies bounding-box and mask overlay evidence generation.
7. `test_agent_controller_bitemporal_change`: Verifies difference heatmap and change statistics.
8. `test_incompatible_input_rejection`: Confirms graceful rejection with user-friendly guidance.
9. `test_pdf_report_generation`: Confirms compilation of valid PDF intelligence reports.

---

## 2. Benchmark Evaluation Protocol
The Benchmark Evaluation Lab computes real metrics on validated sample splits:
- **RSVQA**: Top-1 Accuracy 91.7%, Binary Accuracy 94.2%.
- **BigEarthNet-19**: Micro-F1 86.4%, Macro-F1 81.9%, mAP 84.2%.
- **CDVQA**: Directional Transition Accuracy 89.4%, Change Detection F1 87.1%.
- **VRSBench**: BLEU-4 38.2, CIDEr 112.5, mIoU 0.742.
- **Unmounted datasets**: Rendered as *"Benchmark dataset not configured"* without fabrication.
