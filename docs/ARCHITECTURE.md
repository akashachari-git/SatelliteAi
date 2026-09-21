# SatQuery AI — System Architecture & Autonomous Orchestration

## 1. Architectural Philosophy
SatQuery AI bridges the divide between non-expert natural language queries and complex multimodal Earth Observation (EO) analytics. Rather than functioning as a black-box VLM wrapper, SatQuery AI implements an autonomous 13-stage orchestration pipeline backed by a modular Specialist Tool Registry and domain-adapted remote-sensing models.

```
                  ┌─────────────────────────────────────┐
                  │    User Natural-Language Query      │
                  │   + Satellite Rasters (GeoTIFF)     │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Stage 1: Linguistic Normalization & Query Inspection     │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Stage 2: Geospatial Metadata & Radiometric Extraction   │
        │          - GeoTIFF Tags (ModelTiepoint, PixelScale)     │
        │          - CRS Detection (UTM/WGS84) & GSD Resolution   │
        │          - Modality Classifier (Optical, MS, SAR)       │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Stage 3: Deterministic & Rule-Assisted Intent Routing   │
        │          - SINGLE_VQA, CAPTIONING, GROUNDING,           │
        │            CHANGE_ANALYSIS, CHANGE_VQA, OPTICAL_SAR     │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Stage 4: Input Precondition & Co-Registration Check     │
        │          - Spatial Footprint & Overlap Verification     │
        │          - Pixel Dimension & Resolution Ratio Alignment │
        │          - Incompatible Query Rejection & Guidance      │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Stage 5: Specialist Model / Tool Selection              │
        │          - Tool Registry Query                          │
        │          - Parameter Configuration                      │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Stage 6: Specialist Model Inference Execution           │
        │          - BigEarthNet 19-Class Classifier              │
        │          - Multi-Spectral Indices (NDVI, NDWI, NDBI)    │
        │          - Bi-Temporal Difference & Otsu Segmentation   │
        │          - SAR Backscatter & Dielectric Fusion          │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Stage 7: Evidence Extraction & Verification             │
        │          - Segmentation Masks & Bounding Boxes          │
        │          - Difference Heatmaps & False-Color Composites │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Stage 8: Calibrated Multi-Factor Confidence Estimation  │
        │          - Model Agreement + Evidence + Compatibility   │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Stage 9: Response Synthesis & Intelligence Dossier      │
        │          - Human-Readable Grounded Answer               │
        │          - Auditable Observable Execution Trace         │
        │          - Downloadable PDF & JSON Report               │
        └─────────────────────────────────────────────────────────┘
```

## 2. Component Directory Structure
- `/backend`: FastAPI microservices, agents, models, tools, and report generator.
  - `/app/agents`: `agent_controller.py` (13-stage orchestrator) and `query_classifier.py`.
  - `/app/models`: `base.py` (`BaseRemoteSensingModel` abstract contract) and `adaptation.py` (BigEarthNet-19 taxonomy).
  - `/app/tools`: Specialist services (`vqa_service.py`, `captioning_service.py`, `grounding_service.py`, `change_detection_service.py`, `change_vqa_service.py`, `optical_sar_service.py`, `tool_registry.py`).
  - `/app/remote_sensing`: `metadata_extractor.py`, `co_registration.py`, `image_processor.py`, `spectral_indices.py`.
  - `/app/reports`: `report_service.py` (ReportLab PDF compilation with embedded evidence).
  - `/app/evaluation`: `benchmark_service.py` (RSVQA, BigEarthNet, CDVQA, VRSBench).
  - `/app/demo`: `demo_generator.py` (Authentic GeoTIFF generator for instant evaluation).
- `/frontend`: React 18, TypeScript, Tailwind CSS, Lucide icons.
  - `/src/components`: `Navbar`, `ImageViewer`, `SwipeViewer`, `DualModalityViewer`, `TraceTimeline`, `MetadataCard`, `ConfidenceGauge`, `ReportModal`.
  - `/src/pages`: `LandingPage`, `WorkspacePage`, `ChangeStudioPage`, `OpticalSarPage`, `AgentTraceView`, `BenchmarkLabPage`, `DocsPage`.
