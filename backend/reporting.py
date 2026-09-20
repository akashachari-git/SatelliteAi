"""
SatQuery AI - Production Analysis Report Generator.
Generates comprehensive, non-fabricated intelligence dossiers from genuine AnalyzeResponse payloads.
"""
import time
from typing import Dict, Any, List, Optional


def generate_analysis_markdown(data: Dict[str, Any]) -> str:
    """
    Generates a structured, auditable Markdown intelligence report
    from a verified AnalyzeResponse payload.
    """
    query = data.get("query", "N/A")
    mode = data.get("mode", "single")
    task_type = data.get("taskType", "vqa")
    selected_model = data.get("selectedModel", "SatQuery AI")
    answer = data.get("answer", "No answer provided.")
    timestamp = data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    input_info = data.get("inputInformation", "Observation raster")
    evidence = data.get("evidence", [])
    bounding_boxes = data.get("boundingBoxes") or []
    execution_steps = data.get("executionSteps") or []
    imagery_meta = data.get("imageryMetadata") or {}
    evidence_hierarchy = data.get("evidenceHierarchy") or {}
    geospatial_evidence = data.get("geospatialEvidence") or {}
    cross_modal = data.get("crossModalEvidence") or {}
    agent_plan = data.get("agentPlan") or {}

    direct_evidence = evidence_hierarchy.get("direct_evidence") or evidence
    supporting_evidence = evidence_hierarchy.get("supporting_evidence") or []
    limitations = evidence_hierarchy.get("limitations") or geospatial_evidence.get("limitations") or []
    disagreements = evidence_hierarchy.get("disagreements") or []

    lines = [
        "# SatQuery AI — Mission Intelligence Dossier",
        "",
        f"**Timestamp**: `{timestamp}`  ",
        f"**Analysis Mode**: `{mode.upper()}`  ",
        f"**Classified Task**: `{task_type}`  ",
        f"**Executing Specialist**: `{selected_model}`  ",
        "",
        "---",
        "",
        "## 1. Input Observation Information",
        "",
        f"- **Input Identification**: {input_info}",
        f"- **Dimensions**: {imagery_meta.get('dimensions', 'Unknown')}",
        f"- **Spatial Resolution / GSD**: {imagery_meta.get('resolution', 'Unknown')}",
        f"- **Sensor / Modality**: {imagery_meta.get('modality', 'Optical')} ({imagery_meta.get('sensor', 'Remote Sensing')})",
        f"- **Coordinate Reference System (CRS)**: {imagery_meta.get('crs') or geospatial_evidence.get('crs') or 'Local Pixel Space (Unprojected)'}",
        "",
        "---",
        "",
        "## 2. Natural Language Query & Grounded Answer",
        "",
        f"**User Question**: *\"{query}\"*",
        "",
        "### Synthesized Intelligence Answer",
        f"> {answer}",
        "",
        "---",
        "",
        "## 3. Evidence Provenance & Hierarchy",
        "",
    ]

    if direct_evidence:
        lines.append("### Primary Direct Evidence")
        for ev in direct_evidence:
            lines.append(f"- {ev}")
        lines.append("")

    if supporting_evidence:
        lines.append("### Secondary Supporting Evidence")
        for ev in supporting_evidence:
            lines.append(f"- {ev}")
        lines.append("")

    if disagreements:
        lines.extend([
            "### Specialist Disagreement Notice",
            "> [!WARNING]",
            "> The following divergence was identified between specialist models. Both perspectives are preserved:",
        ])
        for dis in disagreements:
            lines.append(f"- {dis}")
        lines.append("")

    # Geospatial Evidence Section
    lines.extend([
        "---",
        "",
        "## 4. Geospatial Grounding & Coordinate Evidence",
        "",
    ])
    geo_status = geospatial_evidence.get("status", "unavailable")
    if geo_status == "available":
        lines.extend([
            f"- **Geospatial Status**: `VERIFIED ({geospatial_evidence.get('crs', 'WGS84')})`",
            f"- **Coordinate Reference System**: `{geospatial_evidence.get('crs')}`",
        ])
        if geospatial_evidence.get("geographicBounds"):
            b = geospatial_evidence["geographicBounds"]
            lines.append(
                f"- **Geographic Extents (WGS84)**: North {b.get('north', 'N/A')}°, South {b.get('south', 'N/A')}°, East {b.get('east', 'N/A')}°, West {b.get('west', 'N/A')}°"
            )
        if geospatial_evidence.get("geographicCoordinates"):
            c = geospatial_evidence["geographicCoordinates"]
            lines.append(f"- **Scene Center**: Lat {c.get('latitude', 'N/A')}°, Lon {c.get('longitude', 'N/A')}°")
        if geospatial_evidence.get("area"):
            a = geospatial_evidence["area"]
            lines.append(f"- **Ground Surface Area**: {a.get('area_km2', 'N/A')} km² ({a.get('area_m2', 'N/A')} m²)")
        if geospatial_evidence.get("alignmentStatus"):
            lines.append(f"- **Spatial Alignment**: `{geospatial_evidence.get('alignmentStatus')}`")
    else:
        lines.extend([
            "- **Geospatial Status**: `UNAVAILABLE`",
            "> [!NOTE]",
            "> Geospatial coordinates unavailable because the source raster does not contain sufficient georeferencing metadata (CRS/Affine Geotransform). Spatial positions remain grounded in relative pixel space.",
        ])
    lines.append("")

    # Spatial Bounding Boxes
    if bounding_boxes:
        lines.extend([
            "---",
            "",
            "## 5. Localized Spatial Regions",
            "",
            "| ID | Label | Pixel Bounds (X, Y, W, H %) | Geographic Coordinates (WGS84) | Description |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for b in bounding_boxes:
            b_id = b.get("id", "box")
            b_lbl = b.get("label", "Feature")
            b_px = f"[{b.get('x', 0):.1f}%, {b.get('y', 0):.1f}%, {b.get('width', 0):.1f}%, {b.get('height', 0):.1f}%]"
            geo_info = "N/A (Pixel Space)"
            if b.get("centerLatLon"):
                cll = b["centerLatLon"]
                geo_info = f"{cll.get('latitude', 0):.5f}° N, {cll.get('longitude', 0):.5f}° E"
            elif b.get("geographicBbox"):
                g = b["geographicBbox"]
                geo_info = f"[{g.get('min_lon', 0):.4f}, {g.get('min_lat', 0):.4f}] to [{g.get('max_lon', 0):.4f}, {g.get('max_lat', 0):.4f}]"
            b_desc = b.get("description", "").replace("|", "/")
            lines.append(f"| `{b_id}` | **{b_lbl}** | `{b_px}` | {geo_info} | {b_desc} |")
        lines.append("")

    # Cross-Modal Optical + SAR Findings (if applicable)
    if cross_modal:
        lines.extend([
            "---",
            "",
            "## 6. Cross-Modal Optical + SAR Findings",
            "",
        ])
        if cross_modal.get("opticalEvidence"):
            lines.append("### Optical Spectral Evidence")
            for o_ev in cross_modal["opticalEvidence"]:
                lines.append(f"- {o_ev}")
            lines.append("")
        if cross_modal.get("sarEvidence"):
            lines.append("### SAR Radar Backscatter Evidence")
            for s_ev in cross_modal["sarEvidence"]:
                lines.append(f"- {s_ev}")
            lines.append("")
        if cross_modal.get("crossModalCorroboration"):
            lines.append("### Cross-Sensor Corroboration")
            for c_ev in cross_modal["crossModalCorroboration"]:
                lines.append(f"- {c_ev}")
            lines.append("")

    # Limitations & Calibration
    lines.extend([
        "---",
        "",
        "## 7. Limitations & Uncertainty Disclaimers",
        "",
    ])
    if limitations:
        for lim in limitations:
            lines.append(f"- {lim}")
    else:
        lines.append("- Analysis executed within standard specialist operational envelopes.")
    lines.append("")

    # Execution Trace
    if execution_steps:
        lines.extend([
            "---",
            "",
            "## 8. Agentic Execution Trace",
            "",
            "| Step # | Stage Title | Status | Duration | Summary |",
            "| :---: | :--- | :---: | :---: | :--- |",
        ])
        for step in execution_steps:
            s_num = step.get("stepNumber", 0)
            s_title = step.get("title", "Stage")
            s_status = step.get("status", "completed").upper()
            s_dur = f"{step.get('durationMs', 0)} ms"
            s_sum = step.get("summary", "").replace("|", "/")
            lines.append(f"| {s_num} | **{s_title}** | `{s_status}` | {s_dur} | {s_sum} |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "> *Report generated autonomously by SatQuery AI — Reproducible Evidence-Grounded Remote Sensing Architecture.*",
    ])

    return "\n".join(lines) + "\n"
