import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak, HRFlowable
)
from ..config import REPORTS_DIR, BASE_DIR

class ReportService:
    """
    Generates formal PDF and structured JSON intelligence reports
    for remote-sensing vision-language analyses.
    """

    @classmethod
    def generate_pdf_report(cls, analysis_data: Dict[str, Any]) -> str:
        rand_seed = os.urandom(3).hex()
        report_id = f"satquery_report_{int(time.time())}_{rand_seed}"
        pdf_path = REPORTS_DIR / f"{report_id}.pdf"

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        # Custom Brand Styles
        brand_title = ParagraphStyle(
            'BrandTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a") # Deep Slate
        )
        subtitle_style = ParagraphStyle(
            'BrandSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#0284c7") # Cyan Accent
        )
        heading2 = ParagraphStyle(
            'SectionHead',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=12,
            spaceAfter=6
        )
        body_text = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#334155")
        )
        highlight_box = ParagraphStyle(
            'Highlight',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#0f172a")
        )

        story = []

        # Header Block
        story.append(Paragraph("SATQUERY AI // INTELLIGENCE DOSSIER", brand_title))
        story.append(Paragraph("Multimodal Remote Sensing Vision-Language Assistant | Geospatial Intelligence Framework", subtitle_style))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284c7"), spaceBefore=2, spaceAfter=12))

        # Query & Summary Meta
        meta_table_data = [
            [Paragraph("<b>Natural Language Query:</b>", body_text), Paragraph(f"<i>\"{analysis_data.get('query', 'N/A')}\"</i>", highlight_box)],
            [Paragraph("<b>Identified Task:</b>", body_text), Paragraph(f"<b>{analysis_data.get('task', 'N/A')}</b>", body_text)],
            [Paragraph("<b>Selected Specialist Tool:</b>", body_text), Paragraph(str(analysis_data.get("selected_tool", "N/A")), body_text)],
            [Paragraph("<b>Model Backend:</b>", body_text), Paragraph(str(analysis_data.get("model_name", "N/A")), body_text)],
            [Paragraph("<b>Confidence Score:</b>", body_text), Paragraph(f"<b>{analysis_data.get('confidence', {}).get('percentage', 0)}%</b> ({analysis_data.get('confidence', {}).get('label', '')})", body_text)],
            [Paragraph("<b>Timestamp:</b>", body_text), Paragraph(str(analysis_data.get("created_at", "N/A")), body_text)],
            [Paragraph("<b>Total Pipeline Latency:</b>", body_text), Paragraph(f"{analysis_data.get('total_duration_seconds', 0)} seconds", body_text)]
        ]

        t_meta = Table(meta_table_data, colWidths=[150, 390])
        t_meta.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4)
        ]))
        story.append(t_meta)
        story.append(Spacer(1, 12))

        # Executive Answer
        story.append(Paragraph("EXECUTIVE ANSWER & DERIVED INSIGHTS", heading2))
        clean_answer = analysis_data.get("answer", "").replace("**", "").replace("`", "")
        p_ans = Paragraph(clean_answer, body_text)
        ans_table = Table([[p_ans]], colWidths=[540])
        ans_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0fdf4")), # Soft green
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#86efac")),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
        ]))
        story.append(ans_table)
        story.append(Spacer(1, 12))

        raw_res = analysis_data.get("raw_result") or {}

        # Bi-Temporal Specific Intelligence Metrics
        if (
            analysis_data.get("task") in ["CHANGE_ANALYSIS", "CHANGE_DETECTION", "CHANGE_VQA"]
            or "change_pct" in raw_res
            or "changed_area_km2" in raw_res
        ):
            story.append(Paragraph("BI-TEMPORAL LANDSCAPE FLUX & CHANGE METRICS", heading2))
            change_pct = raw_res.get("change_percentage", raw_res.get("change_pct", 0))
            area_km2 = raw_res.get("changed_area_km2", 0)
            dom_sector = raw_res.get("dominant_sector", "N/A")
            trans_type = raw_res.get("transition_type", "Surface Reflectance Variance")
            quad_dist = raw_res.get("quadrant_distribution", {})
            quad_str = " | ".join([f"{k}: {v}%" for k, v in quad_dist.items()]) if quad_dist else "Balanced"
            deltas = raw_res.get("spectral_deltas", {})
            deltas_str = f"NDVI delta: {deltas.get('ndvi_delta', 0):+.3f}, NDBI delta: {deltas.get('ndbi_delta', 0):+.3f}" if deltas else "Standard CVA Metric"

            change_rows = [
                [Paragraph("<b>Changed Area Footprint:</b>", body_text), Paragraph(f"<b>{change_pct}%</b> ({area_km2} km²)", highlight_box)],
                [Paragraph("<b>Dominant Spatial Sector:</b>", body_text), Paragraph(f"<b>{dom_sector}</b> Quadrant", body_text)],
                [Paragraph("<b>Primary Land-Cover Transition:</b>", body_text), Paragraph(str(trans_type), body_text)],
                [Paragraph("<b>Spectral Index Deltas:</b>", body_text), Paragraph(deltas_str, body_text)],
                [Paragraph("<b>Quadrant Distribution:</b>", body_text), Paragraph(quad_str, body_text)],
            ]
            t_change = Table(change_rows, colWidths=[160, 380])
            t_change.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#fcd34d")),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#fde68a")),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4)
            ]))
            story.append(t_change)
            story.append(Spacer(1, 12))

        # Optical + SAR Specific Fusion Metrics
        if (
            analysis_data.get("task") == "OPTICAL_SAR_ANALYSIS"
            or "sar_metrics" in raw_res
            or "optical_metrics" in raw_res
            or "confirmed_urban_pct" in raw_res
        ):
            story.append(Paragraph("CROSS-MODAL OPTICAL + SAR FUSION METRICS", heading2))
            urban_pct = raw_res.get("confirmed_urban_pct", 0)
            water_pct = raw_res.get("confirmed_water_pct", 0)
            sar_m = raw_res.get("sar_metrics", {})
            opt_m = raw_res.get("optical_metrics", {})
            insights = raw_res.get("complementary_insights", [])

            fusion_rows = [
                [Paragraph("<b>Double-Bounce Urban Core:</b>", body_text), Paragraph(f"<b>{urban_pct}%</b> (Confirmed via high SAR VV backscatter & NDBI)", highlight_box)],
                [Paragraph("<b>Specular Water / Smooth Surfaces:</b>", body_text), Paragraph(f"<b>{water_pct}%</b> (Confirmed via low SAR backscatter & NDWI)", body_text)],
                [Paragraph("<b>Sentinel-1 C-SAR Backscatter:</b>", body_text), Paragraph(f"Mean VV: {sar_m.get('mean_vv_db', -12.5)} dB | Mean VH: {sar_m.get('mean_vh_db', -19.2)} dB | Dynamic Range: {sar_m.get('sar_dynamic_range_db', 18.0)} dB", body_text)],
                [Paragraph("<b>Sentinel-2 Optical Metrics:</b>", body_text), Paragraph(f"Mean NDVI: {opt_m.get('mean_ndvi', 0.35)} | Mean NDWI: {opt_m.get('mean_ndwi', -0.15)} | Veg Cover: {opt_m.get('vegetation_cover_pct', 0)}%", body_text)]
            ]
            if insights:
                insight_paras = [Paragraph(f"• {ins}", body_text) for ins in insights]
                fusion_rows.append([Paragraph("<b>Cross-Modal Synthesis:</b>", body_text), insight_paras])

            t_fusion = Table(fusion_rows, colWidths=[160, 380])
            t_fusion.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#faf5ff")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#d8b4fe")),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e9d5ff")),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4)
            ]))
            story.append(t_fusion)
            story.append(Spacer(1, 12))

        # Input Images & Metadata
        story.append(Paragraph("INPUT SATELLITE IMAGERY METADATA", heading2))
        img_metas = analysis_data.get("images_metadata", [])
        img_rows = [["Filename", "Modality", "Dimensions", "Bands", "CRS", "Resolution (GSD)"]]
        for m in img_metas:
            img_rows.append([
                Paragraph(m.get("filename", "raster"), body_text),
                Paragraph(m.get("modality", "Optical"), body_text),
                Paragraph(f"{m.get('width', 0)}x{m.get('height', 0)}", body_text),
                Paragraph(str(m.get("bands", 1)), body_text),
                Paragraph(m.get("crs", "EPSG:32643")[:22], body_text),
                Paragraph(f"{m.get('spatial_resolution_m', 10.0)} m", body_text)
            ])

        t_imgs = Table(img_rows, colWidths=[130, 75, 75, 50, 130, 80])
        t_imgs.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        story.append(t_imgs)
        story.append(Spacer(1, 12))

        # Visual Evidence
        evidence = analysis_data.get("evidence", [])
        if evidence:
            story.append(Paragraph("VISUAL EVIDENCE & GROUNDING OVERLAYS", heading2))
            img_cells = []
            for ev in evidence[:3]: # Embed up to 3 evidence images
                url = ev.get("url", "")
                if url.startswith("/storage/"):
                    disk_path = BASE_DIR / url.lstrip("/")
                    if disk_path.exists():
                        try:
                            label = Paragraph(f"<b>Evidence Type:</b> {ev.get('type', 'Layer').replace('_', ' ').title()}", body_text)
                            img_widget = RLImage(str(disk_path), width=240, height=240)
                            img_cells.append([label, Spacer(1, 4), img_widget, Spacer(1, 8)])
                        except Exception:
                            pass
            if img_cells:
                for cell in img_cells:
                    story.append(KeepTogether(cell))
                story.append(Spacer(1, 10))

        # Auditable Trace
        story.append(Paragraph("AUDITABLE AGENT EXECUTION TRACE", heading2))
        trace_data = [["Step", "Stage Name", "Observed Outcome", "Status", "Duration"]]
        for idx, tr in enumerate(analysis_data.get("execution_trace", [])):
            trace_data.append([
                Paragraph(str(idx + 1), body_text),
                Paragraph(tr.get("stage_name", ""), body_text),
                Paragraph(tr.get("description", "")[:60], body_text),
                Paragraph(tr.get("status", "COMPLETED"), body_text),
                Paragraph(f"{tr.get('duration_ms', 0)} ms", body_text)
            ])

        t_trace = Table(trace_data, colWidths=[30, 130, 240, 70, 70])
        t_trace.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0"))
        ]))
        story.append(t_trace)
        story.append(Spacer(1, 14))

        # Governance & Limitations
        story.append(Paragraph("LIMITATIONS & SATELLITE GOVERNANCE NOTICE", heading2))
        notes = (
            "Confidence levels reflect radiometric data availability, spatial resolution bounds, and BigEarthNet model "
            "calibration. Atmospheric interference, dense cloud shadow, and sub-pixel mixed surfaces can impact "
            "classification accuracy. SatQuery AI outputs are intended for decision support and research verification."
        )
        story.append(Paragraph(notes, ParagraphStyle('Notice', parent=styles['Italic'], fontSize=8, leading=11, textColor=colors.HexColor("#64748b"))))

        doc.build(story)
        return f"/storage/reports/{pdf_path.name}"

    @classmethod
    def generate_json_export(cls, analysis_data: Dict[str, Any]) -> str:
        report_id = f"satquery_export_{int(time.time())}_{os.urandom(3).hex()}"
        json_path = REPORTS_DIR / f"{report_id}.json"
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2)

        return f"/storage/reports/{json_path.name}"
