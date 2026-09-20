import { AnalysisResult } from '../types';
import { generateReport } from '../services/api';

/**
 * Generates and downloads a clean, professional intelligence report for SatQuery AI.
 * Uses real backend report generation when available, with client-side fallback.
 * Strictly non-fabricated: zero hardcoded coordinates, zero fake metrics.
 */
export async function downloadAnalysisReport(
  result: AnalysisResult,
  format: 'text' | 'json' | 'geojson' = 'text'
): Promise<void> {
  const timestamp = result.timestamp || new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  const cleanDate = timestamp.replace(/[^a-zA-Z0-9]/g, '_').slice(0, 19);

  const isBiTemporal = result.mode === 'bi-temporal';
  const isOpticalSar = result.mode === 'optical-sar';

  const analysisTypeName = isBiTemporal
    ? 'Past & Present Change Detection'
    : isOpticalSar
    ? 'Optical + SAR Cross-Sensor Analysis'
    : 'Single Image Analysis';

  const imageFiles = result.inputInformation || 'Satellite Observation Raster';

  // 1. GeoJSON format
  if (format === 'geojson') {
    const isGeoAvailable = result.geospatialEvidence?.status === 'available';
    const crsName = isGeoAvailable
      ? result.geospatialEvidence?.crs || 'urn:ogc:def:crs:OGC:1.3:CRS84'
      : 'Local Pixel Coordinates (Unprojected)';

    const features: any[] = [];
    if (result.boundingBoxes && result.boundingBoxes.length > 0) {
      result.boundingBoxes.forEach((b) => {
        let coords: [number, number][][] = [];
        if (isGeoAvailable) {
          if (b.geographicBbox) {
            const g = b.geographicBbox;
            coords = [
              [
                [g.min_lon, g.max_lat],
                [g.max_lon, g.max_lat],
                [g.max_lon, g.min_lat],
                [g.min_lon, g.min_lat],
                [g.min_lon, g.max_lat],
              ],
            ];
          } else if (result.geospatialEvidence?.bounds) {
            const bounds = result.geospatialEvidence.bounds;
            const west = bounds.west + (b.x / 100) * (bounds.east - bounds.west);
            const east = bounds.west + ((b.x + b.width) / 100) * (bounds.east - bounds.west);
            const north = bounds.north - (b.y / 100) * (bounds.north - bounds.south);
            const south = bounds.north - ((b.y + b.height) / 100) * (bounds.north - bounds.south);
            coords = [
              [
                [west, north],
                [east, north],
                [east, south],
                [west, south],
                [west, north],
              ],
            ];
          }
        } else {
          coords = [
            [
              [b.x, b.y],
              [b.x + b.width, b.y],
              [b.x + b.width, b.y + b.height],
              [b.x, b.y + b.height],
              [b.x, b.y],
            ],
          ];
        }

        features.push({
          type: 'Feature',
          id: b.id,
          geometry: coords.length > 0 ? { type: 'Polygon', coordinates: coords } : null,
          properties: {
            label: b.label,
            description: b.description,
            pixelBounds: { x: b.x, y: b.y, width: b.width, height: b.height },
            query: result.query,
          },
        });
      });
    }

    const geoJsonDoc = {
      type: 'FeatureCollection',
      crs: {
        type: 'name',
        properties: { name: crsName },
      },
      metadata: {
        title: 'SatQuery AI Spatial Evidence Report',
        analysisType: analysisTypeName,
        query: result.query,
        answer: result.answer,
        analysisDate: timestamp,
        imagery: imageFiles,
        geospatialStatus: result.geospatialEvidence?.status || 'unavailable',
      },
      features,
    };

    const blob = new Blob([JSON.stringify(geoJsonDoc, null, 2)], { type: 'application/geo+json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `SATQUERY_SPATIAL_EVIDENCE_${cleanDate}.geojson`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    return;
  }

  // 2. Structured Clean JSON
  if (format === 'json') {
    const jsonDoc = {
      reportTitle: 'SatQuery AI - Satellite Imagery Analysis Report',
      analysisDate: timestamp,
      analysisType: analysisTypeName,
      selectedModel: result.selectedModel,
      userQuestion: result.query,
      imagesUsed: {
        fileNames: imageFiles,
        dimensions: result.imageryMetadata?.dimensions || 'Unspecified dimensions',
        spatialResolution: result.imageryMetadata?.resolution || 'Unspecified GSD',
        coordinateSystem: result.geospatialEvidence?.crs || result.imageryMetadata?.crs || 'Local Pixel Space',
      },
      mainFindings: {
        answer: result.answer,
        directEvidence: result.evidenceHierarchy?.direct_evidence || result.evidence || [],
        supportingEvidence: result.evidenceHierarchy?.supporting_evidence || [],
        limitations: result.evidenceHierarchy?.limitations || result.geospatialEvidence?.limitations || [],
        disagreements: result.evidenceHierarchy?.disagreements || [],
      },
      geospatialEvidence: result.geospatialEvidence || { status: 'unavailable' },
      crossModalEvidence: result.crossModalEvidence || null,
      visualEvidence: {
        detectedFeaturesCount: result.boundingBoxes?.length || 0,
        items: result.boundingBoxes || [],
      },
      executionTrace: result.executionSteps || [],
    };

    const blob = new Blob([JSON.stringify(jsonDoc, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `SATQUERY_REPORT_${cleanDate}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    return;
  }

  // 3. Text / Markdown Report: Try backend report generator first
  try {
    const backendReport = await generateReport(result);
    if (backendReport?.markdown) {
      const blob = new Blob([backendReport.markdown], { type: 'text/markdown;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `SATQUERY_REPORT_${cleanDate}.md`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      return;
    }
  } catch {
    // Fall back to client-side markdown generation
  }

  // Client-side structured report
  const geoStatus = result.geospatialEvidence?.status === 'available'
    ? `VERIFIED (${result.geospatialEvidence.crs})`
    : 'UNAVAILABLE (Pixel Space)';

  const lines = [
    '================================================================================',
    '                    SATQUERY AI - MISSION INTELLIGENCE REPORT',
    '               "Understand your satellite imagery with AI"',
    '================================================================================',
    `ANALYSIS DATE: ${timestamp}`,
    `ANALYSIS TYPE: ${analysisTypeName}`,
    `SPECIALIST:    ${result.selectedModel || 'SatQuery AI Multi-Specialist'}`,
    `GEOSPATIAL:    ${geoStatus}`,
    '',
    '================================================================================',
    '1. INPUT OBSERVATIONS',
    '================================================================================',
    `Files:              ${imageFiles}`,
    `Dimensions:         ${result.imageryMetadata?.dimensions || 'Unspecified'}`,
    `Spatial Resolution: ${result.imageryMetadata?.resolution || 'Unspecified GSD'}`,
    `Sensor / Modality:  ${result.imageryMetadata?.modality || 'Optical'} (${result.imageryMetadata?.sensor || 'Remote Sensing'})`,
    `CRS:                ${result.geospatialEvidence?.crs || result.imageryMetadata?.crs || 'Local Pixel Space (Unprojected)'}`,
    '',
    '================================================================================',
    "2. USER'S QUESTION",
    '================================================================================',
    `"${result.query}"`,
    '',
    '================================================================================',
    '3. GROUNDED ANSWER',
    '================================================================================',
    result.answer,
    '',
    'Primary Evidence:',
  ];

  const direct = result.evidenceHierarchy?.direct_evidence || result.evidence || [];
  if (direct.length > 0) {
    direct.forEach((e, idx) => lines.push(`  [${idx + 1}] ${e}`));
  } else {
    lines.push('  • Evidence synthesized directly in answer.');
  }

  const supp = result.evidenceHierarchy?.supporting_evidence || [];
  if (supp.length > 0) {
    lines.push('', 'Supporting Evidence:');
    supp.forEach((s) => lines.push(`  • ${s}`));
  }

  const dis = result.evidenceHierarchy?.disagreements || [];
  if (dis.length > 0) {
    lines.push('', 'Specialist Disagreements:');
    dis.forEach((d) => lines.push(`  ⚠ ${d}`));
  }

  const lims = result.evidenceHierarchy?.limitations || result.geospatialEvidence?.limitations || [];
  if (lims.length > 0) {
    lines.push('', 'Limitations & Uncertainty:');
    lims.forEach((l) => lines.push(`  - ${l}`));
  }

  if (result.boundingBoxes && result.boundingBoxes.length > 0) {
    lines.push(
      '',
      '================================================================================',
      '4. SPATIAL EVIDENCE REGIONS',
      '================================================================================'
    );
    result.boundingBoxes.forEach((b) => {
      lines.push(`  • ${b.label}: [X:${b.x.toFixed(1)}%, Y:${b.y.toFixed(1)}%, W:${b.width.toFixed(1)}%, H:${b.height.toFixed(1)}%] - ${b.description}`);
    });
  }

  lines.push(
    '',
    '================================================================================',
    'END OF SATQUERY AI REPORT',
    '================================================================================'
  );

  const textContent = lines.join('\n') + '\n';
  const blob = new Blob([textContent], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `SATQUERY_REPORT_${cleanDate}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
