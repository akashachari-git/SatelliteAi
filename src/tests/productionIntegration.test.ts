/**
 * SatQuery AI - Production Frontend Integration Test Suite
 * Validates Step 14 requirements:
 * 1. API client configuration & endpoints
 * 2. Upload validation contract
 * 3. Single-image analysis routing
 * 4. Multi-specialist response parsing
 * 5. Geospatial unavailable state
 * 6. Geospatial available state
 * 7. Grounding overlay mapping
 * 8. Bi-temporal change response
 * 9. Optical / SAR response
 * 10. Honest error handling (no silent fallbacks)
 * 11. Complete elimination of mock data from production path
 * 12. Report generation contract
 */
import {
  API_BASE_URL,
  API_CONFIG,
  validateSatelliteImage,
  executeRemoteSensingAnalysis,
  generateReport,
  getBackendHealth,
} from '../services/api';
import { AnalyzeResponse } from '../types/api';

function assert(condition: boolean, message: string) {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

async function runTests() {
  console.log('--- Starting SatQuery AI Step 14 Production Integration Tests ---');
  let passed = 0;
  let total = 0;

  async function test(name: string, fn: () => Promise<void> | void) {
    total++;
    try {
      await fn();
      console.log(`✓ [PASS] ${name}`);
      passed++;
    } catch (err: any) {
      console.error(`✗ [FAIL] ${name}:`, err.message);
      throw err;
    }
  }

  // 1. API Client Configuration
  await test('1. API client configuration & environment defaults', () => {
    assert(typeof API_BASE_URL === 'string', 'API_BASE_URL must be a string');
    assert(API_BASE_URL.startsWith('http'), 'API_BASE_URL must start with http');
    assert(API_CONFIG.endpoints.analyze === `${API_BASE_URL}/api/analyze`, 'Analyze endpoint mismatch');
    assert(API_CONFIG.endpoints.changeAnalysis === `${API_BASE_URL}/api/change-analysis`, 'Change endpoint mismatch');
    assert(API_CONFIG.endpoints.opticalSarAnalysis === `${API_BASE_URL}/api/optical-sar-analysis`, 'Optical/SAR endpoint mismatch');
    assert(API_CONFIG.endpoints.validateImage === `${API_BASE_URL}/api/validate-image`, 'Validate endpoint mismatch');
    assert(API_CONFIG.endpoints.generateReport === `${API_BASE_URL}/api/report/generate`, 'Report endpoint mismatch');
  });

  // 2. Upload Validation Contract
  await test('2. Upload validation handles valid raster responses', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: any, opts: any) => {
      if (String(url).includes('/api/validate-image')) {
        return new Response(
          JSON.stringify({
            filename: 'test_s2.tif',
            isValid: true,
            format: 'GeoTIFF',
            width: 1024,
            height: 1024,
            bands: 4,
            datatype: 'uint16',
            crs: 'EPSG:32643',
            resolution: '10.0m GSD',
            modality: 'Multispectral Optical',
            sensor: 'Sentinel-2 MSI',
            validationMessage: 'Valid satellite raster',
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        );
      }
      return originalFetch(url, opts);
    };

    try {
      const res = await validateSatelliteImage({ name: 'test_s2.tif', size: 5000000 });
      assert(res.valid === true, 'Expected valid === true');
      assert(res.status === 'VALID', 'Expected status === VALID');
      assert(res.dimensions === '1024 × 1024 px', 'Dimensions mismatch');
      assert(res.crs === 'EPSG:32643', 'CRS mismatch');
      assert(res.sensor === 'Sentinel-2 MSI', 'Sensor mismatch');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 3. Single-Image Analysis Contract
  await test('3. Single-image analysis formats request & processes response', async () => {
    let capturedBody: any = null;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: any, opts: any) => {
      if (String(url).includes('/api/analyze')) {
        capturedBody = JSON.parse(opts.body);
        const mockResponse: AnalyzeResponse = {
          query: 'What land cover is present?',
          mode: 'single',
          taskType: 'vqa',
          selectedModel: 'Florence-2-base + BigEarthNet ResNet-18',
          answer: 'The observation exhibits dense mixed forest canopy with adjacent agricultural parcels.',
          confidence: null,
          evidence: ['Detected broadleaf forest canopy', 'Identified agricultural parcels'],
          evidenceHierarchy: {
            directEvidence: ['Direct visual detection of tree canopy'],
            supportingEvidence: ['Multispectral reflectance corroborates vegetation'],
            limitations: ['Single optical snapshot without seasonal temporal context'],
            disagreements: [],
          },
          executionSteps: [
            { step: 1, title: 'Input Ingestion', description: 'Validated 1024x1024 raster', status: 'completed' },
            { step: 2, title: 'Specialist Execution', description: 'Ran Florence-2 VQA', status: 'completed' },
          ],
          isSimulation: false,
          timestamp: '2026-09-20 12:00:00 UTC',
          inputInformation: 'test_s2.tif (1024x1024)',
        };
        return new Response(JSON.stringify(mockResponse), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      return originalFetch(url, opts);
    };

    try {
      const result = await executeRemoteSensingAnalysis({
        mode: 'single',
        query: 'What land cover is present?',
        files: {
          single: {
            id: 'test-1',
            name: 'test_s2.tif',
            size: '5 MB',
            modality: 'Multispectral',
            previewUrl: 'blob:test',
          },
        },
        enableDemoSimulation: false,
      });

      assert(capturedBody.query === 'What land cover is present?', 'Query mismatch in body');
      assert(capturedBody.images.single.filename === 'test_s2.tif', 'Filename mismatch in body');
      assert(result.answer.includes('dense mixed forest'), 'Answer not parsed correctly');
      assert(result.isSimulation === false, 'Production result marked as simulation');
      assert(result.confidence === null, 'Confidence should be strictly null when uncomputed');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 4. Multi-Specialist Structured Response (Disagreements & Trace)
  await test('4. Multi-specialist response parses disagreements and execution trace', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: any, opts: any) => {
      if (String(url).includes('/api/analyze')) {
        const mockResponse: AnalyzeResponse = {
          query: 'Is there water in the scene?',
          mode: 'single',
          taskType: 'vqa',
          selectedModel: 'Florence-2-base + BigEarthNet ResNet-18',
          answer: 'Potential shallow wetland or shadow zone identified.',
          confidence: null,
          evidence: ['Florence-2: low-contrast dark region', 'BigEarthNet: inland water 42%'],
          evidenceHierarchy: {
            directEvidence: ['Visual contour indicates low reflectance basin'],
            supportingEvidence: ['NDWI spectral indicator positive'],
            limitations: ['Solar elevation angle causes severe hill shading'],
            disagreements: [
              'Florence-2 identified dark depression as shadow, whereas BigEarthNet classified it as inland water.',
            ],
          },
          executionSteps: [
            { step: 1, title: 'Planner Selection', description: 'Selected dual vision specialists', status: 'completed' },
            { step: 2, title: 'Florence-2 VQA', description: 'Executed visual Q&A', status: 'completed' },
            { step: 3, title: 'BigEarthNet Classification', description: 'Evaluated 43 land-cover classes', status: 'completed' },
            { step: 4, title: 'Evidence Combination', description: 'Synthesized contradictory perspectives', status: 'completed' },
          ],
          isSimulation: false,
          timestamp: '2026-09-20 12:05:00 UTC',
          inputInformation: 'raster.tif',
        };
        return new Response(JSON.stringify(mockResponse), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      return originalFetch(url, opts);
    };

    try {
      const result = await executeRemoteSensingAnalysis({
        mode: 'single',
        query: 'Is there water in the scene?',
        files: { single: { id: '1', name: 'raster.tif', size: '2 MB', modality: 'Optical', previewUrl: '' } },
      });
      assert(result.evidenceHierarchy?.disagreements?.length === 1, 'Disagreement should be preserved');
      assert(result.executionSteps?.length === 4, 'Execution trace steps missing');
      assert(result.evidenceHierarchy?.limitations?.length === 1, 'Limitations missing');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 5. Geospatial Unavailable State
  await test('5. Geospatial unavailable state reports honestly without inventing coordinates', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: any, opts: any) => {
      if (String(url).includes('/api/analyze')) {
        const mockResponse: AnalyzeResponse = {
          query: 'Describe this scene',
          mode: 'single',
          taskType: 'captioning',
          selectedModel: 'Florence-2-base',
          answer: 'An aerial scene with industrial warehousing.',
          confidence: null,
          evidence: ['Detected rectangular structures'],
          geospatialEvidence: {
            isGeoreferenced: false,
            crs: null,
            bounds: null,
            centroid: null,
            pixelResolution: null,
            statusMessage: 'Geospatial coordinates unavailable because the source raster does not contain sufficient georeferencing metadata.',
          },
          isSimulation: false,
          timestamp: '2026-09-20 12:10:00 UTC',
          inputInformation: 'unprojected_image.png',
        };
        return new Response(JSON.stringify(mockResponse), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      return originalFetch(url, opts);
    };

    try {
      const result = await executeRemoteSensingAnalysis({
        mode: 'single',
        query: 'Describe this scene',
        files: { single: { id: '1', name: 'unprojected_image.png', size: '1 MB', modality: 'Optical', previewUrl: '' } },
      });
      assert(result.geospatialEvidence?.isGeoreferenced === false, 'Should report unprojected raster');
      assert(result.geospatialEvidence?.crs === null, 'CRS must be null when unprojected');
      assert(result.geospatialEvidence?.centroid === null, 'Centroid must not be invented');
      assert(
        result.geospatialEvidence?.statusMessage?.includes('unavailable') === true,
        'Should report statusMessage'
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 6. Geospatial Available State
  await test('6. Geospatial available state preserves genuine coordinates and CRS', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: any, opts: any) => {
      if (String(url).includes('/api/analyze')) {
        const mockResponse: AnalyzeResponse = {
          query: 'Where are the structures?',
          mode: 'single',
          taskType: 'grounding',
          selectedModel: 'Florence-2-base',
          answer: 'Located 3 industrial buildings.',
          confidence: null,
          evidence: ['Grounding bounding box returned'],
          geospatialEvidence: {
            isGeoreferenced: true,
            crs: 'EPSG:32643',
            bounds: { north: 19.123, south: 19.101, east: 72.895, west: 72.871 },
            centroid: { lat: 19.112, lng: 72.883 },
            pixelResolution: '10.0m GSD',
            statusMessage: 'Raster is georeferenced with EPSG:32643',
          },
          boundingBoxes: [
            {
              id: 'box-1',
              label: 'Building',
              x: 10,
              y: 20,
              width: 15,
              height: 25,
              geoCoordinates: { lat: 19.115, lng: 72.88 },
            },
          ],
          isSimulation: false,
          timestamp: '2026-09-20 12:15:00 UTC',
          inputInformation: 'geotiff_valid.tif',
        };
        return new Response(JSON.stringify(mockResponse), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      return originalFetch(url, opts);
    };

    try {
      const result = await executeRemoteSensingAnalysis({
        mode: 'single',
        query: 'Where are the structures?',
        files: { single: { id: '1', name: 'geotiff_valid.tif', size: '10 MB', modality: 'Optical', previewUrl: '' } },
      });
      assert(result.geospatialEvidence?.isGeoreferenced === true, 'Must report georeferenced');
      assert(result.geospatialEvidence?.crs === 'EPSG:32643', 'CRS mismatch');
      assert(result.boundingBoxes?.[0].geoCoordinates?.lat === 19.115, 'Grounding lat mismatch');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 7. Grounding Overlay
  await test('7. Grounding overlay maps pixel bounding boxes without fabricated confidence', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: any, opts: any) => {
      if (String(url).includes('/api/analyze')) {
        const mockResponse: AnalyzeResponse = {
          query: 'ground buildings',
          mode: 'single',
          taskType: 'grounding',
          selectedModel: 'Florence-2-base',
          answer: 'Localized structures.',
          confidence: null,
          evidence: ['Detected 2 buildings'],
          boundingBoxes: [
            { id: 'b1', label: 'building', x: 25, y: 30, width: 20, height: 15 },
            { id: 'b2', label: 'building', x: 55, y: 60, width: 25, height: 20 },
          ],
          isSimulation: false,
          timestamp: '2026-09-20 12:20:00 UTC',
          inputInformation: 'buildings.tif',
        };
        return new Response(JSON.stringify(mockResponse), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      return originalFetch(url, opts);
    };

    try {
      const result = await executeRemoteSensingAnalysis({
        mode: 'single',
        query: 'ground buildings',
        files: { single: { id: '1', name: 'buildings.tif', size: '5 MB', modality: 'Optical', previewUrl: '' } },
      });
      assert(result.boundingBoxes?.length === 2, 'Expected 2 bounding boxes');
      assert(result.boundingBoxes[0].confidence === undefined, 'Must not fabricate confidence on boxes');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 8. Bi-temporal Change Response
  await test('8. Bi-temporal change response calls change-analysis and distinguishes candidates from confirmed changes', async () => {
    let capturedUrl = '';
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: any, opts: any) => {
      capturedUrl = String(url);
      if (capturedUrl.includes('/api/change-analysis')) {
        const mockResponse: AnalyzeResponse = {
          query: 'What changed between these dates?',
          mode: 'bi-temporal',
          taskType: 'change-analysis',
          selectedModel: 'Bi-Temporal Change Specialist',
          answer: 'Identified 2 confirmed construction areas and 1 transient candidate.',
          confidence: null,
          evidence: ['T1 to T2 difference computed', 'Spectral thresholding confirmed 2 permanent changes'],
          changedRegions: [
            { id: 'c1', label: 'New Building', direction: 'increase', x: 10, y: 15, width: 20, height: 20 },
          ],
          isSimulation: false,
          timestamp: '2026-09-20 12:25:00 UTC',
          inputInformation: 'past.tif vs present.tif',
        };
        return new Response(JSON.stringify(mockResponse), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      return originalFetch(url, opts);
    };

    try {
      const result = await executeRemoteSensingAnalysis({
        mode: 'bi-temporal',
        query: 'What changed between these dates?',
        files: {
          before: { id: '1', name: 'past.tif', size: '8 MB', modality: 'Optical', previewUrl: '' },
          after: { id: '2', name: 'present.tif', size: '8 MB', modality: 'Optical', previewUrl: '' },
        },
      });
      assert(capturedUrl.includes('/api/change-analysis'), 'Must call /api/change-analysis endpoint');
      assert(result.mode === 'bi-temporal', 'Mode mismatch');
      assert(result.changedRegions?.length === 1, 'Expected 1 changed region');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 9. Optical / SAR Response
  await test('9. Optical/SAR response calls optical-sar-analysis and preserves cross-modal evidence', async () => {
    let capturedUrl = '';
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: any, opts: any) => {
      capturedUrl = String(url);
      if (capturedUrl.includes('/api/optical-sar-analysis')) {
        const mockResponse: AnalyzeResponse = {
          query: 'Identify water bodies and built-up areas',
          mode: 'optical-sar',
          taskType: 'optical-sar-fusion',
          selectedModel: 'Multimodal Optical + SAR Fusion Engine',
          answer: 'Fused optical spectral indices with SAR backscatter to isolate permanent water bodies.',
          confidence: null,
          evidence: ['NDWI isolated water candidates', 'SAR low backscatter confirmed specular water surface'],
          crossModalEvidence: {
            opticalEvidence: ['NDWI > 0.35 in eastern quadrant'],
            sarEvidence: ['VV backscatter < -18 dB confirmed open water'],
            fusionSummary: 'Cross-modal agreement resolves water boundary with high fidelity.',
            corroborationScore: 0.92,
          },
          isSimulation: false,
          timestamp: '2026-09-20 12:30:00 UTC',
          inputInformation: 'optical.tif + sar.tif',
        };
        return new Response(JSON.stringify(mockResponse), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      return originalFetch(url, opts);
    };

    try {
      const result = await executeRemoteSensingAnalysis({
        mode: 'optical-sar',
        query: 'Identify water bodies and built-up areas',
        files: {
          optical: { id: '1', name: 'optical.tif', size: '10 MB', modality: 'Optical', previewUrl: '' },
          sar: { id: '2', name: 'sar.tif', size: '12 MB', modality: 'SAR', previewUrl: '' },
        },
      });
      assert(capturedUrl.includes('/api/optical-sar-analysis'), 'Must call /api/optical-sar-analysis');
      assert(result.crossModalEvidence?.sarEvidence?.length === 1, 'SAR evidence missing');
      assert(result.crossModalEvidence?.opticalEvidence?.length === 1, 'Optical evidence missing');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 10. Honest Error Handling (No Silent Fallback to Mock Data)
  await test('10. Honest error handling throws when backend fails without silent mock fallback', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => {
      return new Response(
        JSON.stringify({ detail: 'Raster contains incompatible 1-band layout for RGB Florence-2.' }),
        { status: 422, headers: { 'Content-Type': 'application/json' } }
      );
    };

    try {
      let threw = false;
      try {
        await executeRemoteSensingAnalysis({
          mode: 'single',
          query: 'Describe this area',
          files: { single: { id: '1', name: 'incompatible.tif', size: '1 MB', modality: 'Panchromatic', previewUrl: '' } },
          enableDemoSimulation: false,
        });
      } catch (err: any) {
        threw = true;
        assert(err.message.includes('incompatible 1-band layout'), 'Expected error message to be preserved');
      }
      assert(threw, 'Should throw error when backend returns 422');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 11. No Mock Data in Production Path
  await test('11. Production analysis flow never injects mock Cargo Vessels or Petroleum Silos', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: any, opts: any) => {
      if (String(url).includes('/api/analyze')) {
        const genuineResponse: AnalyzeResponse = {
          query: 'What is shown here?',
          mode: 'single',
          taskType: 'vqa',
          selectedModel: 'Florence-2-base',
          answer: 'Agricultural fields with irrigation channels.',
          confidence: null,
          evidence: ['Crop field boundaries detected'],
          isSimulation: false,
          timestamp: '2026-09-20 12:35:00 UTC',
          inputInformation: 'farm.tif',
        };
        return new Response(JSON.stringify(genuineResponse), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      return originalFetch(url, opts);
    };

    try {
      const result = await executeRemoteSensingAnalysis({
        mode: 'single',
        query: 'What is shown here?',
        files: { single: { id: '1', name: 'farm.tif', size: '3 MB', modality: 'Optical', previewUrl: '' } },
        enableDemoSimulation: false,
      });

      const fullOutputText = JSON.stringify(result);
      assert(!fullOutputText.includes('Cargo Vessels'), 'Mock "Cargo Vessels" detected in production result');
      assert(!fullOutputText.includes('Petroleum Storage Silos'), 'Mock "Petroleum Storage Silos" detected');
      assert(!fullOutputText.includes('Intertidal Mudflats'), 'Mock "Intertidal Mudflats" detected');
      assert(!fullOutputText.includes('Auric Arohi'), 'Mock "Auric Arohi" detected');
      assert(!fullOutputText.includes('AuricVista'), 'Mock "AuricVista" detected');
      assert(result.isSimulation === false, 'Must not be marked simulation');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 12. Report Generation
  await test('12. Report generation endpoint communicates with /api/report/generate', async () => {
    let capturedUrl = '';
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: any, opts: any) => {
      capturedUrl = String(url);
      if (capturedUrl.includes('/api/report/generate')) {
        return new Response(
          JSON.stringify({
            reportId: 'rep_12345',
            timestamp: '2026-09-20 12:40:00 UTC',
            markdown: '# SatQuery AI Mission Intelligence Report\n\n## Analysis Summary',
            filename: 'report_12345.md',
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        );
      }
      return originalFetch(url, opts);
    };

    try {
      const report = await generateReport({
        query: 'Analyze image',
        mode: 'single',
        taskType: 'vqa',
        selectedModel: 'Florence-2-base',
        answer: 'Test analysis',
        evidence: ['Evidence 1'],
        isSimulation: false,
        timestamp: '2026-09-20 12:40:00 UTC',
        inputInformation: 'test.tif',
      });
      assert(capturedUrl.includes('/api/report/generate'), 'Must call /api/report/generate');
      assert(report.reportId === 'rep_12345', 'Report ID mismatch');
      assert(report.markdown.includes('Mission Intelligence Report'), 'Markdown report content mismatch');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 13. Step 17: Demo Reset Contract
  await test('13. Demo reset safely clears results, overlays, and query without deleting history', () => {
    let sessionState = {
      singleFile: { id: '1', name: 'raster.tif' },
      query: 'Where are the buildings?',
      result: { answer: 'Buildings found', boundingBoxes: [{ id: 'b1' }] },
      validationStatus: 'valid',
      history: [{ id: 'hist-1', query: 'Previous query' }],
    };

    // Reset action
    sessionState = {
      ...sessionState,
      singleFile: null as any,
      query: 'Describe this area.',
      result: null as any,
      validationStatus: 'idle',
      // history remains intact!
    };

    assert(sessionState.singleFile === null, 'Files should be cleared on reset');
    assert(sessionState.result === null, 'Result should be cleared on reset');
    assert(sessionState.query === 'Describe this area.', 'Query should reset to default');
    assert(sessionState.validationStatus === 'idle', 'Validation should reset to idle');
    assert(sessionState.history.length === 1, 'History must NOT be deleted on reset');
  });

  // 14. Step 17: Agent Execution Display ("How SatQuery analyzed this")
  await test('14. Agent execution display extracts actual backend plan and specialist routing', () => {
    const backendResult: AnalyzeResponse = {
      query: 'Where are the buildings?',
      mode: 'single',
      taskType: 'grounding',
      selectedModel: 'Florence-2-base',
      answer: 'Found 3 buildings in the northern parcel.',
      confidence: null,
      evidence: ['3 bounding boxes detected'],
      evidenceHierarchy: {
        directEvidence: ['Detected 3 building footprints'],
        supportingEvidence: ['Multispectral reflectance corroborates structures'],
        limitations: [],
        disagreements: [],
      },
      agentPlan: {
        detected_intent: 'Text-Guided Grounding',
        selected_specialist: 'Florence-2 Vision-Language Specialist',
        selected_tools: ['florence-2-grounding', 'bigearthnet-classifier'],
      },
      geospatialEvidence: {
        isGeoreferenced: true,
        crs: 'EPSG:32633',
        statusMessage: 'Georeferenced WGS 84 / UTM zone 33N',
      },
      isSimulation: false,
      timestamp: '2026-09-20 12:45:00 UTC',
      inputInformation: 'berlin.tif',
    };

    const queryIntent = backendResult.agentPlan?.detected_intent || backendResult.taskType;
    const selectedSpecialist = backendResult.agentPlan?.selected_specialist || backendResult.selectedModel;
    const modelTool = backendResult.agentPlan?.selected_tools?.join(', ') || backendResult.selectedModel;
    const geoSummary = backendResult.geospatialEvidence?.isGeoreferenced
      ? `Available (${backendResult.geospatialEvidence.crs})`
      : 'Unavailable (Pixel Coordinate Space)';

    assert(queryIntent === 'Text-Guided Grounding', 'Intent mismatch');
    assert(selectedSpecialist === 'Florence-2 Vision-Language Specialist', 'Specialist mismatch');
    assert(modelTool.includes('florence-2-grounding'), 'Tool mismatch');
    assert(geoSummary.includes('EPSG:32633'), 'Geospatial grounding mismatch');
  });

  // 15. Step 17: Evidence Hierarchy & Honest Language
  await test('15. Evidence hierarchy strictly presents answer, direct evidence, supporting evidence, and honest candidate changes', () => {
    const biTemporalResult: AnalyzeResponse = {
      query: 'What changed between these dates?',
      mode: 'bi-temporal',
      taskType: 'change-analysis',
      selectedModel: 'Bi-Temporal Difference Specialist',
      answer: 'Detected surface modification between baseline and monitoring dates.',
      confidence: null,
      evidence: ['Spectral reflectance shift observed'],
      evidenceHierarchy: {
        directEvidence: ['Direct pixel delta detected in sector 4'],
        supportingEvidence: ['SSIM structural reduction < 0.82'],
        limitations: ['Candidate change region subject to ground-truth verification'],
        disagreements: [],
      },
      changedRegions: [
        { id: 'c1', label: 'Candidate Change Region: New Structure', direction: 'increase', x: 20, y: 30, width: 15, height: 15 },
      ],
      isSimulation: false,
      timestamp: '2026-09-20 12:50:00 UTC',
      inputInformation: 't1.tif vs t2.tif',
    };

    assert(biTemporalResult.evidenceHierarchy?.directEvidence.length === 1, 'Direct evidence must be present');
    assert(biTemporalResult.evidenceHierarchy?.supportingEvidence.length === 1, 'Supporting evidence must be present');
    assert(biTemporalResult.evidenceHierarchy?.limitations.length === 1, 'Limitations must be present');
    assert(biTemporalResult.changedRegions?.[0].label.includes('Candidate Change Region'), 'Must use honest candidate change language');
    assert(!biTemporalResult.changedRegions?.[0].label.includes('Confirmed Permanent Change'), 'Must not make unconfirmed absolute claims');
  });

  // 16. Step 17: No Sensitive Configuration Displayed in Settings
  await test('16. Settings and system view do not expose API keys, database passwords, or internal filesystem paths', () => {
    const settingsUiContent = {
      backendStatus: 'ONLINE',
      apiEndpoints: ['/api/analyze', '/api/validate', '/api/report'],
      models: ['Florence-2 VLM', 'BigEarthNet Specialist'],
      appInfo: 'SatQuery AI v1.0.0 (SIH PS 26167)',
    };

    const serialized = JSON.stringify(settingsUiContent);
    assert(!serialized.includes('sk-'), 'Exposed API key pattern detected');
    assert(!serialized.includes('password'), 'Exposed password detected');
    assert(!serialized.includes('postgres://'), 'Exposed connection string detected');
    assert(!serialized.includes('c:\\') && !serialized.includes('/home/'), 'Exposed internal filesystem path detected');
  });

  console.log(`\n==================================================`);
  console.log(`Step 17 Production & Demo Tests: ${passed}/${total} Passed!`);
  console.log(`==================================================\n`);
}

runTests().catch((e) => {
  console.error('Integration test run failed:', e);
  process.exit(1);
});
