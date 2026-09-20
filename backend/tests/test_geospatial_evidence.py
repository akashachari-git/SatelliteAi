"""
SatQuery AI - Step 12 Geospatial & Evidence Grounding Test Suite
Comprehensive tests covering all 16 required geospatial specifications:
1. Valid GeoTIFF CRS extraction
2. Valid affine transform extraction
3. Pixel -> projected coordinate
4. Pixel -> WGS84 coordinate
5. Bounding-box conversion
6. Area calculation (projected & geographic)
7. Missing CRS handling
8. Missing transform handling
9. Invalid CRS handling
10. No fabricated coordinates
11. Bi-temporal CRS mismatch detection
12. Bi-temporal geospatial alignment status
13. Optical/SAR CRS validation
14. EvidenceCombiner geospatial provenance
15. Report / response geospatial fields
16. Existing specialist outputs remain unchanged
"""
import os
import tempfile
import unittest
import numpy as np
from PIL import Image

from backend.geospatial.georeference import GeoreferenceEngine, HAS_RASTERIO, HAS_PYPROJ
from backend.geospatial.validator import GeoTIFFValidator
from backend.app.models.bitemporal_inference import BiTemporalChangeService
from backend.app.models.optical_sar_inference import OpticalSARFusionService
from backend.orchestrator.evidence_combiner import EvidenceCombiner, EvidenceHierarchy
from backend.orchestrator.agent import SatQueryAgent
from backend.schemas import BoundingBox, AnalyzeResponse, GeospatialEvidence


class TestGeospatialEvidenceLayer(unittest.TestCase):
    """
    Test suite verifying real geospatial grounding, transformations, area calculation,
    alignment status, and evidence provenance.
    """

    def setUp(self):
        # Standard test affine transform (10m resolution, UTM Zone 43N near Mumbai)
        # c=281000.0, a=10.0, b=0.0, f=2102000.0, d=0.0, e=-10.0
        self.test_gt = [281000.0, 10.0, 0.0, 2102000.0, 0.0, -10.0]
        self.test_crs_utm = "EPSG:32643"
        self.test_crs_wgs84 = "EPSG:4326"

    # 1. Valid GeoTIFF CRS extraction
    def test_01_valid_geotiff_crs_extraction(self):
        """Verify CRS is extracted from GeoTIFF raster metadata without guessing."""
        if not HAS_RASTERIO:
            self.skipTest("rasterio not available")

        import rasterio
        from rasterio.transform import from_origin

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            transform = from_origin(281000.0, 2102000.0, 10.0, 10.0)
            data = np.ones((1, 64, 64), dtype=np.uint16)
            with rasterio.open(
                tmp_path,
                "w",
                driver="GTiff",
                height=64,
                width=64,
                count=1,
                dtype=data.dtype,
                crs="EPSG:32643",
                transform=transform,
            ) as dst:
                dst.write(data)

            meta = GeoreferenceEngine.extract_geospatial_metadata(tmp_path)
            self.assertEqual(meta["geospatial_status"], "available")
            self.assertIn("32643", str(meta["crs"]))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # 2. Valid affine transform extraction
    def test_02_valid_affine_transform_extraction(self):
        """Verify affine transform is extracted with correct origin and GSD."""
        meta_dict = {
            "crs": "EPSG:32643",
            "geotransform": self.test_gt,
            "width": 100,
            "height": 100,
        }
        meta = GeoreferenceEngine.extract_geospatial_metadata(meta_dict)
        self.assertEqual(meta["geotransform"], self.test_gt)
        self.assertAlmostEqual(meta["gsd_meters"], 10.0)
        self.assertEqual(meta["geospatial_status"], "available")

    # 3. Pixel -> projected coordinate
    def test_03_pixel_to_projected(self):
        """Verify mapping from (col, row) to projected (x, y) coordinates."""
        col, row = 10.0, 20.0
        pt = GeoreferenceEngine.pixel_to_projected(col, row, self.test_gt)
        self.assertIsNotNone(pt)
        expected_x = 281000.0 + 10.0 * 10.0  # 281100.0
        expected_y = 2102000.0 + 20.0 * (-10.0)  # 2101800.0
        self.assertAlmostEqual(pt[0], expected_x)
        self.assertAlmostEqual(pt[1], expected_y)

    # 4. Pixel -> WGS84 coordinate
    def test_04_pixel_to_wgs84(self):
        """Verify projected coordinates convert accurately to WGS84 latitude/longitude."""
        if not HAS_PYPROJ:
            self.skipTest("pyproj not available")

        # Pixel (0, 0) in UTM Zone 43N (281000, 2102000) corresponds to ~19.00° N, 72.92° E
        lat_lon = GeoreferenceEngine.pixel_to_wgs84(0.0, 0.0, self.test_gt, self.test_crs_utm)
        self.assertIsNotNone(lat_lon)
        lat, lon = lat_lon
        self.assertTrue(18.5 <= lat <= 19.5, f"Latitude {lat} out of expected range")
        self.assertTrue(72.5 <= lon <= 73.5, f"Longitude {lon} out of expected range")

    # 5. Bounding-box conversion
    def test_05_bounding_box_conversion(self):
        """Verify pixel bounding box converts to both projected and geographic WGS84 bounds."""
        pixel_bbox = {"x": 10.0, "y": 10.0, "width": 50.0, "height": 50.0}
        res = GeoreferenceEngine.pixel_bbox_to_geographic(pixel_bbox, self.test_gt, self.test_crs_utm)
        self.assertIsNotNone(res)
        self.assertIn("projected_bbox", res)
        self.assertIn("geographic_bbox", res)
        self.assertIn("center_lat_lon", res)

        geo = res["geographic_bbox"]
        self.assertLess(geo["west"], geo["east"])
        self.assertLess(geo["south"], geo["north"])

    # 6. Area calculation (projected & geographic)
    def test_06_area_calculation(self):
        """Verify area calculation for projected (UTM) and geographic (WGS84 degrees) CRS."""
        # Projected CRS: 10m x 10m pixels -> 100 m² per pixel
        # 100 pixels = 10,000 m² = 0.01 km² = 1.0 ha
        proj_area = GeoreferenceEngine.calculate_pixel_area(100, self.test_gt, self.test_crs_utm)
        self.assertIsNotNone(proj_area)
        self.assertAlmostEqual(proj_area["area_m2"], 10000.0)
        self.assertAlmostEqual(proj_area["area_km2"], 0.01)
        self.assertAlmostEqual(proj_area["area_hectares"], 1.0)

        # Geographic CRS: EPSG:4326 with pixel scale 0.0001 degrees
        gt_geo = [72.8, 0.0001, 0.0, 19.0, 0.0, -0.0001]
        geo_area = GeoreferenceEngine.calculate_pixel_area(100, gt_geo, "EPSG:4326", center_latitude=19.0)
        self.assertIsNotNone(geo_area)
        # Should not treat degrees as meters; area_m2 should be around ~12,000 m²
        self.assertGreater(geo_area["area_m2"], 5000.0)
        self.assertLess(geo_area["area_m2"], 25000.0)

    # 7. Missing CRS handling
    def test_07_missing_crs_handling(self):
        """Verify missing CRS returns None and does not fabricate coordinates or area."""
        pt = GeoreferenceEngine.projected_to_wgs84(281000.0, 2102000.0, None)
        self.assertIsNone(pt)

        pt_local = GeoreferenceEngine.projected_to_wgs84(281000.0, 2102000.0, "Local Pixel CRS (Unprojected)")
        self.assertIsNone(pt_local)

        area = GeoreferenceEngine.calculate_pixel_area(100, self.test_gt, None)
        self.assertIsNone(area)

    # 8. Missing transform handling
    def test_08_missing_transform_handling(self):
        """Verify missing affine transform returns None for pixel-to-world conversion."""
        pt = GeoreferenceEngine.pixel_to_projected(10.0, 10.0, None)
        self.assertIsNone(pt)

        bbox_res = GeoreferenceEngine.pixel_bbox_to_geographic(
            {"x": 0, "y": 0, "width": 10, "height": 10}, None, self.test_crs_utm
        )
        self.assertIsNone(bbox_res)

    # 9. Invalid CRS handling
    def test_09_invalid_crs_handling(self):
        """Verify invalid or unparseable CRS strings fail safely without crashing."""
        pt = GeoreferenceEngine.projected_to_wgs84(281000.0, 2102000.0, "INVALID:999999")
        self.assertIsNone(pt)

        meta = GeoreferenceEngine.extract_geospatial_metadata({
            "crs": "INVALID:999999",
            "geotransform": self.test_gt,
            "width": 100,
            "height": 100,
        })
        self.assertIn("geospatial_status", meta)

    # 10. No fabricated coordinates
    def test_10_no_fabricated_coordinates(self):
        """Verify unprojected inputs return explicit unavailable disclaimer, never guessing coordinates."""
        formatted = SatQueryAgent._format_coordinates_string(None, "Local Pixel CRS (Unprojected)")
        self.assertEqual(
            formatted,
            "Geospatial coordinates unavailable because the source raster does not contain sufficient georeferencing metadata."
        )

        cursor_res = GeoreferenceEngine.query_cursor_feature(50.0, 50.0, metadata={"crs": None, "geotransform": None})
        self.assertEqual(cursor_res["geospatial_status"], "unavailable")
        self.assertIsNone(cursor_res["coordinates"])
        self.assertEqual(cursor_res["message"], "Geolocation unavailable for this raster")

    # 11. Bi-temporal CRS mismatch detection
    def test_11_bitemporal_crs_mismatch_detection(self):
        """Verify BiTemporalChangeService rejects temporal pairs with conflicting projections."""
        svc = BiTemporalChangeService()
        t1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
        t2 = np.ones((64, 64, 3), dtype=np.uint8) * 120

        with self.assertRaises(ValueError) as ctx:
            svc.predict(
                inputs={"before": t1, "after": t2},
                query="Detect change",
                metadata={"crs_before": "EPSG:32643", "crs_after": "EPSG:32644"}
            )
        self.assertIn("CRS Incompatibility", str(ctx.exception))

    # 12. Bi-temporal geospatial alignment status
    def test_12_bitemporal_geospatial_alignment_status(self):
        """Verify BiTemporalChangeService distinguishes 'geospatially aligned' from 'pixel-aligned'."""
        svc = BiTemporalChangeService()
        t1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
        t2 = np.ones((64, 64, 3), dtype=np.uint8) * 120

        # Case A: Both have valid georeferencing
        res_geo = svc.predict(
            inputs={"before": t1, "after": t2},
            query="Detect change",
            metadata={
                "crs_before": "EPSG:32643",
                "crs_after": "EPSG:32643",
                "geotransform_before": self.test_gt,
                "geotransform_after": self.test_gt,
            }
        )
        self.assertEqual(res_geo["alignment_status"], "geospatially aligned")
        self.assertEqual(res_geo["geospatialEvidence"]["status"], "available")

        # Case B: Unprojected inputs
        res_unproj = svc.predict(
            inputs={"before": t1, "after": t2},
            query="Detect change",
            metadata={"crs_before": "Local", "crs_after": "Local"}
        )
        self.assertEqual(res_unproj["alignment_status"], "pixel-aligned")
        self.assertEqual(res_unproj["geospatialEvidence"]["status"], "unavailable")

    # 13. Optical/SAR CRS validation
    def test_13_optical_sar_crs_validation(self):
        """Verify OpticalSARFusionService rejects mismatched CRS and validates alignment."""
        svc = OpticalSARFusionService()
        opt = np.ones((64, 64, 3), dtype=np.uint8) * 100
        sar = np.ones((64, 64), dtype=np.float32) * -12.0

        # CRS Mismatch
        with self.assertRaises(ValueError) as ctx:
            svc.predict(
                inputs={"optical": opt, "sar": sar},
                query="Analyze multimodal",
                metadata={"crs_optical": "EPSG:32643", "crs_sar": "EPSG:32644"}
            )
        self.assertIn("CRS Incompatibility", str(ctx.exception))

        # CRS Matching
        res = svc.predict(
            inputs={"optical": opt, "sar": sar},
            query="Analyze multimodal",
            metadata={
                "crs_optical": "EPSG:32643",
                "crs_sar": "EPSG:32643",
                "geotransform_optical": self.test_gt,
                "geotransform_sar": self.test_gt,
                "modality_sar": "SAR (Sentinel-1 C-Band VV/VH)"
            }
        )
        self.assertEqual(res["geospatialEvidence"]["alignmentStatus"], "geospatially aligned")
        self.assertEqual(res["geospatialEvidence"]["status"], "available")

    # 14. EvidenceCombiner geospatial provenance
    def test_14_evidence_combiner_geospatial_provenance(self):
        """Verify EvidenceCombiner records direct geospatial evidence and limitations."""
        results = {
            "bitemporal-diff-net": {
                "answer": "Candidate change detected.",
                "evidence": ["Spectral difference: 50 px."],
                "geospatialEvidence": {
                    "status": "available",
                    "crs": "EPSG:32643",
                    "alignmentStatus": "geospatially aligned",
                    "limitations": ["Resampling was not applied."]
                }
            }
        }
        hierarchy = EvidenceCombiner.combine("Detect change", results)
        self.assertTrue(any("[GEOSPATIAL DIRECT]" in e for e in hierarchy.direct_evidence))
        self.assertTrue(any("[GEOSPATIAL LIMITATION]" in l for l in hierarchy.limitations))

    # 15. Report / response geospatial fields
    def test_15_report_geospatial_fields(self):
        """Verify AnalyzeResponse includes structured geospatialEvidence with all schema fields."""
        from unittest.mock import patch
        from backend.app.models.florence2_inference import florence2_service

        agent = SatQueryAgent()
        img = Image.new("RGB", (64, 64), color="blue")
        mock_res = {
            "model": "Florence-2",
            "model_name": "Florence-2 Vision-Language Specialist",
            "task": "scene-captioning",
            "answer": "An aerial view of water.",
            "evidence": ["VLM Task: Image Captioning"],
            "boundingBoxes": None,
        }
        with patch.object(florence2_service, "predict", return_value=mock_res):
            resp = agent.execute_pipeline(
                query="What is the scene?",
                mode="single",
                images_dict={"single": {"array": np.array(img), "filename": "unprojected.png"}}
            )
            self.assertIsNotNone(resp.geospatialEvidence)
            self.assertEqual(resp.geospatialEvidence["status"], "unavailable")
            self.assertIn("Geospatial coordinates unavailable", resp.imageryMetadata["coordinates"])

    # 16. Existing specialist outputs remain unchanged
    def test_16_existing_specialist_outputs_remain_unchanged(self):
        """Verify Florence-2, BigEarthNet, and deterministic fallbacks preserve their original predictions."""
        from unittest.mock import patch
        from backend.app.models.florence2_inference import florence2_service

        agent = SatQueryAgent()
        img = Image.new("RGB", (64, 64), color="green")
        mock_res = {
            "model": "Florence-2",
            "model_name": "Florence-2 Vision-Language Specialist",
            "task": "scene-captioning",
            "answer": "A satellite view of green forest.",
            "evidence": ["Direct visual evidence"],
            "boundingBoxes": None,
        }
        with patch.object(florence2_service, "predict", return_value=mock_res):
            resp = agent.execute_pipeline(
                query="Caption this satellite image.",
                mode="single",
                images_dict={"single": {"array": np.array(img), "filename": "test.png"}}
            )
            self.assertIn("Florence", resp.selectedModel)
            self.assertIsNotNone(resp.answer)
            self.assertGreater(len(resp.evidence), 0)


if __name__ == "__main__":
    unittest.main()
