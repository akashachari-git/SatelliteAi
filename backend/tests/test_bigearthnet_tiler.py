"""
Step 6 Test Suite: Real Spatial Tile/Window BigEarthNet Analysis.
Validates:
A. 120x120 raster -> exactly 1 tile.
B. 240x240 raster -> 4 tiles with 120 stride.
C. 300x250 raster -> correct edge-tile handling (9 tiles: 3x3, boundary padding).
D. Each tile reaches the real ONNX model.
E. Each tile produces exactly 19 outputs.
F. Aggregation produces exactly 19 classes.
G. Supporting tile counts never exceed total tile count.
H. No invented geolocation when CRS/transform is absent.
I. Real CRS/transform is preserved when present.
J. RGB image remains rejected.
K. Existing BigEarthNet tests still pass.
L. Complete backend test suite still passes.
"""
import unittest
import numpy as np

from backend.models.bigearthnet_loader import (
    BIGEARTHNET_19_CLASSES,
    REQUIRED_S2_BANDS,
    bigearthnet_loader,
)
from backend.app.models.bigearthnet_tiler import (
    BigEarthNetTiler,
    bigearthnet_tiler,
)
from backend.app.models.bigearthnet_inference import (
    bigearthnet_service,
)
from backend.orchestrator.agent import SatQueryAgent


def create_synthetic_sentinel2_test_data(height: int, width: int) -> np.ndarray:
    """
    Creates a realistic synthetic 10-band Sentinel-2 raster for pipeline testing.
    LABELED CLEARLY AS TEST DATA: NOT A REAL-WORLD SATELLITE OBSERVATION.

    Simulates characteristic multispectral reflectance:
    - Vegetated canopy (chlorophyll absorption in B04, red-edge surge in B05-B07, NIR peak in B08/B8A).
    - Water/non-vegetated gradient.
    """
    raster = np.zeros((10, height, width), dtype=np.float32)
    half_h = height // 2

    # Top half: Vegetation signature (reflectance in [0, 1])
    raster[0, :half_h, :] = 0.04  # B02 Blue
    raster[1, :half_h, :] = 0.08  # B03 Green
    raster[2, :half_h, :] = 0.04  # B04 Red
    raster[3, :half_h, :] = 0.22  # B05 RedEdge 1
    raster[4, :half_h, :] = 0.42  # B06 RedEdge 2
    raster[5, :half_h, :] = 0.48  # B07 RedEdge 3
    raster[6, :half_h, :] = 0.52  # B08 NIR
    raster[7, :half_h, :] = 0.53  # B8A Narrow NIR
    raster[8, :half_h, :] = 0.20  # B11 SWIR 1
    raster[9, :half_h, :] = 0.10  # B12 SWIR 2

    # Bottom half: Water signature (strong NIR/SWIR absorption)
    raster[0, half_h:, :] = 0.06  # B02 Blue
    raster[1, half_h:, :] = 0.05  # B03 Green
    raster[2, half_h:, :] = 0.02  # B04 Red
    raster[3, half_h:, :] = 0.01  # B05 RedEdge 1
    raster[4, half_h:, :] = 0.005 # B06 RedEdge 2
    raster[5, half_h:, :] = 0.005 # B07 RedEdge 3
    raster[6, half_h:, :] = 0.004 # B08 NIR
    raster[7, half_h:, :] = 0.003 # B8A Narrow NIR
    raster[8, half_h:, :] = 0.002 # B11 SWIR 1
    raster[9, half_h:, :] = 0.001 # B12 SWIR 2

    return raster


class TestBigEarthNetTiler(unittest.TestCase):
    """Test suite for Step 6: Real spatial tile/window BigEarthNet analysis."""

    @classmethod
    def setUpClass(cls):
        cls.tiler = BigEarthNetTiler(tile_size=120, stride=120)
        cls.agent = SatQueryAgent()

    def test_A_120x120_raster_yields_exactly_one_tile(self):
        """A. 120x120 raster -> exactly 1 tile."""
        windows = self.tiler.generate_tile_windows(height=120, width=120)
        self.assertEqual(len(windows), 1)

        win = windows[0]
        self.assertEqual(win["tile_id"], "tile_0_0")
        self.assertEqual(win["row"], 0)
        self.assertEqual(win["column"], 0)
        self.assertEqual(win["pixel_bounds"], {"x_min": 0, "y_min": 0, "x_max": 120, "y_max": 120})
        self.assertFalse(win["is_padded"])
        self.assertEqual(win["valid_pixel_area"], 120 * 120)

        # Execute tile inference end-to-end
        raster_120 = create_synthetic_sentinel2_test_data(120, 120)
        result = self.tiler.run_tile_inference(raster_120)
        self.assertEqual(result["tile_count"], 1)
        self.assertEqual(len(result["tiles"]), 1)

    def test_B_240x240_raster_yields_four_tiles_with_120_stride(self):
        """B. 240x240 raster -> 4 tiles with 120 stride."""
        windows = self.tiler.generate_tile_windows(height=240, width=240)
        self.assertEqual(len(windows), 4)

        expected_bounds = [
            {"x_min": 0, "y_min": 0, "x_max": 120, "y_max": 120},
            {"x_min": 120, "y_min": 0, "x_max": 240, "y_max": 120},
            {"x_min": 0, "y_min": 120, "x_max": 120, "y_max": 240},
            {"x_min": 120, "y_min": 120, "x_max": 240, "y_max": 240},
        ]
        for i, exp in enumerate(expected_bounds):
            self.assertEqual(windows[i]["pixel_bounds"], exp)
            self.assertFalse(windows[i]["is_padded"])

        # Execute tile inference end-to-end
        raster_240 = create_synthetic_sentinel2_test_data(240, 240)
        result = self.tiler.run_tile_inference(raster_240)
        self.assertEqual(result["tile_count"], 4)
        self.assertEqual(len(result["tiles"]), 4)

    def test_C_300x250_raster_correct_edge_tile_handling(self):
        """
        C. 300x250 raster -> correct edge-tile handling.
        Width: 300 px -> cols: [0..120], [120..240], [240..300] (width=60, padded=60) -> 3 cols
        Height: 250 px -> rows: [0..120], [120..240], [240..250] (height=10, padded=110) -> 3 rows
        Total windows: 3 x 3 = 9 tiles.
        """
        windows = self.tiler.generate_tile_windows(height=250, width=300)
        self.assertEqual(len(windows), 9)

        # Check edge tiles have correct padding recorded
        # Bottom-right tile (row 2, col 2): x in [240, 300], y in [240, 250]
        br_tile = next(w for w in windows if w["row"] == 2 and w["column"] == 2)
        self.assertEqual(br_tile["original_width"], 60)
        self.assertEqual(br_tile["original_height"], 10)
        self.assertEqual(br_tile["pad_right"], 60)
        self.assertEqual(br_tile["pad_bottom"], 110)
        self.assertTrue(br_tile["is_padded"])
        self.assertEqual(br_tile["valid_pixel_area"], 60 * 10)

        # Verify extract_and_pad_tile produces exact (10, 120, 120) shape with zero-padding
        raster_300x250 = create_synthetic_sentinel2_test_data(250, 300)
        padded_tensor = self.tiler.extract_and_pad_tile(raster_300x250, br_tile)
        self.assertEqual(padded_tensor.shape, (10, 120, 120))
        # Check that the padded region contains neutral zeros
        self.assertTrue(np.all(padded_tensor[:, 10:, :] == 0.0))
        self.assertTrue(np.all(padded_tensor[:, :, 60:] == 0.0))

        # Run inference across all 9 tiles
        result = self.tiler.run_tile_inference(raster_300x250)
        self.assertEqual(result["tile_count"], 9)
        self.assertEqual(len(result["tiles"]), 9)

    def test_D_each_tile_reaches_the_real_onnx_model(self):
        """D. Each tile reaches the real ONNX model (uses cached session, returns genuine probs)."""
        raster = create_synthetic_sentinel2_test_data(240, 240)
        result = self.tiler.run_tile_inference(raster)

        self.assertEqual(result["tile_count"], 4)
        # Verify inference engine is ONNX
        self.assertEqual(bigearthnet_loader.engine.value.upper(), "ONNX")

        # Verify each tile has distinct probabilities reflecting real model evaluation
        tile_probs_0 = result["tiles"][0]["probability_vector"]
        tile_probs_3 = result["tiles"][3]["probability_vector"]
        self.assertEqual(len(tile_probs_0), 19)
        self.assertEqual(len(tile_probs_3), 19)
        # The top half of our test raster has vegetation and bottom half has water
        # Tile 0 (top-left) should have different probabilities than Tile 3 (bottom-right)
        self.assertFalse(np.allclose(tile_probs_0, tile_probs_3))

    def test_E_each_tile_produces_exactly_19_outputs(self):
        """E. Each tile produces exactly 19 outputs."""
        raster = create_synthetic_sentinel2_test_data(250, 300)
        result = self.tiler.run_tile_inference(raster)

        for tile in result["tiles"]:
            self.assertEqual(len(tile["probability_vector"]), 19)
            self.assertEqual(len(tile["probabilities"]), 19)
            for prob in tile["probabilities"].values():
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)

    def test_F_aggregation_produces_exactly_19_classes(self):
        """F. Aggregation produces exactly 19 classes."""
        raster = create_synthetic_sentinel2_test_data(240, 240)
        result = self.tiler.run_tile_inference(raster)

        self.assertEqual(len(result["class_aggregates"]), 19)
        self.assertEqual(len(result["image_probabilities"]), 19)
        for cls_name in BIGEARTHNET_19_CLASSES:
            self.assertIn(cls_name, result["class_aggregates"])
            self.assertIn(cls_name, result["image_probabilities"])
            agg = result["class_aggregates"][cls_name]
            self.assertEqual(agg["class"], cls_name)
            self.assertIn("model_probability", agg)
            self.assertIn("mean_probability", agg)
            self.assertIn("supporting_tiles", agg)
            self.assertIn("total_tiles", agg)
            self.assertIn("coverage_fraction", agg)

    def test_G_supporting_tile_counts_never_exceed_total_tile_count(self):
        """G. Supporting tile counts never exceed total tile count."""
        raster = create_synthetic_sentinel2_test_data(250, 300)
        result = self.tiler.run_tile_inference(raster)

        total_tiles = result["tile_count"]
        self.assertEqual(total_tiles, 9)

        for cls_name, agg in result["class_aggregates"].items():
            supp = agg["supporting_tiles"]
            cov = agg["coverage_fraction"]
            self.assertGreaterEqual(supp, 0)
            self.assertLessEqual(supp, total_tiles)
            self.assertGreaterEqual(cov, 0.0)
            self.assertLessEqual(cov, 1.0)
            self.assertAlmostEqual(cov, supp / total_tiles, places=4)

    def test_H_no_invented_geolocation_when_crs_transform_absent(self):
        """H. No invented geolocation when CRS/transform is absent."""
        raster = create_synthetic_sentinel2_test_data(240, 240)
        # When geotransform is None, geo_bounds must be None for all tiles
        result = self.tiler.run_tile_inference(raster, geotransform=None, crs=None)

        for tile in result["tiles"]:
            self.assertIsNone(tile["geo_bounds"])
            # Pixel bounds must be strictly pixel coordinates
            pb = tile["pixel_bounds"]
            self.assertIn("x_min", pb)
            self.assertIn("y_min", pb)
            self.assertIn("x_max", pb)
            self.assertIn("y_max", pb)

        self.assertFalse(result["geospatial_metadata"]["has_geotransform"])

    def test_I_real_crs_transform_is_preserved_when_present(self):
        """I. Real CRS/transform is preserved when present."""
        raster = create_synthetic_sentinel2_test_data(240, 240)
        # Affine geotransform: [origin_x, pixel_w, rot_x, origin_y, rot_y, pixel_h]
        geotransform = [281000.0, 10.0, 0.0, 2102000.0, 0.0, -10.0]
        crs = "EPSG:32643"

        result = self.tiler.run_tile_inference(raster, geotransform=geotransform, crs=crs)
        self.assertTrue(result["geospatial_metadata"]["has_geotransform"])
        self.assertEqual(result["geospatial_metadata"]["crs"], "EPSG:32643")

        # Tile 0: x in [0, 120], y in [0, 120]
        # geo_x = 281000 + x * 10 -> [281000, 282200]
        # geo_y = 2102000 + y * (-10) -> [2100800, 2102000]
        tile_0 = result["tiles"][0]
        self.assertIsNotNone(tile_0["geo_bounds"])
        gb_0 = tile_0["geo_bounds"]
        self.assertEqual(gb_0["minX"], 281000.0)
        self.assertEqual(gb_0["maxX"], 282200.0)
        self.assertEqual(gb_0["minY"], 2100800.0)
        self.assertEqual(gb_0["maxY"], 2102000.0)

    def test_J_rgb_image_remains_rejected(self):
        """J. RGB image remains rejected from BigEarthNet."""
        rgb_data = np.zeros((3, 240, 240), dtype=np.uint8)
        res = bigearthnet_service.predict(
            image_source=rgb_data,
            bands=["Red", "Green", "Blue"],
            query="Analyze land cover",
        )
        self.assertFalse(res["is_ai_prediction"])
        self.assertIn("bypassed", res["answer"].lower())
        self.assertNotIn("tiles", res)

    def test_K_service_and_agent_end_to_end_tile_pipeline(self):
        """
        Integration test: BigEarthNet service and SatQuery agent run end-to-end
        with real tile inference and produce formatted execution trace.
        """
        raster_240 = create_synthetic_sentinel2_test_data(240, 240)
        geotransform = [281000.0, 10.0, 0.0, 2102000.0, 0.0, -10.0]
        crs = "EPSG:32643"

        # 1. Direct Service Call
        service_res = bigearthnet_service.predict(
            image_source=raster_240,
            bands=REQUIRED_S2_BANDS,
            query="Analyze land cover across spatial windows",
            metadata={"geotransform": geotransform, "crs": crs, "filename": "s2_test_data_240.tif"},
        )
        self.assertTrue(service_res["is_ai_prediction"])
        self.assertEqual(service_res["tile_count"], 4)
        self.assertEqual(len(service_res["tiles"]), 4)
        self.assertIn("across 4 spatial windows", service_res["answer"])

        # 2. Agent Orchestrator Call
        pipeline_res = self.agent.execute_pipeline(
            query="What is the land cover distribution across this scene?",
            mode="single",
            images_dict={
                "single": {
                    "name": "s2_test_data_240.tif",
                    "array": raster_240,
                    "bands": REQUIRED_S2_BANDS,
                }
            },
        )
        self.assertEqual(pipeline_res.selectedModel, "BigEarthNet ResNet-18 Land-Cover Specialist")
        self.assertFalse(pipeline_res.isSimulation)

        # Verify Stage 6 execution trace details
        step6 = next((s for s in pipeline_res.executionSteps if s.id == "step-6"), None)
        self.assertIsNotNone(step6)
        details_dict = {d["label"]: d["value"] for d in step6.details}

        self.assertEqual(details_dict["Raster Dimensions"], "240 × 240 px")
        self.assertEqual(details_dict["Number of Tiles"], "4")
        self.assertEqual(details_dict["Tile Size"], "120 × 120 px")
        self.assertEqual(details_dict["Stride"], "120 px")
        self.assertEqual(details_dict["Valid Tile Count"], "4 / 4 processed")
        self.assertEqual(details_dict["Model Used"], "BigEarthNet ResNet-18 Land-Cover Specialist")
        self.assertEqual(details_dict["Aggregation Method"], "Maximum tile probability across spatial windows")
        self.assertIn("Completed", details_dict["Inference Completion Status"])


if __name__ == "__main__":
    unittest.main()
