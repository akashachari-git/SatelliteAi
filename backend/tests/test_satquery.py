import unittest
import os
from pathlib import Path

class TestSatQueryBackend(unittest.TestCase):

    def setUp(self):
        from backend.app.demo.demo_generator import DemoDataGenerator
        self.scenarios = DemoDataGenerator.ensure_demo_files()

    def test_demo_files_generated(self):
        from backend.app.config import DEMO_DATA_DIR
        self.assertTrue((DEMO_DATA_DIR / "urban_satellite_eo.tif").exists())
        self.assertTrue((DEMO_DATA_DIR / "water_reservoir_eo.tif").exists())
        self.assertTrue((DEMO_DATA_DIR / "temporal_t1_2021.tif").exists())
        self.assertTrue((DEMO_DATA_DIR / "temporal_t2_2023.tif").exists())
        self.assertTrue((DEMO_DATA_DIR / "crossmodal_optical.tif").exists())
        self.assertTrue((DEMO_DATA_DIR / "crossmodal_sar_s1.tif").exists())

    def test_metadata_extraction_geotiff(self):
        from backend.app.remote_sensing.metadata_extractor import RemoteSensingMetadataExtractor
        from backend.app.config import DEMO_DATA_DIR
        meta = RemoteSensingMetadataExtractor.extract_metadata(str(DEMO_DATA_DIR / "urban_satellite_eo.tif"))
        self.assertEqual(meta["width"], 512)
        self.assertEqual(meta["height"], 512)
        self.assertTrue(meta["is_geotiff"])
        self.assertIn("EPSG", meta["crs"])
        self.assertEqual(meta["spatial_resolution_m"], 10.0)

    def test_query_routing(self):
        from backend.app.agents.query_classifier import QueryClassifier
        
        # Grounding
        res1 = QueryClassifier.classify("Highlight the water body", 1, ["Optical"])
        self.assertEqual(res1["task"], "GROUNDING")

        # Captioning
        res2 = QueryClassifier.classify("Describe this satellite image", 1, ["Optical"])
        self.assertEqual(res2["task"], "CAPTIONING")

        # Change analysis
        res3 = QueryClassifier.classify("What changed between these two dates?", 2, ["Optical", "Optical"])
        self.assertEqual(res3["task"], "CHANGE_ANALYSIS")

        # Change VQA
        res4 = QueryClassifier.classify("Has the built-up area increased?", 2, ["Optical", "Optical"])
        self.assertEqual(res4["task"], "CHANGE_VQA")

        # Optical + SAR
        res5 = QueryClassifier.classify("Identify built-up and water-covered regions using both images", 2, ["Optical", "SAR"])
        self.assertEqual(res5["task"], "OPTICAL_SAR_ANALYSIS")

    def test_coregistration_checker(self):
        from backend.app.remote_sensing.co_registration import CoRegistrationChecker
        from backend.app.remote_sensing.metadata_extractor import RemoteSensingMetadataExtractor
        from backend.app.config import DEMO_DATA_DIR

        meta1 = RemoteSensingMetadataExtractor.extract_metadata(str(DEMO_DATA_DIR / "temporal_t1_2021.tif"))
        meta2 = RemoteSensingMetadataExtractor.extract_metadata(str(DEMO_DATA_DIR / "temporal_t2_2023.tif"))
        eval_res = CoRegistrationChecker.evaluate_pair(meta1, meta2, pair_mode="BI_TEMPORAL")
        self.assertTrue(eval_res["is_compatible"])
        self.assertGreaterEqual(eval_res["co_registration_score"], 80)

    def test_agent_controller_single_vqa(self):
        from backend.app.agents.agent_controller import AgentController
        from backend.app.config import DEMO_DATA_DIR
        controller = AgentController()
        img_path = str(DEMO_DATA_DIR / "urban_satellite_eo.tif")
        res = controller.process_analysis_request(
            query="Describe the major land-cover types visible in this image",
            image_paths=[img_path]
        )
        self.assertTrue(res["success"])
        self.assertIn(res["task"], ["SINGLE_VQA", "CAPTIONING", "LAND_COVER_ANALYSIS"])
        self.assertGreater(len(res["execution_trace"]), 3)
        self.assertGreaterEqual(res["confidence"]["score"], 0.70)

    def test_agent_controller_grounding(self):
        from backend.app.agents.agent_controller import AgentController
        from backend.app.config import DEMO_DATA_DIR
        controller = AgentController()
        img_path = str(DEMO_DATA_DIR / "water_reservoir_eo.tif")
        res = controller.process_analysis_request(
            query="Highlight the water body",
            image_paths=[img_path]
        )
        self.assertTrue(res["success"])
        self.assertIn(res["task"], ["GROUNDING", "WATER_DETECTION"])
        self.assertTrue(any(ev["type"] == "mask_overlay" for ev in res["evidence"]))

    def test_agent_controller_bitemporal_change(self):
        from backend.app.agents.agent_controller import AgentController
        from backend.app.config import DEMO_DATA_DIR
        controller = AgentController()
        p1 = str(DEMO_DATA_DIR / "temporal_t1_2021.tif")
        p2 = str(DEMO_DATA_DIR / "temporal_t2_2023.tif")
        res = controller.process_analysis_request(
            query="What changed between these two dates?",
            image_paths=[p1, p2]
        )
        self.assertTrue(res["success"])
        self.assertIn(res["task"], ["CHANGE_ANALYSIS", "CHANGE_DETECTION"])
        self.assertTrue(any(ev["type"] == "mask_overlay" or ev["type"] == "change_heatmap" for ev in res["evidence"]))

    def test_incompatible_input_rejection(self):
        from backend.app.agents.agent_controller import AgentController
        from backend.app.config import DEMO_DATA_DIR
        controller = AgentController()
        # Request change detection on only 1 image!
        p1 = str(DEMO_DATA_DIR / "urban_satellite_eo.tif")
        res = controller.process_analysis_request(
            query="What changed between these two dates?",
            image_paths=[p1]
        )
        self.assertFalse(res["success"])
        self.assertIn("requires two", res["answer"])

    def test_pdf_report_generation(self):
        from backend.app.agents.agent_controller import AgentController
        from backend.app.reports.report_service import ReportService
        from backend.app.config import DEMO_DATA_DIR, BASE_DIR
        controller = AgentController()
        img_path = str(DEMO_DATA_DIR / "water_reservoir_eo.tif")
        analysis = controller.process_analysis_request(
            query="Highlight the water body",
            image_paths=[img_path]
        )
        pdf_url = ReportService.generate_pdf_report(analysis)
        self.assertTrue(pdf_url.endswith(".pdf"))
        disk_path = BASE_DIR / pdf_url.lstrip("/")
        self.assertTrue(disk_path.exists())
        self.assertGreater(disk_path.stat().st_size, 1000)

    def test_satellite_image_validation(self):
        from backend.app.remote_sensing.satellite_validator import SatelliteImageValidator
        from backend.app.remote_sensing.metadata_extractor import RemoteSensingMetadataExtractor
        from backend.app.config import DEMO_DATA_DIR

        sat_path = str(DEMO_DATA_DIR / "urban_satellite_eo.tif")
        meta = RemoteSensingMetadataExtractor.extract_metadata(sat_path)
        is_sat, reason, conf = SatelliteImageValidator.validate_satellite_image(sat_path, meta)
        self.assertTrue(is_sat)
        self.assertGreaterEqual(conf, 0.80)

    def test_non_satellite_image_rejection(self):
        from backend.app.remote_sensing.satellite_validator import SatelliteImageValidator
        from backend.app.config import STORAGE_DIR
        import numpy as np
        from PIL import Image

        # Create a synthetic portrait photo with human skin tones (RGB ~ [220, 160, 130])
        non_sat_path = STORAGE_DIR / "test_fake_portrait.jpg"
        arr = np.ones((256, 256, 3), dtype=np.uint8)
        arr[:, :, 0] = 225  # R
        arr[:, :, 1] = 165  # G
        arr[:, :, 2] = 135  # B
        Image.fromarray(arr).save(str(non_sat_path))

        meta = {"is_geotiff": False, "filename": "test_fake_portrait.jpg"}
        is_sat, reason, conf = SatelliteImageValidator.validate_satellite_image(str(non_sat_path), meta)
        
        # Cleanup
        if non_sat_path.exists():
            non_sat_path.unlink()

        self.assertFalse(is_sat)
        self.assertEqual(reason, SatelliteImageValidator.REJECTION_CLEAR)

    def test_non_satellite_categories_rejection(self):
        """Tests that documents, food, cars/objects, and terrestrial photos are rejected with exact message."""
        from backend.app.remote_sensing.satellite_validator import SatelliteImageValidator
        from backend.app.config import STORAGE_DIR
        import numpy as np
        from PIL import Image

        # 1. Document / text screenshot (predominantly white with text lines)
        doc_path = STORAGE_DIR / "test_doc.jpg"
        arr = np.ones((256, 256, 3), dtype=np.uint8) * 250
        arr[50:60, 30:220] = 10  # text line
        arr[80:90, 30:220] = 10  # text line
        Image.fromarray(arr).save(str(doc_path))
        is_sat, reason, _ = SatelliteImageValidator.validate_satellite_image(str(doc_path))
        if doc_path.exists():
            doc_path.unlink()
        self.assertFalse(is_sat)
        self.assertEqual(reason, SatelliteImageValidator.REJECTION_CLEAR)

        # 2. Food / macro plate (warm food centrally isolated on dark plate)
        food_path = STORAGE_DIR / "test_food.jpg"
        arr = np.ones((256, 256, 3), dtype=np.uint8) * 40
        # Warm food in center
        arr[70:180, 70:180, 0] = 220
        arr[70:180, 70:180, 1] = 130
        arr[70:180, 70:180, 2] = 30
        Image.fromarray(arr).save(str(food_path))
        is_sat, reason, _ = SatelliteImageValidator.validate_satellite_image(str(food_path))
        if food_path.exists():
            food_path.unlink()
        self.assertFalse(is_sat)
        self.assertEqual(reason, SatelliteImageValidator.REJECTION_CLEAR)

    def test_uncertain_image_rejection(self):
        """Tests that blurry/ambiguous low-information images return the uncertain message."""
        from backend.app.remote_sensing.satellite_validator import SatelliteImageValidator
        from backend.app.config import STORAGE_DIR
        import numpy as np
        from PIL import Image

        # Uniform low-gradient blurry noise (unidentifiable)
        blur_path = STORAGE_DIR / "test_blurry.jpg"
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            arr[i, :, :] = int(120 + 8 * np.sin(i / 30.0))
        Image.fromarray(arr).save(str(blur_path))

        is_sat, reason, _ = SatelliteImageValidator.validate_satellite_image(str(blur_path))
        if blur_path.exists():
            blur_path.unlink()
        self.assertFalse(is_sat)
        self.assertEqual(reason, SatelliteImageValidator.REJECTION_UNCERTAIN)

    def test_pipeline_bypass_prevention(self):
        """Verifies that asking any question with a non-satellite image is rejected at Stage 0 without running VQA/captioning."""
        from backend.app.agents.agent_controller import AgentController
        from backend.app.remote_sensing.satellite_validator import SatelliteImageValidator
        from backend.app.config import STORAGE_DIR
        import numpy as np
        from PIL import Image

        # Create synthetic portrait
        portrait_path = STORAGE_DIR / "test_bypass_portrait.jpg"
        arr = np.ones((256, 256, 3), dtype=np.uint8)
        arr[:, :, 0] = 225
        arr[:, :, 1] = 165
        arr[:, :, 2] = 135
        Image.fromarray(arr).save(str(portrait_path))

        controller = AgentController()
        # Attempt various queries that would normally route to different features
        for query in ["Detect buildings in this image", "Highlight water body", "Describe this scene", "Is there vegetation?"]:
            res = controller.process_analysis_request(query=query, image_paths=[str(portrait_path)])
            self.assertFalse(res["success"])
            self.assertEqual(res["task"], "REJECTED_NON_SATELLITE")
            self.assertEqual(res["answer"], SatelliteImageValidator.REJECTION_CLEAR)
            self.assertEqual(res["execution_trace"][0]["stage_id"], "STAGE_0_SATELLITE_VALIDATION")
            self.assertEqual(res["execution_trace"][0]["status"], "FAILED")

        if portrait_path.exists():
            portrait_path.unlink()

    def test_auth_session_lifecycle(self):
        from backend.app.auth.auth_service import AuthService
        fake_user = {
            "id": "10987654321",
            "sub": "10987654321",
            "email": "satellite.researcher@example.com",
            "name": "Dr. EO Researcher",
            "picture": "https://lh3.googleusercontent.com/a/default-user",
            "auth_provider": "google",
            "verified_at": 1700000000
        }
        # 1. Create session
        token = AuthService.create_session(fake_user)
        self.assertIsNotNone(token)
        self.assertGreater(len(token), 20)

        # 2. Retrieve session
        user = AuthService.get_session_user(token)
        self.assertIsNotNone(user)
        self.assertEqual(user["sub"], "10987654321")
        self.assertEqual(user["email"], "satellite.researcher@example.com")

        # 3. Destroy session
        destroyed = AuthService.destroy_session(token)
        self.assertTrue(destroyed)

        # 4. Verify gone
        gone_user = AuthService.get_session_user(token)
        self.assertIsNone(gone_user)

    def test_invalid_google_token_rejection(self):
        from backend.app.auth.auth_service import AuthService
        # Testing with empty token
        with self.assertRaises(ValueError):
            AuthService.verify_google_id_token("")

        # Testing with fake / forged token
        with self.assertRaises(ValueError):
            AuthService.verify_google_id_token("fake.forged.jwt_token")

    def test_non_satellite_terrestrial_sky_rejection(self):
        from backend.app.remote_sensing.satellite_validator import SatelliteImageValidator
        from backend.app.config import STORAGE_DIR
        import numpy as np
        from PIL import Image

        # Create a ground photo with blue sky on top (top third sky blue: [100, 180, 245]) and ground below
        sky_path = STORAGE_DIR / "test_terrestrial_sky.jpg"
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        # Sky
        arr[:85, :, 0] = 90
        arr[:85, :, 1] = 170
        arr[:85, :, 2] = 240
        # Ground
        arr[85:, :, 0] = 50
        arr[85:, :, 1] = 70
        arr[85:, :, 2] = 30
        Image.fromarray(arr).save(str(sky_path))

        meta = {"is_geotiff": False, "filename": "terrestrial_sky.jpg"}
        is_sat, reason, conf = SatelliteImageValidator.validate_satellite_image(str(sky_path), meta)

        if sky_path.exists():
            sky_path.unlink()

        self.assertFalse(is_sat)
        self.assertEqual(reason, SatelliteImageValidator.REJECTION_CLEAR)

if __name__ == "__main__":
    unittest.main()

