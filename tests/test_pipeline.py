"""
test_pipeline.py
================
Integration test that processes sample video frames through detection,
tracking, signal control, and SQLite logging to ensure end-to-end stability.
"""

import os
import unittest
import cv2
from src.detector import VehicleDetector
from src.tracker import CentroidTracker
from src.signal_controller import TrafficSignalController
from src.database import init_db, log_traffic_observation, log_signal_recommendation, get_traffic_history


class TestPipeline(unittest.TestCase):

    def test_end_to_end_pipeline(self):
        video_path = "data/sample_traffic.mp4"
        self.assertTrue(os.path.exists(video_path), "Sample video must exist.")

        cap = cv2.VideoCapture(video_path)
        self.assertTrue(cap.isOpened(), "VideoCapture must open sample video.")

        # Initialize pipeline components
        detector = VehicleDetector(model_name="yolov8n.pt", conf_threshold=0.25)
        tracker = CentroidTracker(num_lanes=3)
        controller = TrafficSignalController(num_lanes=3)
        test_db = "data/test_pipeline.db"
        init_db(test_db)

        frames_processed = 0
        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # 1. Detect
            detections = detector.detect(frame)

            # 2. Track
            tracking_info = tracker.update(detections, w, h)

            # 3. Signal control step (dt = 0.04s for 25fps)
            sig_state = controller.step(
                dt_sec=0.04,
                lane_densities=tracking_info["lane_densities"],
                emergency_lane=None,
            )

            # 4. Annotate
            annotated = detector.annotate_frame(frame, tracking_info, sig_state)
            self.assertEqual(annotated.shape, frame.shape)

            # 5. Database logging
            for lane_id, count in tracking_info["lane_densities"].items():
                log_traffic_observation(
                    db_path=test_db,
                    video_source=video_path,
                    frame_number=frames_processed,
                    lane_id=lane_id,
                    active_count=count,
                    cumulative_crossed=tracking_info["lane_crossed"].get(lane_id, 0),
                )

            frames_processed += 1

        cap.release()
        self.assertGreater(frames_processed, 0)

        # Verify DB records
        history = get_traffic_history(test_db, limit=50)
        self.assertFalse(history.empty)

        # Cleanup test DB
        if os.path.exists(test_db):
            try:
                os.remove(test_db)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
