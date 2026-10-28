"""
test_db.py
==========
Automated unit tests for SQLite database operations.
Verifies table creation, insertions, queries, and empty result resilience.
"""

import os
import tempfile
import unittest
from src.database import (
    init_db,
    log_traffic_observation,
    log_signal_recommendation,
    get_traffic_history,
    get_signal_history,
    get_summary_statistics,
    clear_database,
)


class TestDatabase(unittest.TestCase):

    def setUp(self):
        """Creates a temporary SQLite database file for isolation."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        init_db(self.db_path)

    def tearDown(self):
        """Cleans up the temporary database file after tests."""
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_empty_database_returns_valid_structures(self):
        """Fetching from an empty database should return empty DataFrames without crashing."""
        traffic_df = get_traffic_history(self.db_path)
        signal_df = get_signal_history(self.db_path)
        stats = get_summary_statistics(self.db_path)

        self.assertTrue(traffic_df.empty)
        self.assertTrue(signal_df.empty)
        self.assertEqual(stats["total_records"], 0)
        self.assertEqual(stats["emergency_events"], 0)

    def test_log_and_fetch_traffic_observation(self):
        """Should insert and retrieve vehicle observations accurately."""
        log_traffic_observation(
            db_path=self.db_path,
            video_source="test_video.mp4",
            frame_number=42,
            lane_id=2,
            active_count=5,
            cumulative_crossed=12,
            breakdown={"car": 3, "truck": 2},
        )

        df = get_traffic_history(self.db_path, limit=10)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["lane_id"], 2)
        self.assertEqual(row["active_count"], 5)
        self.assertEqual(row["cumulative_crossed"], 12)
        self.assertEqual(row["cars"], 3)
        self.assertEqual(row["trucks"], 2)

    def test_log_and_fetch_signal_recommendation(self):
        """Should insert and retrieve signal decisions and emergency simulation flags."""
        log_signal_recommendation(
            db_path=self.db_path,
            recommended_lane=3,
            green_duration_sec=35.0,
            is_emergency=True,
            reason="Simulated Ambulance clearance",
            lane_densities={1: 2, 2: 3, 3: 1},
        )

        df = get_signal_history(self.db_path, limit=10)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["recommended_lane"], 3)
        self.assertEqual(row["green_duration_sec"], 35.0)
        self.assertEqual(row["is_emergency"], 1)

    def test_clear_database(self):
        """Clearing the database should empty all tables."""
        log_traffic_observation(
            db_path=self.db_path,
            video_source="test.mp4",
            frame_number=1,
            lane_id=1,
            active_count=2,
            cumulative_crossed=2,
        )
        clear_database(self.db_path)
        df = get_traffic_history(self.db_path)
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
