"""
test_signal.py
==============
Automated unit tests for TrafficSignalController.
Verifies dynamic green-time calculations, starvation prevention,
and simulated emergency priority preemption.
"""

import unittest
from src.signal_controller import TrafficSignalController


class TestTrafficSignalController(unittest.TestCase):

    def setUp(self):
        """Initializes a fresh controller before each test."""
        self.controller = TrafficSignalController(
            num_lanes=3,
            min_green_sec=10.0,
            max_green_sec=45.0,
            base_green_sec=10.0,
            sec_per_vehicle=2.5,
            yellow_duration_sec=3.0,
            all_red_duration_sec=1.0,
            starvation_threshold=2,
        )

    def test_green_time_clamping_minimum(self):
        """Zero vehicles should receive the minimum green time clamp (10.0s)."""
        duration = self.controller.calculate_green_time(0)
        self.assertEqual(duration, 10.0)

    def test_green_time_scaling(self):
        """4 vehicles should receive base (10) + 4 * 2.5 = 20.0s."""
        duration = self.controller.calculate_green_time(4)
        self.assertEqual(duration, 20.0)

    def test_green_time_clamping_maximum(self):
        """Very heavy traffic (e.g. 50 vehicles) should be capped at max_green_sec (45.0s)."""
        duration = self.controller.calculate_green_time(50)
        self.assertEqual(duration, 45.0)

    def test_high_density_lane_selection(self):
        """The controller should pick the non-active lane with the highest density."""
        self.controller.active_lane = 1
        densities = {1: 2, 2: 8, 3: 1}  # Lane 2 has highest traffic

        next_lane, green_time, is_em, reason = self.controller.select_next_lane(densities)
        self.assertEqual(next_lane, 2)
        self.assertFalse(is_em)
        self.assertIn("Lane 2", reason)

    def test_starvation_prevention(self):
        """A lane waiting for >= starvation_threshold cycles should be guaranteed green."""
        self.controller.active_lane = 1
        self.controller.wait_cycles = {1: 0, 2: 1, 3: 3}  # Lane 3 is starved
        densities = {1: 1, 2: 10, 3: 1}  # Lane 2 has more traffic, but Lane 3 is starved

        next_lane, green_time, is_em, reason = self.controller.select_next_lane(densities)
        self.assertEqual(next_lane, 3)
        self.assertIn("Fairness guarantee", reason)

    def test_emergency_simulation_priority(self):
        """Emergency override must preempt normal rotation and grant priority green time."""
        densities = {1: 10, 2: 12, 3: 1}
        emergency_lane = 3  # Lane 3 has an emergency simulation

        next_lane, green_time, is_em, reason = self.controller.select_next_lane(
            densities, emergency_lane=emergency_lane
        )
        self.assertEqual(next_lane, 3)
        self.assertTrue(is_em)
        self.assertEqual(green_time, self.controller.max_green_sec)
        self.assertIn("SIMULATION ONLY", reason)


if __name__ == "__main__":
    unittest.main()
