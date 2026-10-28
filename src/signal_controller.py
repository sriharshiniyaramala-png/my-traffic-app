"""
signal_controller.py
====================
Adaptive Traffic Signal Controller and Emergency Priority Logic.

Features:
1. Dynamic Green Time calculation based on lane traffic density.
2. Configurable minimum and maximum green-time limits.
3. Fair round-robin rotation with starvation prevention.
4. Clearly labeled Emergency Priority Simulation Mode (Safe, Educational Simulation Only).
"""

from typing import Dict, Tuple, Optional, Any


class TrafficSignalController:
    """
    Simulates an intelligent, density-adaptive traffic signal controller
    with emergency preemption for a multi-lane intersection.

    IMPORTANT SAFETY DISCLAIMER:
    This controller is for educational simulation and academic research only.
    It does NOT connect to or control real-world traffic signal hardware.
    """

    def __init__(
        self,
        num_lanes: int = 3,
        min_green_sec: float = 10.0,
        max_green_sec: float = 45.0,
        base_green_sec: float = 10.0,
        sec_per_vehicle: float = 2.5,
        yellow_duration_sec: float = 3.0,
        all_red_duration_sec: float = 1.0,
        starvation_threshold: int = 2,
    ):
        """
        Initializes the traffic signal controller parameters.

        Args:
            num_lanes: Total number of controlled lanes (default 3).
            min_green_sec: Minimum green light time to ensure pedestrian & startup clearance.
            max_green_sec: Upper green light limit to prevent excessive waiting on other lanes.
            base_green_sec: Base green time allocated even with light traffic.
            sec_per_vehicle: Additional seconds of green added per detected vehicle.
            yellow_duration_sec: Duration of yellow warning phase before switching.
            all_red_duration_sec: Safety buffer where all lights are red before next green.
            starvation_threshold: Cycles a lane can wait before guaranteed turn.
        """
        self.num_lanes = num_lanes
        self.min_green_sec = float(min_green_sec)
        self.max_green_sec = float(max_green_sec)
        self.base_green_sec = float(base_green_sec)
        self.sec_per_vehicle = float(sec_per_vehicle)
        self.yellow_duration_sec = float(yellow_duration_sec)
        self.all_red_duration_sec = float(all_red_duration_sec)
        self.starvation_threshold = starvation_threshold

        # Active state
        self.active_lane = 1
        self.next_lane = 1
        self.state = "GREEN"  # "GREEN", "YELLOW", "ALL_RED"
        self.allocated_green_sec = self.min_green_sec
        self.time_remaining_sec = self.min_green_sec

        # Starvation tracking: maps lane_id -> cycles waited since last green
        self.wait_cycles: Dict[int, int] = {lane: 0 for lane in range(1, num_lanes + 1)}

        # Emergency simulation tracking
        self.is_emergency_active = False
        self.emergency_lane: Optional[int] = None
        self.emergency_reason: str = ""

    def calculate_green_time(self, vehicle_count: int) -> float:
        """
        Calculates recommended green light duration based on vehicle count.
        Formula: Time = base_green + (count * sec_per_vehicle)
        Clamped within [min_green_sec, max_green_sec].
        """
        count = max(0, vehicle_count)
        raw_duration = self.base_green_sec + (count * self.sec_per_vehicle)
        clamped_duration = max(self.min_green_sec, min(self.max_green_sec, raw_duration))
        return round(clamped_duration, 1)

    def select_next_lane(
        self, lane_densities: Dict[int, int], emergency_lane: Optional[int] = None
    ) -> Tuple[int, float, bool, str]:
        """
        Determines the next lane to receive green light.

        Priority hierarchy:
        1. Emergency Simulation Override (Preempts immediately).
        2. Starvation Prevention (Any lane waiting >= starvation_threshold).
        3. Highest Density Lane.
        4. Fair Round-Robin Rotation (if densities are equal or near zero).

        Returns:
            (recommended_lane, green_duration_sec, is_emergency, reason)
        """
        # 1. Emergency Preemption Check (SIMULATION ONLY)
        if emergency_lane is not None and 1 <= emergency_lane <= self.num_lanes:
            green_time = self.max_green_sec
            reason = f"[SIMULATION ONLY] Emergency vehicle priority override granted for Lane {emergency_lane}."
            return emergency_lane, green_time, True, reason

        # 2. Check for Starved Lanes
        # Exclude currently active lane from starved candidates
        starved_lanes = [
            lane for lane, cycles in self.wait_cycles.items()
            if cycles >= self.starvation_threshold and lane != self.active_lane
        ]
        if starved_lanes:
            # Pick the starved lane with the highest count (or first in order)
            chosen_lane = max(starved_lanes, key=lambda l: lane_densities.get(l, 0))
            count = lane_densities.get(chosen_lane, 0)
            green_time = self.calculate_green_time(count)
            reason = f"Fairness guarantee: Lane {chosen_lane} was waiting {self.wait_cycles[chosen_lane]} cycles."
            return chosen_lane, green_time, False, reason

        # 3. Density-Based Selection
        # Exclude active lane if others have significant traffic
        other_lanes = [l for l in range(1, self.num_lanes + 1) if l != self.active_lane]
        max_density = -1
        candidate_lane = None

        for lane in other_lanes:
            density = lane_densities.get(lane, 0)
            if density > max_density:
                max_density = density
                candidate_lane = lane

        active_lane_density = lane_densities.get(self.active_lane, 0)

        # If highest candidate lane has higher traffic than currently active lane, switch to it
        if candidate_lane is not None and max_density > 0:
            count = max_density
            green_time = self.calculate_green_time(count)
            reason = f"High density demand: Lane {candidate_lane} has {count} detected vehicles."
            return candidate_lane, green_time, False, reason

        # 4. Fallback: Standard Fair Round-Robin
        next_rr_lane = (self.active_lane % self.num_lanes) + 1
        count = lane_densities.get(next_rr_lane, 0)
        green_time = self.calculate_green_time(count)
        reason = f"Normal round-robin rotation to Lane {next_rr_lane}."
        return next_rr_lane, green_time, False, reason

    def step(
        self,
        dt_sec: float,
        lane_densities: Dict[int, int],
        emergency_lane: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Advances the traffic signal simulation by dt_sec seconds.
        Manages light transitions (GREEN -> YELLOW -> ALL_RED -> next GREEN)
        and responds to emergency simulation events.

        Returns a dictionary representing the current traffic light state.
        """
        # Handle emergency trigger
        if emergency_lane is not None and not self.is_emergency_active:
            self.is_emergency_active = True
            self.emergency_lane = emergency_lane
            self.emergency_reason = f"[SIMULATION] Priority clearance for Lane {emergency_lane}"

            # If emergency lane is NOT currently green, immediately cut to Yellow
            if self.active_lane != emergency_lane and self.state == "GREEN":
                self.state = "YELLOW"
                self.time_remaining_sec = min(self.time_remaining_sec, self.yellow_duration_sec)
                self.next_lane = emergency_lane

        elif emergency_lane is None and self.is_emergency_active:
            # Emergency cleared
            self.is_emergency_active = False
            self.emergency_lane = None

        # Decrement timer
        self.time_remaining_sec -= dt_sec

        decision_made = False
        decision_info = None

        # State transition logic
        if self.time_remaining_sec <= 0:
            if self.state == "GREEN":
                # GREEN -> YELLOW
                self.state = "YELLOW"
                self.time_remaining_sec = self.yellow_duration_sec

                # Plan next lane
                nxt, g_time, is_em, rsn = self.select_next_lane(
                    lane_densities, emergency_lane=self.emergency_lane
                )
                self.next_lane = nxt
                self.allocated_green_sec = g_time

                decision_made = True
                decision_info = {
                    "recommended_lane": nxt,
                    "green_duration_sec": g_time,
                    "is_emergency": is_em,
                    "reason": rsn,
                }

            elif self.state == "YELLOW":
                # YELLOW -> ALL_RED (Safety clearance)
                self.state = "ALL_RED"
                self.time_remaining_sec = self.all_red_duration_sec

            elif self.state == "ALL_RED":
                # ALL_RED -> GREEN for next lane
                self.active_lane = self.next_lane
                self.state = "GREEN"
                self.time_remaining_sec = self.allocated_green_sec

                # Update wait cycles
                for lane in range(1, self.num_lanes + 1):
                    if lane == self.active_lane:
                        self.wait_cycles[lane] = 0
                    else:
                        self.wait_cycles[lane] += 1

        # Build lane-by-lane signal light status
        lane_lights = {}
        for lane in range(1, self.num_lanes + 1):
            if self.state == "ALL_RED":
                lane_lights[lane] = "RED"
            elif lane == self.active_lane:
                lane_lights[lane] = self.state  # GREEN or YELLOW
            else:
                lane_lights[lane] = "RED"

        return {
            "active_lane": self.active_lane,
            "next_lane": self.next_lane,
            "state": self.state,
            "time_remaining_sec": max(0.0, round(self.time_remaining_sec, 1)),
            "allocated_green_sec": self.allocated_green_sec,
            "lane_lights": lane_lights,
            "is_emergency": self.is_emergency_active,
            "emergency_lane": self.emergency_lane,
            "decision_made": decision_made,
            "decision_info": decision_info,
        }
