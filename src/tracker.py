"""
tracker.py
==========
Centroid-Based Multi-Object Tracker with Lane Assignment and Line-Crossing Counter.

Key Features:
1. Centroid Euclidean Distance Matching: Tracks vehicles across consecutive frames.
2. Persistent Vehicle IDs: Prevents counting the same physical vehicle multiple times.
3. Virtual Counting Line: Increments the cumulative count only once when a vehicle crosses the line.
4. Configurable Lanes: Partitions video coordinates into distinct traffic lanes to estimate density.
"""

import math
from typing import Dict, List, Tuple, Set, Optional, Any


class CentroidTracker:
    """
    Beginner-friendly vehicle tracker based on centroid Euclidean distance.
    Maintains active track IDs, tracks trajectory, and counts vehicles crossing a virtual tripwire.
    """

    def __init__(
        self,
        max_disappeared_frames: int = 25,
        max_distance_px: float = 80.0,
        num_lanes: int = 3,
        lane_boundaries: Optional[List[float]] = None,
        counting_line_ratio: float = 0.65,
    ):
        """
        Args:
            max_disappeared_frames: How many frames an object can be missing before being deregistered.
            max_distance_px: Max distance between frame centroids to consider them the same object.
            num_lanes: Number of lanes to divide the frame into.
            lane_boundaries: Fractions of frame width representing lane dividers, e.g. [0.33, 0.67].
            counting_line_ratio: Vertical fraction of frame height where counting line is placed (0.0 to 1.0).
        """
        self.next_object_id = 1
        self.max_disappeared_frames = max_disappeared_frames
        self.max_distance_px = max_distance_px
        self.num_lanes = num_lanes

        # Default lane dividers if not provided: equal vertical partitions
        if lane_boundaries is None:
            self.lane_boundaries = [i / num_lanes for i in range(1, num_lanes)]
        else:
            self.lane_boundaries = lane_boundaries

        self.counting_line_ratio = counting_line_ratio

        # State dictionaries
        # object_id -> (cx, cy)
        self.objects: Dict[int, Tuple[int, int]] = {}
        # object_id -> vehicle_class (e.g. "car", "truck")
        self.object_classes: Dict[int, str] = {}
        # object_id -> bounding_box (x1, y1, x2, y2)
        self.object_boxes: Dict[int, Tuple[int, int, int, int]] = {}
        # object_id -> number of consecutive frames missing
        self.disappeared: Dict[int, int] = {}
        # object_id -> previous cy position (to detect line crossing)
        self.prev_positions: Dict[int, Tuple[int, int]] = {}

        # Set of object IDs that have already crossed the line (prevents duplicate counts)
        self.crossed_object_ids: Set[int] = set()

        # Cumulative crossed counts per lane: lane_id -> total count
        self.lane_crossed_counts: Dict[int, int] = {lane: 0 for lane in range(1, num_lanes + 1)}

    def get_lane_for_x(self, x: int, frame_width: int) -> int:
        """
        Determines which lane an x-coordinate falls into.
        Lanes are 1-indexed (1, 2, ..., num_lanes).
        """
        if frame_width <= 0:
            return 1
        rel_x = x / frame_width
        for idx, boundary in enumerate(self.lane_boundaries):
            if rel_x < boundary:
                return idx + 1
        return self.num_lanes

    def register(
        self,
        centroid: Tuple[int, int],
        class_name: str,
        box: Tuple[int, int, int, int],
    ) -> int:
        """Registers a newly detected object with a fresh ID."""
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.prev_positions[obj_id] = centroid
        self.object_classes[obj_id] = class_name
        self.object_boxes[obj_id] = box
        self.disappeared[obj_id] = 0
        self.next_object_id += 1
        return obj_id

    def deregister(self, obj_id: int) -> None:
        """Removes an object that has left the camera field of view."""
        self.objects.pop(obj_id, None)
        self.prev_positions.pop(obj_id, None)
        self.object_classes.pop(obj_id, None)
        self.object_boxes.pop(obj_id, None)
        self.disappeared.pop(obj_id, None)

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, Any]:
        """
        Updates the tracker with detections from the current frame.

        Args:
            detections: List of dicts, each with 'box' [x1, y1, x2, y2], 'class_name', 'confidence'.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            Dictionary containing:
            - 'active_tracks': list of tracked objects with IDs, boxes, classes, centroids, lanes
            - 'lane_densities': current number of active vehicles in each lane
            - 'lane_crossed': cumulative number of unique vehicles that crossed the counting line
            - 'counting_line_y': y-coordinate of the counting line
            - 'lane_divider_xs': list of x-coordinates for visual lane dividers
        """
        counting_line_y = int(frame_height * self.counting_line_ratio)

        # If no detections in this frame, mark all currently tracked objects as disappeared
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared_frames:
                    self.deregister(obj_id)

            return self._build_result_summary(frame_width, frame_height, counting_line_y)

        # Compute centroids for current detections
        input_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))

        # If currently tracking no objects, register all detections
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                det = detections[i]
                self.register(centroid, det["class_name"], tuple(det["box"]))
        else:
            # Pair existing objects with new detections using minimum Euclidean distance
            object_ids = list(self.objects.keys())
            existing_centroids = list(self.objects.values())

            # Compute distance matrix: (len(existing) x len(new))
            distances = []
            for e_idx, e_pt in enumerate(existing_centroids):
                row = []
                for n_idx, n_pt in enumerate(input_centroids):
                    dist = math.hypot(e_pt[0] - n_pt[0], e_pt[1] - n_pt[1])
                    row.append(dist)
                distances.append(row)

            # Greedy matching based on lowest distance
            used_existing = set()
            used_new = set()

            # Flatten and sort all (distance, existing_idx, new_idx) tuples
            flat_pairs = []
            for e_idx in range(len(existing_centroids)):
                for n_idx in range(len(input_centroids)):
                    flat_pairs.append((distances[e_idx][n_idx], e_idx, n_idx))
            flat_pairs.sort(key=lambda item: item[0])

            for dist, e_idx, n_idx in flat_pairs:
                if e_idx in used_existing or n_idx in used_new:
                    continue
                if dist > self.max_distance_px:
                    continue

                obj_id = object_ids[e_idx]
                new_centroid = input_centroids[n_idx]
                det = detections[n_idx]

                # Update tracked object position
                self.prev_positions[obj_id] = self.objects[obj_id]
                self.objects[obj_id] = new_centroid
                self.object_boxes[obj_id] = tuple(det["box"])
                self.object_classes[obj_id] = det["class_name"]
                self.disappeared[obj_id] = 0

                used_existing.add(e_idx)
                used_new.add(n_idx)

            # Unmatched existing objects mark as disappeared
            for e_idx, obj_id in enumerate(object_ids):
                if e_idx not in used_existing:
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared_frames:
                        self.deregister(obj_id)

            # Unmatched new detections are registered as new objects
            for n_idx, centroid in enumerate(input_centroids):
                if n_idx not in used_new:
                    det = detections[n_idx]
                    self.register(centroid, det["class_name"], tuple(det["box"]))

        # Check for counting line crossings
        for obj_id, current_pt in self.objects.items():
            if obj_id in self.crossed_object_ids:
                continue

            prev_pt = self.prev_positions.get(obj_id, current_pt)
            prev_y = prev_pt[1]
            curr_y = current_pt[1]

            # Downward motion crossing the line
            if prev_y < counting_line_y <= curr_y:
                self.crossed_object_ids.add(obj_id)
                crossing_lane = self.get_lane_for_x(current_pt[0], frame_width)
                self.lane_crossed_counts[crossing_lane] = (
                    self.lane_crossed_counts.get(crossing_lane, 0) + 1
                )

        return self._build_result_summary(frame_width, frame_height, counting_line_y)

    def _build_result_summary(
        self, frame_width: int, frame_height: int, counting_line_y: int
    ) -> Dict[str, Any]:
        """Compiles active tracks, lane densities, and line coordinates."""
        lane_densities = {lane: 0 for lane in range(1, self.num_lanes + 1)}
        active_tracks = []

        for obj_id, (cx, cy) in self.objects.items():
            lane = self.get_lane_for_x(cx, frame_width)
            lane_densities[lane] = lane_densities.get(lane, 0) + 1

            active_tracks.append({
                "id": obj_id,
                "class_name": self.object_classes.get(obj_id, "vehicle"),
                "box": self.object_boxes.get(obj_id, (0, 0, 0, 0)),
                "centroid": (cx, cy),
                "lane": lane,
                "has_crossed": obj_id in self.crossed_object_ids,
            })

        lane_divider_xs = [int(frame_width * b) for b in self.lane_boundaries]

        return {
            "active_tracks": active_tracks,
            "lane_densities": lane_densities,
            "lane_crossed": dict(self.lane_crossed_counts),
            "counting_line_y": counting_line_y,
            "lane_divider_xs": lane_divider_xs,
        }
