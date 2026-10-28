"""
detector.py
===========
YOLO-based Vehicle Detection and Visual Annotation Module.

Key Features:
1. Loads pretrained YOLOv8n (nano) model from Ultralytics without training from scratch.
2. Filters detections specifically for target vehicle classes from Microsoft COCO:
   - Car (Class ID: 2)
   - Motorcycle (Class ID: 3)
   - Bus (Class ID: 5)
   - Truck (Class ID: 7)
3. Annotates frames with bounding boxes, vehicle labels, track IDs, lane dividers,
   counting lines, and real-time traffic signal status.
4. Gracefully handles model load failures and missing files.
"""

from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np


# Target vehicle classes defined in Microsoft COCO dataset
# Note: Standard COCO does NOT have an emergency vehicle class (e.g. Ambulance).
# Emergency priority is handled via simulation mode for academic realism.
TARGET_VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Distinct BGR colors for vehicle visualization
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (255, 178, 50),         # Vibrant Cyan/Blue
    "motorcycle": (0, 215, 255),    # Amber/Gold
    "bus": (255, 105, 180),        # Hot Pink / Purple
    "truck": (50, 205, 50),        # Lime Green
    "emergency_sim": (0, 0, 255),  # Red for simulated emergency vehicle
    "default": (200, 200, 200),
}


class VehicleDetector:
    """
    Wrapper around the Ultralytics YOLO model for detecting and annotating vehicles.
    """

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.35):
        """
        Initializes the YOLO model.

        Args:
            model_name: Pretrained YOLO model file or identifier (default 'yolov8n.pt').
            conf_threshold: Minimum detection confidence score (0.0 to 1.0).
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = self._load_yolo_model(model_name)

    def _load_yolo_model(self, model_name: str):
        """
        Loads the Ultralytics YOLO model safely with informative error messages.
        """
        try:
            from ultralytics import YOLO
            # Ultralytics will automatically cache yolov8n.pt locally if not present
            print(f"[Info] Loading pretrained YOLO model: {model_name}...")
            model = YOLO(model_name)
            print("[Success] YOLO model loaded successfully.")
            return model
        except Exception as e:
            print(f"[Error] Failed to load YOLO model '{model_name}': {e}")
            raise RuntimeError(
                f"Could not load YOLO model '{model_name}'. "
                "Ensure internet access for initial download or verify that ultralytics is installed."
            ) from e

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Performs inference on a single video frame and filters for vehicles.

        Args:
            frame: OpenCV image array (BGR format).

        Returns:
            List of detected vehicle dictionaries containing:
            - 'box': [x1, y1, x2, y2]
            - 'class_id': int
            - 'class_name': str
            - 'confidence': float
        """
        if frame is None or frame.size == 0:
            return []

        # Run inference in verbose=False mode for clean terminal output
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=list(TARGET_VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                coords = [int(v) for v in box.xyxy[0].tolist()]

                class_name = TARGET_VEHICLE_CLASSES.get(cls_id, "vehicle")

                detections.append({
                    "box": coords,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 2),
                })

        return detections

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracking_info: Dict[str, Any],
        signal_state: Optional[Dict[str, Any]] = None,
        simulated_emergency_lane: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draws bounding boxes, track IDs, lane boundary guides, counting line,
        and the real-time traffic signal status HUD onto the video frame.

        Args:
            frame: OpenCV image array.
            tracking_info: Output dictionary from CentroidTracker.update().
            signal_state: Output dictionary from TrafficSignalController.step().
            simulated_emergency_lane: Lane ID currently carrying simulated emergency vehicle.

        Returns:
            Annotated frame (OpenCV BGR image).
        """
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        counting_line_y = tracking_info.get("counting_line_y", int(height * 0.65))
        lane_divider_xs = tracking_info.get("lane_divider_xs", [])
        active_tracks = tracking_info.get("active_tracks", [])
        lane_densities = tracking_info.get("lane_densities", {})
        lane_crossed = tracking_info.get("lane_crossed", {})

        # 1. Draw Lane Dividers
        for div_x in lane_divider_xs:
            cv2.line(
                annotated,
                (div_x, 0),
                (div_x, height),
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )

        # 2. Draw Lane Headers and Metrics at the top of each lane
        num_lanes = len(lane_divider_xs) + 1
        x_bounds = [0] + lane_divider_xs + [width]

        for i in range(num_lanes):
            lane_num = i + 1
            x_start = x_bounds[i]
            x_end = x_bounds[i + 1]
            center_x = (x_start + x_end) // 2

            density = lane_densities.get(lane_num, 0)
            crossed = lane_crossed.get(lane_num, 0)

            # Highlight emergency lane with red marker
            is_em_lane = simulated_emergency_lane == lane_num

            # Determine traffic signal light for this lane
            light_color_bgr = (0, 0, 255)  # Default RED
            light_text = "RED"
            if signal_state:
                light_status = signal_state.get("lane_lights", {}).get(lane_num, "RED")
                if light_status == "GREEN":
                    light_color_bgr = (0, 220, 0)
                    light_text = "GREEN"
                elif light_status == "YELLOW":
                    light_color_bgr = (0, 215, 255)
                    light_text = "YELLOW"

            # Lane badge background
            badge_y1 = 12
            badge_y2 = 72
            badge_x1 = max(x_start + 10, center_x - 90)
            badge_x2 = min(x_end - 10, center_x + 90)

            # Draw semi-transparent header box
            overlay = annotated.copy()
            bg_color = (0, 0, 160) if is_em_lane else (35, 35, 35)
            cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), bg_color, -1)
            cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

            # Draw signal indicator circle
            circle_x = badge_x1 + 18
            circle_y = badge_y1 + 18
            cv2.circle(annotated, (circle_x, circle_y), 8, light_color_bgr, -1)
            cv2.circle(annotated, (circle_x, circle_y), 9, (255, 255, 255), 1)

            # Lane title text
            em_tag = " [EMERGENCY]" if is_em_lane else ""
            cv2.putText(
                annotated,
                f"Lane {lane_num}{em_tag}",
                (badge_x1 + 32, badge_y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Density & Crossed Counts
            cv2.putText(
                annotated,
                f"Active: {density}  Passed: {crossed}",
                (badge_x1 + 12, badge_y1 + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        # 3. Draw Virtual Counting Line
        cv2.line(
            annotated,
            (0, counting_line_y),
            (width, counting_line_y),
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "--- TRIPWIRE COUNTING LINE ---",
            (15, counting_line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # 4. Draw Vehicle Bounding Boxes and Persistent Track IDs
        for track in active_tracks:
            obj_id = track["id"]
            cls_name = track["class_name"]
            x1, y1, x2, y2 = track["box"]
            cx, cy = track["centroid"]
            lane = track["lane"]
            has_crossed = track["has_crossed"]

            # Visual color scheme
            is_lane_emergency = (simulated_emergency_lane == lane)
            box_color = CLASS_COLORS.get(cls_name, CLASS_COLORS["default"])
            if is_lane_emergency:
                box_color = (0, 0, 255)  # Alert Red

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Centroid marker
            center_color = (0, 255, 255) if has_crossed else (0, 255, 0)
            cv2.circle(annotated, (cx, cy), 4, center_color, -1)

            # Label badge
            status_tag = " [OK]" if has_crossed else ""
            label = f"#{obj_id} {cls_name}{status_tag}"

            # Calculate label text size
            (txt_w, txt_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - txt_h - 6)),
                (x1 + txt_w + 6, max(0, y1)),
                box_color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 5. Draw Bottom Status Bar (Simulation Notice & Controller Summary)
        bar_height = 42
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, height - bar_height), (width, height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        total_vehicles_active = len(active_tracks)
        total_passed = sum(lane_crossed.values())

        status_left = f"Live Vehicles: {total_vehicles_active} | Cumulative Passed: {total_passed}"
        cv2.putText(
            annotated,
            status_left,
            (15, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Simulation Mode Disclaimer
        disclaimer = "ACADEMIC SIMULATION - NO REAL INFRASTRUCTURE CONTROL"
        (d_w, d_h), _ = cv2.getTextSize(disclaimer, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            annotated,
            disclaimer,
            (width - d_w - 15, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        return annotated
