"""
video_generator.py
==================
Self-contained synthetic traffic footage generator using OpenCV.

Generates a realistic multi-lane roadway video with:
- Textured asphalt road surface.
- Dashed lane divider lines.
- Simulated cars, trucks, motorcycles moving at realistic speeds.
- Simulated emergency ambulance with flashing red/blue beacon lights.
- Guarantees immediate zero-setup testing for students without needing external downloads.
"""

import math
import random
import cv2
import numpy as np


def generate_synthetic_traffic_video(
    output_path: str = "data/sample_traffic.mp4",
    width: int = 960,
    height: int = 540,
    num_frames: int = 250,
    fps: int = 25,
) -> str:
    """
    Renders and saves a synthetic traffic video containing cars, motorcycles,
    trucks, and a simulated emergency vehicle.

    Args:
        output_path: Path to save the MP4 video.
        width: Video frame width in pixels.
        height: Video frame height in pixels.
        num_frames: Total frames to generate (~10 seconds at 25 fps).
        fps: Frames per second.

    Returns:
        output_path on success.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        # Fallback to AVI with MJPG if mp4v codec is unavailable on this OS
        fallback_path = output_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(fallback_path, fourcc, fps, (width, height))
        output_path = fallback_path

    # Define 3 lane center X coordinates
    lane_width = width // 3
    lane_centers = [lane_width // 2, lane_width + lane_width // 2, 2 * lane_width + lane_width // 2]

    # Pre-configure simulated vehicles
    # Each vehicle: {'type', 'lane', 'x', 'y', 'w', 'h', 'speed', 'color', 'is_emergency'}
    vehicles = [
        # Lane 1 (Dense traffic)
        {"type": "car", "lane": 1, "y": -60, "w": 65, "h": 110, "speed": 4.2, "color": (40, 40, 210), "is_emergency": False},
        {"type": "motorcycle", "lane": 1, "y": -220, "w": 35, "h": 60, "speed": 4.8, "color": (220, 150, 30), "is_emergency": False},
        {"type": "car", "lane": 1, "y": -340, "w": 65, "h": 105, "speed": 4.0, "color": (180, 50, 50), "is_emergency": False},
        {"type": "car", "lane": 1, "y": -490, "w": 68, "h": 115, "speed": 4.1, "color": (50, 180, 50), "is_emergency": False},

        # Lane 2 (Moderate traffic with simulated emergency vehicle)
        {"type": "car", "lane": 2, "y": -80, "w": 66, "h": 110, "speed": 4.5, "color": (200, 200, 40), "is_emergency": False},
        # Emergency vehicle (Ambulance with white body & red cross + flashing beacon)
        {"type": "truck", "lane": 2, "y": -260, "w": 75, "h": 130, "speed": 5.5, "color": (245, 245, 245), "is_emergency": True},
        {"type": "motorcycle", "lane": 2, "y": -440, "w": 35, "h": 60, "speed": 4.6, "color": (30, 180, 200), "is_emergency": False},

        # Lane 3 (Light traffic: bus & truck)
        {"type": "bus", "lane": 3, "y": -120, "w": 85, "h": 160, "speed": 3.6, "color": (20, 140, 230), "is_emergency": False},
        {"type": "car", "lane": 3, "y": -380, "w": 65, "h": 105, "speed": 4.3, "color": (120, 120, 120), "is_emergency": False},
    ]

    for frame_idx in range(num_frames):
        # 1. Base Road (Dark Gray Asphalt)
        frame = np.full((height, width, 3), (45, 45, 45), dtype=np.uint8)

        # 2. Road shoulders / curb borders
        cv2.line(frame, (8, 0), (8, height), (220, 220, 220), 4)
        cv2.line(frame, (width - 8, 0), (width - 8, height), (220, 220, 220), 4)

        # 3. Animated dashed lane divider lines
        dash_offset = (frame_idx * 6) % 40
        for divider_idx in [1, 2]:
            div_x = divider_idx * lane_width
            for y_start in range(-dash_offset, height, 40):
                cv2.line(
                    frame,
                    (div_x, y_start),
                    (div_x, y_start + 22),
                    (230, 230, 230),
                    3,
                )

        # 4. Render and update vehicles
        for v in vehicles:
            v["y"] += v["speed"]

            # Vehicle bounding rectangle
            lane_cx = lane_centers[v["lane"] - 1]
            vx1 = int(lane_cx - v["w"] // 2)
            vy1 = int(v["y"])
            vx2 = int(lane_cx + v["w"] // 2)
            vy2 = int(v["y"] + v["h"])

            # Only draw if within frame view
            if vy2 >= 0 and vy1 <= height:
                # Vehicle body with rounded appearance
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), v["color"], -1)
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (20, 20, 20), 2)

                # Windshield and rear windows
                cv2.rectangle(frame, (vx1 + 6, vy1 + 15), (vx2 - 6, vy1 + 35), (20, 20, 20), -1)
                cv2.rectangle(frame, (vx1 + 6, vy2 - 25), (vx2 - 6, vy2 - 12), (20, 20, 20), -1)

                # Wheels
                cv2.rectangle(frame, (vx1 - 3, vy1 + 18), (vx1, vy1 + 38), (10, 10, 10), -1)
                cv2.rectangle(frame, (vx2, vy1 + 18), (vx2 + 3, vy1 + 38), (10, 10, 10), -1)
                cv2.rectangle(frame, (vx1 - 3, vy2 - 38), (vx1, vy2 - 18), (10, 10, 10), -1)
                cv2.rectangle(frame, (vx2, vy2 - 38), (vx2 + 3, vy2 - 18), (10, 10, 10), -1)

                # Emergency Vehicle Features
                if v["is_emergency"]:
                    # Red cross symbol on ambulance roof
                    cross_cx = (vx1 + vx2) // 2
                    cross_cy = (vy1 + vy2) // 2
                    cv2.rectangle(frame, (cross_cx - 4, cross_cy - 14), (cross_cx + 4, cross_cy + 14), (0, 0, 220), -1)
                    cv2.rectangle(frame, (cross_cx - 14, cross_cy - 4), (cross_cx + 14, cross_cy + 4), (0, 0, 220), -1)

                    # Alternating flashing red/blue emergency light beacon
                    flash_phase = (frame_idx // 3) % 2
                    left_light = (0, 0, 255) if flash_phase == 0 else (255, 100, 0)
                    right_light = (255, 100, 0) if flash_phase == 0 else (0, 0, 255)

                    cv2.circle(frame, (cross_cx - 15, vy1 + 8), 6, left_light, -1)
                    cv2.circle(frame, (cross_cx + 15, vy1 + 8), 6, right_light, -1)

        out.write(frame)

    out.release()
    print(f"[Success] Generated synthetic test video at: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_synthetic_traffic_video()
