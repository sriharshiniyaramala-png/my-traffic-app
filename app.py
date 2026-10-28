"""
app.py
======
Smart Emergency Traffic Priority System - Streamlit Web Dashboard.

Features:
- Video source selection (Sample synthetic traffic video or user MP4 upload).
- Real-time YOLOv8 vehicle detection & persistent centroid tracking.
- Density-based adaptive traffic signal recommendation.
- Simulated Emergency Priority Override with safe, labeled clearance mode.
- SQLite persistence of traffic observations and signal state changes.
- Interactive Plotly visualizations and historical analytics.
"""

import os
import time
import tempfile
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.database import (
    init_db,
    log_traffic_observation,
    log_signal_recommendation,
    get_traffic_history,
    get_signal_history,
    get_summary_statistics,
    clear_database,
    DEFAULT_DB_PATH,
)
from src.detector import VehicleDetector
from src.tracker import CentroidTracker
from src.signal_controller import TrafficSignalController
from src.video_generator import generate_synthetic_traffic_video


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Smart Emergency Traffic Priority System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database tables on startup
init_db(DEFAULT_DB_PATH)


# ---------------- CACHED MODEL LOADER ----------------
@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def get_detector(conf_threshold: float = 0.35) -> VehicleDetector:
    """Loads and caches the YOLO detector instance across Streamlit reruns."""
    return VehicleDetector(model_name="yolov8n.pt", conf_threshold=conf_threshold)


# ---------------- HELPER: TRAFFIC LIGHT HTML ----------------
def render_traffic_light(lane_num: int, state: str, is_emergency: bool = False) -> str:
    """
    Generates a clean HTML badge representing the traffic light for a lane.
    """
    color_map = {
        "GREEN": ("#10b981", "#d1fae5"),   # Green
        "YELLOW": ("#f59e0b", "#fef3c7"),  # Amber
        "RED": ("#ef4444", "#fee2e2"),     # Red
    }
    glow, bg = color_map.get(state, ("#ef4444", "#fee2e2"))
    em_badge = " 🚨 [EMERGENCY]" if is_emergency else ""

    html = f"""
    <div style="
        border: 2px solid {glow};
        border-radius: 10px;
        padding: 12px;
        background: {bg};
        text-align: center;
        margin-bottom: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    ">
        <h4 style="margin: 0; color: #1f2937; font-weight: 700;">Lane {lane_num}{em_badge}</h4>
        <div style="
            display: inline-block;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background-color: {glow};
            margin: 8px 0;
            box-shadow: 0 0 10px {glow};
        "></div>
        <p style="margin: 0; font-size: 14px; font-weight: 800; color: {glow}; letter-spacing: 1px;">
            {state}
        </p>
    </div>
    """
    return html


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Reverse.svg",
        width=200,
    )
    st.title("🚦 System Controls")

    st.markdown("---")
    st.subheader("1. Video Source")
    video_source_type = st.radio(
        "Select Video Input:",
        ("Sample 3-Lane Traffic (Built-in)", "Upload Video (MP4 / AVI)"),
    )

    video_path = None
    if video_source_type == "Sample 3-Lane Traffic (Built-in)":
        sample_path = "data/sample_traffic.mp4"
        if not os.path.exists(sample_path):
            with st.spinner("Generating sample traffic video..."):
                generate_synthetic_traffic_video(sample_path)
        video_path = sample_path
        st.success("Sample video ready.")
    else:
        uploaded_file = st.file_uploader("Upload video file:", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            video_path = tfile.name
            st.success("Custom video uploaded.")

    st.markdown("---")
    st.subheader("2. Emergency Priority Simulation")
    st.info(
        "⚠️ **Simulation Only:** Never controls real traffic infrastructure. "
        "Activates safe preemption override for the selected lane."
    )

    emergency_toggle = st.checkbox("🚨 Activate Emergency Simulation Mode", value=False)
    emergency_lane = None
    if emergency_toggle:
        emergency_lane = st.selectbox("Select Emergency Vehicle Lane:", (1, 2, 3), index=1)
        st.warning(f"Simulating Emergency Vehicle in **Lane {emergency_lane}**.")

    st.markdown("---")
    st.subheader("3. Signal Timing Parameters")
    min_green = st.slider("Minimum Green Light (sec):", 5.0, 20.0, 10.0, 1.0)
    max_green = st.slider("Maximum Green Light (sec):", 20.0, 90.0, 45.0, 5.0)
    sec_per_veh = st.slider("Green Extension per Vehicle (sec):", 1.0, 5.0, 2.5, 0.5)

    st.markdown("---")
    st.subheader("4. Detection Threshold")
    conf_threshold = st.slider("YOLO Confidence:", 0.20, 0.80, 0.35, 0.05)


# ---------------- MAIN DASHBOARD LAYOUT ----------------
st.title("Smart Emergency Traffic Priority System")
st.caption(
    "AI-driven Vehicle Detection (YOLOv8) | Centroid Tracking & Line Crossing | "
    "Adaptive Signal Timing | Emergency Preemption Simulation"
)

# Tabs
tab_monitor, tab_analytics, tab_about = st.tabs([
    "📹 Live Video & Signal Control",
    "📊 Historical Analytics & Database",
    "📘 Project Guide & Viva Prep",
])


# =================================================================
# TAB 1: LIVE MONITORING & SIGNAL SIMULATION
# =================================================================
with tab_monitor:
    col_video, col_signals = st.columns([2.2, 1.0])

    with col_signals:
        st.subheader("🚥 Signal State")
        signal_placeholder = st.empty()

        st.subheader("📈 Real-time Metrics")
        kpi_active = st.metric("Live Active Vehicles", 0)
        kpi_passed = st.metric("Cumulative Passed", 0)
        kpi_lane_density = st.empty()

    with col_video:
        st.subheader("🎥 Annotated Traffic Feed")
        video_placeholder = st.empty()

        # Run control buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            start_btn = st.button("▶️ Start Simulation", type="primary", use_container_width=True)
        with btn_col2:
            stop_btn = st.button("⏹️ Stop Simulation", use_container_width=True)
        with btn_col3:
            reset_db_btn = st.button("🗑️ Reset Database", use_container_width=True)

        if reset_db_btn:
            clear_database(DEFAULT_DB_PATH)
            st.success("Database records cleared.")

    if start_btn:
        if not video_path or not os.path.exists(video_path):
            st.error("Please provide or generate a valid video file to start.")
        else:
            # Initialize Pipeline
            detector = get_detector(conf_threshold=conf_threshold)
            tracker = CentroidTracker(num_lanes=3, counting_line_ratio=0.65)
            controller = TrafficSignalController(
                num_lanes=3,
                min_green_sec=min_green,
                max_green_sec=max_green,
                base_green_sec=min_green,
                sec_per_vehicle=sec_per_veh,
            )

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"Failed to open video file at: {video_path}")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or fps > 120:
                    fps = 25.0
                dt = 1.0 / fps

                frame_idx = 0
                sim_running = True

                # Progress loop
                while cap.isOpened() and sim_running:
                    ret, frame = cap.read()
                    if not ret:
                        # Loop video for continuous presentation
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break

                    h, w = frame.shape[:2]

                    # 1. Detect target vehicles
                    detections = detector.detect(frame)

                    # 2. Update tracking & count crossings
                    tracking_info = tracker.update(detections, w, h)
                    lane_densities = tracking_info["lane_densities"]
                    lane_crossed = tracking_info["lane_crossed"]

                    # 3. Step traffic signal logic
                    sig_state = controller.step(
                        dt_sec=dt,
                        lane_densities=lane_densities,
                        emergency_lane=emergency_lane,
                    )

                    # Log decisions if a transition occurred
                    if sig_state["decision_made"] and sig_state["decision_info"]:
                        d_info = sig_state["decision_info"]
                        log_signal_recommendation(
                            db_path=DEFAULT_DB_PATH,
                            recommended_lane=d_info["recommended_lane"],
                            green_duration_sec=d_info["green_duration_sec"],
                            is_emergency=d_info["is_emergency"],
                            reason=d_info["reason"],
                            lane_densities=lane_densities,
                        )

                    # Log periodic traffic observation (every 10 frames to keep DB lightweight)
                    if frame_idx % 10 == 0:
                        # Count breakdown
                        breakdown = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
                        for track in tracking_info["active_tracks"]:
                            cname = track["class_name"]
                            if cname in breakdown:
                                breakdown[cname] += 1

                        for lane_id, active_c in lane_densities.items():
                            log_traffic_observation(
                                db_path=DEFAULT_DB_PATH,
                                video_source=os.path.basename(video_path),
                                frame_number=frame_idx,
                                lane_id=lane_id,
                                active_count=active_c,
                                cumulative_crossed=lane_crossed.get(lane_id, 0),
                                breakdown=breakdown,
                            )

                    # 4. Annotate frame
                    annotated_frame = detector.annotate_frame(
                        frame=frame,
                        tracking_info=tracking_info,
                        signal_state=sig_state,
                        simulated_emergency_lane=emergency_lane,
                    )

                    # Convert BGR to RGB for Streamlit display
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    # 5. Update Sidebar & KPI cards
                    total_active = len(tracking_info["active_tracks"])
                    total_passed = sum(lane_crossed.values())

                    # Signal lights widget
                    lights_html = ""
                    for l_idx in range(1, 4):
                        l_state = sig_state["lane_lights"].get(l_idx, "RED")
                        is_em = (emergency_lane == l_idx)
                        lights_html += render_traffic_light(l_idx, l_state, is_em)

                    status_banner = f"""
                    <div style="background:#1e293b; color:#f8fafc; padding:10px; border-radius:8px; margin-bottom:10px;">
                        <b>Active Green:</b> Lane {sig_state['active_lane']}<br/>
                        <b>Timer Remaining:</b> {sig_state['time_remaining_sec']}s / {sig_state['allocated_green_sec']}s<br/>
                        <b>Status:</b> {sig_state['state']}
                    </div>
                    """
                    signal_placeholder.markdown(status_banner + lights_html, unsafe_allow_html=True)

                    # Density progress bars
                    density_bar_html = f"""
                    <div style="font-size: 13px;">
                        <b>Lane 1 Density:</b> {lane_densities.get(1, 0)} vehicles<br/>
                        <b>Lane 2 Density:</b> {lane_densities.get(2, 0)} vehicles<br/>
                        <b>Lane 3 Density:</b> {lane_densities.get(3, 0)} vehicles
                    </div>
                    """
                    kpi_lane_density.markdown(density_bar_html, unsafe_allow_html=True)

                    frame_idx += 1
                    time.sleep(0.01)  # Gentle yield to keep UI responsive

                cap.release()


# =================================================================
# TAB 2: HISTORICAL ANALYTICS & DATABASE
# =================================================================
with tab_analytics:
    st.subheader("📊 Traffic Intelligence & Audit History")
    st.caption("All observations and dynamic signal decisions are recorded in SQLite.")

    # High-level KPIs
    summary = get_summary_statistics(DEFAULT_DB_PATH)
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Total DB Records", summary["total_records"])
    sc2.metric("Vehicles Passed", summary["total_vehicles_crossed"])
    sc3.metric("Peak Density", summary["peak_lane_density"])
    sc4.metric("🚨 Emergency Overrides", summary["emergency_events"])
    sc5.metric("Avg Green Time", f"{summary['avg_green_duration']}s")

    st.markdown("---")

    # Fetch tables
    df_traffic = get_traffic_history(DEFAULT_DB_PATH, limit=300)
    df_signals = get_signal_history(DEFAULT_DB_PATH, limit=100)

    if df_traffic.empty:
        st.info("No traffic observations logged yet. Run the simulation in Tab 1 to populate data.")
    else:
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("##### 📈 Lane Traffic Density Trend")
            df_plot = df_traffic.sort_values("id")
            fig_density = px.line(
                df_plot,
                x="id",
                y="active_count",
                color="lane_id",
                labels={"id": "Observation #", "active_count": "Active Vehicles", "lane_id": "Lane"},
                title="Real-Time Lane Density Variations",
            )
            fig_density.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_density, use_container_width=True)

        with col_chart2:
            st.markdown("##### 🚗 Detected Vehicle Classification Breakdown")
            vehicle_sums = {
                "Cars": df_traffic["cars"].sum(),
                "Motorcycles": df_traffic["motorcycles"].sum(),
                "Buses": df_traffic["buses"].sum(),
                "Trucks": df_traffic["trucks"].sum(),
            }
            df_veh = pd.DataFrame(list(vehicle_sums.items()), columns=["Vehicle Type", "Total Detections"])
            fig_bar = px.bar(
                df_veh,
                x="Vehicle Type",
                y="Total Detections",
                color="Vehicle Type",
                title="Vehicle Class Distribution (COCO)",
            )
            fig_bar.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.markdown("##### 🚦 Recent Signal Recommendations & Override Log")
        if df_signals.empty:
            st.write("No signal transitions recorded yet.")
        else:
            st.dataframe(
                df_signals[[
                    "timestamp", "recommended_lane", "green_duration_sec",
                    "is_emergency", "reason"
                ]],
                use_container_width=True,
                height=220,
            )

        st.markdown("##### 📋 Raw Traffic Observations Data")
        st.dataframe(df_traffic, use_container_width=True, height=220)


# =================================================================
# TAB 3: PROJECT GUIDE & VIVA QUESTIONS
# =================================================================
with tab_about:
    st.subheader("📘 Project Overview & Viva Exam Guide")

    st.markdown(
        """
        ### 1. How the System Works
        - **Step 1: Detection**: Video frames pass through a pretrained **YOLOv8n** model trained on Microsoft COCO. Vehicles (cars, motorcycles, buses, trucks) are isolated.
        - **Step 2: Tracking**: A **Centroid Tracker** matches detections across consecutive frames using Euclidean distance, giving each vehicle a consistent unique ID.
        - **Step 3: Counting**: A virtual tripwire detects when vehicle centroids cross the boundary, incrementing cumulative counts **without duplicate counting**.
        - **Step 4: Density Estimation**: Road geometry divides the screen into Lane 1, Lane 2, and Lane 3. The system computes active density per lane.
        - **Step 5: Dynamic Signal Algorithm**:
          - Clamps green light duration between $T_{min}$ and $T_{max}$.
          - Allocates additional green light proportional to vehicle count ($+2.5s$ per car).
          - Enforces starvation prevention so quiet lanes are never ignored.
        - **Step 6: Emergency Preemption Simulation**:
          - Standard COCO has no "ambulance" class.
          - In accordance with honest engineering, emergency priority is implemented as a labeled simulation override that preempts normal traffic safely.
        - **Step 7: SQLite Persistence**: Every frame metric and signal decision is logged for auditing and analytics.
        """
    )

    st.markdown("---")
    st.markdown("### 🎓 Frequently Asked Viva Questions & Answers")

    with st.expander("Q1: Why is detection alone insufficient for counting vehicles?"):
        st.write(
            "**Answer:** A detector (like YOLO) evaluates single frames in isolation. "
            "If a car appears in 30 consecutive frames, detection triggers 30 times. "
            "Object Tracking assigns persistent IDs across frames, enabling a virtual counting line "
            "to register the vehicle only once when it crosses."
        )

    with st.expander("Q2: Why doesn't standard YOLO recognize an ambulance out of the box?"):
        st.write(
            "**Answer:** The standard pretrained model is trained on the Microsoft COCO dataset, "
            "which defines 80 general categories (car, bus, truck, motorcycle), but does NOT include "
            "an 'ambulance' or 'fire engine' category. In a production system, one would collect a custom "
            "fine-tuning dataset or use V2X radio beacons. In this project, we implement an explicit, "
            "labeled Emergency Simulation Override."
        )

    with st.expander("Q3: How does the dynamic signal algorithm prevent starvation?"):
        st.write(
            "**Answer:** Pure density-based systems risk starving low-traffic lanes if one lane remains continuously packed. "
            "Our algorithm tracks wait cycles for each lane. If a lane waits for more than a configured threshold (e.g., 2 cycles), "
            "it is granted guaranteed green light regardless of density."
        )

    with st.expander("Q4: What is the purpose of the minimum and maximum green time clamps?"):
        st.write(
            "**Answer:** Minimum green time (e.g. 10s) ensures pedestrian safety and vehicle startup clearance time. "
            "Maximum green time (e.g. 45s) prevents driver frustration and queue build-up on cross streets."
        )

    with st.expander("Q5: Why choose SQLite for this application?"):
        st.write(
            "**Answer:** SQLite is a serverless, self-contained, transactional database engine. "
            "It requires zero setup or server configuration, stores data in a single portable `.db` file, "
            "and supports standard SQL queries and ACID compliance."
        )
