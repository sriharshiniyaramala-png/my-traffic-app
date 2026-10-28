"""
database.py
===========
SQLite Database Module for the Smart Emergency Traffic Priority System.

This module handles:
1. Creating database tables if they do not exist.
2. Logging vehicle counts and lane density observations with timestamps.
3. Logging traffic signal recommendations and emergency override events.
4. Fetching historical data safely, returning clean empty results if no records exist.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd


DEFAULT_DB_PATH = "data/traffic_system.db"


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Establishes and returns a connection to the SQLite database.
    Enables WAL mode for faster concurrent reads and writes.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes the database schema if tables do not already exist.
    Creates tables for traffic observations and signal recommendations.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()

        # Table 1: Frame-by-frame or windowed traffic observations
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS traffic_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                video_source TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                lane_id INTEGER NOT NULL,
                active_count INTEGER NOT NULL,
                cumulative_crossed INTEGER NOT NULL,
                cars INTEGER NOT NULL DEFAULT 0,
                motorcycles INTEGER NOT NULL DEFAULT 0,
                buses INTEGER NOT NULL DEFAULT 0,
                trucks INTEGER NOT NULL DEFAULT 0
            );
            """
        )

        # Table 2: Signal decisions and emergency simulation events
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS signal_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                recommended_lane INTEGER NOT NULL,
                green_duration_sec REAL NOT NULL,
                is_emergency INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL,
                lane_densities_json TEXT
            );
            """
        )

        # Create indexes for fast query performance on timestamps and lanes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_time ON traffic_observations(timestamp);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sig_time ON signal_recommendations(timestamp);"
        )

        conn.commit()
    finally:
        conn.close()


def log_traffic_observation(
    db_path: str,
    video_source: str,
    frame_number: int,
    lane_id: int,
    active_count: int,
    cumulative_crossed: int,
    breakdown: Optional[Dict[str, int]] = None,
    timestamp: Optional[str] = None,
) -> None:
    """
    Inserts a vehicle observation record into the database.
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    if breakdown is None:
        breakdown = {}

    cars = breakdown.get("car", 0)
    motorcycles = breakdown.get("motorcycle", 0)
    buses = breakdown.get("bus", 0)
    trucks = breakdown.get("truck", 0)

    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO traffic_observations (
                timestamp, video_source, frame_number, lane_id,
                active_count, cumulative_crossed,
                cars, motorcycles, buses, trucks
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                timestamp,
                video_source,
                frame_number,
                lane_id,
                active_count,
                cumulative_crossed,
                cars,
                motorcycles,
                buses,
                trucks,
            ),
        )
        conn.commit()
    except Exception as e:
        # Gracefully handle write errors without crashing video stream
        print(f"[Warning] Failed to log traffic observation: {e}")
    finally:
        if conn is not None:
            conn.close()


def log_signal_recommendation(
    db_path: str,
    recommended_lane: int,
    green_duration_sec: float,
    is_emergency: bool,
    reason: str,
    lane_densities: Optional[Dict[int, int]] = None,
    timestamp: Optional[str] = None,
) -> None:
    """
    Inserts a traffic signal recommendation or emergency override into the database.
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    densities_str = json.dumps(lane_densities) if lane_densities else "{}"

    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO signal_recommendations (
                timestamp, recommended_lane, green_duration_sec,
                is_emergency, reason, lane_densities_json
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                timestamp,
                recommended_lane,
                green_duration_sec,
                1 if is_emergency else 0,
                reason,
                densities_str,
            ),
        )
        conn.commit()
    except Exception as e:
        print(f"[Warning] Failed to log signal recommendation: {e}")
    finally:
        if conn is not None:
            conn.close()


def get_traffic_history(db_path: str = DEFAULT_DB_PATH, limit: int = 500) -> pd.DataFrame:
    """
    Fetches recent traffic observations as a pandas DataFrame.
    Returns an empty DataFrame with expected columns if table is empty or missing.
    """
    columns = [
        "id", "timestamp", "video_source", "frame_number", "lane_id",
        "active_count", "cumulative_crossed", "cars", "motorcycles", "buses", "trucks"
    ]
    conn = None
    try:
        conn = get_db_connection(db_path)
        query = f"SELECT * FROM traffic_observations ORDER BY id DESC LIMIT {limit};"
        df = pd.read_sql_query(query, conn)
        if df.empty:
            return pd.DataFrame(columns=columns)
        return df
    except Exception as e:
        print(f"[Info] Handled database read notice: {e}")
        return pd.DataFrame(columns=columns)
    finally:
        if conn is not None:
            conn.close()


def get_signal_history(db_path: str = DEFAULT_DB_PATH, limit: int = 100) -> pd.DataFrame:
    """
    Fetches recent signal recommendations as a pandas DataFrame.
    Returns an empty DataFrame if no records exist.
    """
    columns = [
        "id", "timestamp", "recommended_lane", "green_duration_sec",
        "is_emergency", "reason", "lane_densities_json"
    ]
    conn = None
    try:
        conn = get_db_connection(db_path)
        query = f"SELECT * FROM signal_recommendations ORDER BY id DESC LIMIT {limit};"
        df = pd.read_sql_query(query, conn)
        if df.empty:
            return pd.DataFrame(columns=columns)
        return df
    except Exception as e:
        print(f"[Info] Handled database read notice: {e}")
        return pd.DataFrame(columns=columns)
    finally:
        if conn is not None:
            conn.close()


def get_summary_statistics(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Computes high-level summary metrics from historical data.
    Gracefully handles empty databases by returning zeros.
    """
    default_summary = {
        "total_records": 0,
        "total_vehicles_crossed": 0,
        "peak_lane_density": 0,
        "emergency_events": 0,
        "avg_green_duration": 0.0,
    }
    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()

        # Total observations
        cursor.execute("SELECT COUNT(*) FROM traffic_observations;")
        obs_count = cursor.fetchone()[0]

        if obs_count == 0:
            return default_summary

        # Max crossed count across all lanes
        cursor.execute("SELECT COALESCE(MAX(cumulative_crossed), 0), COALESCE(MAX(active_count), 0) FROM traffic_observations;")
        crossed, peak_active = cursor.fetchone()

        # Signal recommendations stats
        cursor.execute("SELECT COUNT(*), COALESCE(SUM(is_emergency), 0), COALESCE(AVG(green_duration_sec), 0.0) FROM signal_recommendations;")
        sig_count, emergency_count, avg_green = cursor.fetchone()

        return {
            "total_records": obs_count,
            "total_vehicles_crossed": int(crossed),
            "peak_lane_density": int(peak_active),
            "emergency_events": int(emergency_count),
            "avg_green_duration": round(float(avg_green), 1),
        }
    except Exception:
        return default_summary
    finally:
        if conn is not None:
            conn.close()


def clear_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Clears all rows from the database tables. Useful for resetting test runs.
    """
    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM traffic_observations;")
        cursor.execute("DELETE FROM signal_recommendations;")
        conn.commit()
    finally:
        if conn is not None:
            conn.close()
