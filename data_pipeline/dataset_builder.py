# File: data/dataset_builder.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from utils.logger import setup_logger
from config.settings import SESSION_DATA_PATH, WEATHER_DATA_PATH, LAP_DATA_PATH

logger = setup_logger("DatasetBuilder")

# Define paths
INPUT_LAPS = "data/raw/lap_data.csv"
INPUT_SESSIONS = "data/raw/session_raw.csv"
INPUT_WEATHER = "data/raw/weather_raw.csv"
OUTPUT_CLEAN_DATA = "data/processed/cleaned_f1_data.csv"
SUMMARY_FILE = "data/processed/summary.txt"

def infer_session_type(name):
    """Infer session type based on session name keywords."""
    name = str(name).lower()
    if "qualifying" in name:
        return "qualifying"
    elif "practice" in name:
        return "practice"
    elif "race" in name:
        return "race"
    return "unknown"

def clean_data(df):
    if 'session_type' in df.columns:
        valid_sessions = ['qualifying', 'practice', 'race']
        df = df[df["session_type"].isin(valid_sessions)]
    else:
        logger.warning("❌ 'session_type' column not found. Skipping filtering.")

    df = df[df["lap_duration"] > 0]

    df.fillna({
        "air_temperature": 0,
        "humidity": 0,
        "pressure": 0,
        "rainfall": 0,
        "track_temperature": 0,
        "wind_direction": 0,
        "wind_speed": 0,
        "session_type": "Unknown",
    }, inplace=True)

    missing_after = df.isnull().sum().sum()
    if missing_after > 0:
        logger.warning(f"⚠️ Still missing values: {missing_after}")
    else:
        logger.info(f"✅ All missing values handled.")

    logger.info(f"🗑️ Cleaned data. Remaining rows: {len(df)}")
    return df

def merge_data(lap_data, weather_data, session_data):
    logger.info("🔗 Merging lap data with weather data...")

    if "session_key" not in lap_data.columns:
        logger.error("❌ 'session_key' not found in lap_data.")
        return lap_data

    logger.info("✅ 'session_key' found in lap_data.")

    # Drop duplicate or unnecessary columns
    weather_data = weather_data.drop(columns=["session_key"], errors="ignore")

    # Merge weather data using meeting_key
    merged_data = pd.merge(
        lap_data,
        weather_data,
        on="meeting_key",
        how="left"
    )

    logger.info("🔍 Columns after weather merge: %s", merged_data.columns.tolist())

    # Infer session_type if it's not already in session_data
    if "session_type" not in session_data.columns:
        logger.info("🧠 Inferring 'session_type' from 'session_name'")
        session_data["session_type"] = session_data["session_name"].apply(infer_session_type)

    if "session_key" in merged_data.columns:
        logger.info("🔗 Merging with session data using 'session_key'")
        merged_data = pd.merge(
            merged_data,
            session_data[["session_key", "session_name", "session_type"]],
            on="session_key",
            how="left"
        )
    else:
        logger.warning("⚠️ 'session_key' missing after weather merge. Skipping session data merge.")

    return merged_data

def generate_summary(df, missing_counts, total_missing):
    logger.info("📝 Generating summary...")

    with open(SUMMARY_FILE, "w") as summary_file:
        summary_file.write("F1 Data Cleaning Summary\n")
        summary_file.write(f"Total entries: {len(df)}\n")
        summary_file.write(f"Missing values before fill: {total_missing}\n")
        summary_file.write("Columns with missing values:\n")
        for col, count in missing_counts.items():
            if count > 0:
                summary_file.write(f" - {col}: {count} missing\n")
        summary_file.write("\nMissing values filled.\n")
        summary_file.write(f"Cleaned data saved to: {OUTPUT_CLEAN_DATA}\n")

def save_cleaned_data(df):
    logger.info("💾 Saving cleaned data...")
    df.to_csv(OUTPUT_CLEAN_DATA, index=False)
    logger.info(f"✅ Data saved to {OUTPUT_CLEAN_DATA}")

def main():
    os.makedirs(os.path.dirname(OUTPUT_CLEAN_DATA), exist_ok=True)

    logger.info("📥 Reading input data...")
    lap_data = pd.read_csv(INPUT_LAPS)
    session_data = pd.read_csv(INPUT_SESSIONS)
    weather_data = pd.read_csv(INPUT_WEATHER)

    merged_data = merge_data(lap_data, weather_data, session_data)

    cleaned_data = clean_data(merged_data)

    missing_counts = cleaned_data.isnull().sum()
    total_missing = missing_counts.sum()

    generate_summary(cleaned_data, missing_counts, total_missing)
    save_cleaned_data(cleaned_data)

if __name__ == "__main__":
    main()
