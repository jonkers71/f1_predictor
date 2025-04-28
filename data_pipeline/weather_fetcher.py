# File: data/weather_fetcher.py
# Purpose: Fetch weather data for all F1 sessions using the OpenF1 API and save as structured CSV.
# Output: data/raw/weather_raw.csv

import os
import sys
import csv
import requests
from tqdm import tqdm

# Add root project directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import setup_logger
from config.settings import OPENF1_API_URL

logger = setup_logger("WeatherFetcher")

INPUT_SESSIONS = "data/raw/session_raw.csv"
OUTPUT_WEATHER_RAW = "data/raw/weather_raw.csv"

FIELDNAMES = [
    "meeting_key", "session_key", "date", "country",
    "air_temperature", "humidity", "pressure", "rainfall", 
    "track_temperature", "wind_direction", "wind_speed"
]

def fetch_weather_for_session(meeting_key, session_key):
    """Fetch weather data for a specific session."""
    url = f"{OPENF1_API_URL}weather?meeting_key={meeting_key}&session_key={session_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            logger.warning(f"No weather data for meeting {meeting_key}, session {session_key}")
            return None
        
        latest = sorted(data, key=lambda x: x['date'])[-1]  # Use most recent entry
        return latest
    except Exception as e:
        logger.error(f"❌ Error fetching weather for session {session_key}: {e}")
        return None

def load_sessions(csv_path):
    """Read session metadata from CSV."""
    sessions = []
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sessions.append((row["meeting_key"], row["session_key"], row["country"]))
    return sessions

def save_weather_data(weather_data, output_path):
    """Save weather data to CSV."""
    file_exists = os.path.exists(output_path)
    with open(output_path, mode="a", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(weather_data)

def main():
    os.makedirs(os.path.dirname(OUTPUT_WEATHER_RAW), exist_ok=True)
    sessions = load_sessions(INPUT_SESSIONS)
    weather_rows = []

    logger.info(f"🌤️ Starting weather fetch for {len(sessions)} sessions...")
    for meeting_key, session_key, country in tqdm(sessions, desc="Fetching weather"):
        weather = fetch_weather_for_session(meeting_key, session_key)
        if weather:
            weather_rows.append({
                "meeting_key": meeting_key,
                "session_key": session_key,
                "date": weather.get("date"),
                "country": country,
                "air_temperature": weather.get("air_temperature"),
                "humidity": weather.get("humidity"),
                "pressure": weather.get("pressure"),
                "rainfall": weather.get("rainfall"),
                "track_temperature": weather.get("track_temperature"),
                "wind_direction": weather.get("wind_direction"),
                "wind_speed": weather.get("wind_speed"),
            })
    
    save_weather_data(weather_rows, OUTPUT_WEATHER_RAW)
    logger.info(f"✅ Saved weather data for {len(weather_rows)} sessions.")

if __name__ == "__main__":
    main()
