# File: data_pipeline/weather_fetcher.py
# Purpose: Fetch weather data for F1 sessions using the OpenF1 API, appending only missing sessions.
# Output: data/raw/weather_raw.csv

import os
import sys
import csv
import requests
import time  # For rate limiting
from tqdm import tqdm
import pandas as pd

# Add root project directory to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.logger import setup_logger
from config.settings import OPENF1_API_URL

logger = setup_logger("WeatherFetcher")

# Use absolute paths
INPUT_SESSIONS = os.path.join(PROJECT_ROOT, "data/raw/session_raw.csv")
OUTPUT_WEATHER_RAW = os.path.join(PROJECT_ROOT, "data/raw/weather_raw.csv")

FIELDNAMES = [
    "meeting_key", "session_key", "date", "country",
    "air_temperature", "humidity", "pressure", "rainfall",
    "track_temperature", "wind_direction", "wind_speed"
]

# Rate Limiting and Retry Parameters
MAX_RETRIES = 5
INITIAL_DELAY = 1  # seconds
BACKOFF_FACTOR = 2
REQUEST_DELAY = 0.5  # seconds delay between successful requests
REQUEST_TIMEOUT = 30  # seconds

def fetch_with_retry(url: str):
    """Fetches data from a URL with retry logic for temporary errors."""
    retries = 0
    delay = INITIAL_DELAY
    while retries < MAX_RETRIES:
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return response.json()
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code in [429, 500, 502, 503, 504]:
                retries += 1
                if retries >= MAX_RETRIES:
                    logger.error(f"Max retries reached for {url}. Error: {e}")
                    raise
                logger.warning(f"Request failed ({status_code}), retrying in {delay} seconds... ({retries}/{MAX_RETRIES}) URL: {url}")
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
            elif status_code == 404:
                logger.debug(f"Resource not found (404) at {url}. Assuming no data.")
                return []
            else:
                logger.error(f"HTTP Request failed with non-retryable error: {e}")
                raise
    logger.error(f"Failed to fetch {url} after {MAX_RETRIES} retries.")
    return None

def fetch_weather_for_session(meeting_key, session_key):
    """Fetch weather data for a specific session using retry logic."""
    url = f"{OPENF1_API_URL}weather?meeting_key={meeting_key}&session_key={session_key}"
    try:
        data = fetch_with_retry(url)
        if data is None:
            logger.error(f"❌ Failed to fetch weather for session {session_key} after retries.")
            return None
        if not data:
            logger.warning(f"No weather data found for meeting {meeting_key}, session {session_key}")
            return None

        latest = sorted(data, key=lambda x: x.get("date", ""))[-1]
        return latest
    except Exception as e:
        logger.exception(f"❌ Unexpected error processing weather for session {session_key}: {e}")
        return None

def load_sessions(csv_path):
    """Read session metadata from CSV."""
    sessions = []
    if not os.path.exists(csv_path):
        logger.error(f"Session CSV file not found: {csv_path}")
        return []
    try:
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            required_cols = ["meeting_key", "session_key", "country"]
            if not all(col in reader.fieldnames for col in required_cols):
                logger.error(f"Missing required columns ({required_cols}) in {csv_path}")
                return []
            for row in reader:
                try:
                    meeting_key = int(row["meeting_key"])
                    session_key = int(row["session_key"])
                    sessions.append((meeting_key, session_key, row["country"]))
                except (ValueError, KeyError) as ve:
                    logger.warning(f"Skipping row due to invalid format or missing key: {row} - Error: {ve}")
        logger.info(f"Loaded {len(sessions)} sessions from {csv_path}")
        return sessions
    except Exception as e:
        logger.exception(f"Error loading sessions from {csv_path}: {e}")
        return []

def load_existing_weather_keys(csv_path):
    """Reads the existing weather CSV and returns a set of session_keys."""
    existing_keys = set()
    if not os.path.exists(csv_path):
        logger.info(f"Existing weather file not found at {csv_path}. Will fetch all.")
        return existing_keys
    try:
        df = pd.read_csv(csv_path, usecols=["session_key"], low_memory=False)
        existing_keys = set(df["session_key"].unique())
        logger.info(f"Found {len(existing_keys)} existing session keys in {csv_path}")
    except FileNotFoundError:
        logger.info(f"Existing weather file not found at {csv_path}. Will fetch all.")
    except pd.errors.EmptyDataError:
        logger.info(f"Existing weather file {csv_path} is empty. Will fetch all.")
    except KeyError:
        logger.warning(f"Column 'session_key' not found in {csv_path}. Assuming no existing keys.")
    except Exception as e:
        logger.exception(f"Error reading existing weather keys from {csv_path}: {e}")
        logger.warning("Proceeding as if no existing keys were found due to read error.")
        existing_keys = set()
    return existing_keys

def save_weather_data(weather_data, output_path):
    """Append weather data to CSV, writing header if needed."""
    if not weather_data:
        logger.info("No new weather data to save.")
        return

    file_exists = os.path.exists(output_path)
    is_empty = not file_exists or os.path.getsize(output_path) == 0

    try:
        with open(output_path, mode="a", newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            if is_empty:
                writer.writeheader()
                logger.info(f"Writing header to new/empty file: {output_path}")
            writer.writerows(weather_data)
        logger.info(f"Appended weather data for {len(weather_data)} sessions to {output_path}")
    except Exception as e:
        logger.exception(f"Error saving weather data to {output_path}: {e}")

def main():
    os.makedirs(os.path.dirname(OUTPUT_WEATHER_RAW), exist_ok=True)

    all_sessions = load_sessions(INPUT_SESSIONS)
    if not all_sessions:
        logger.error("No sessions loaded. Ensure session_raw.csv exists and is valid.")
        return

    existing_keys = load_existing_weather_keys(OUTPUT_WEATHER_RAW)
    missing_sessions = [
        (mk, sk, country) for mk, sk, country in all_sessions
        if sk not in existing_keys
    ]

    if not missing_sessions:
        logger.info("✅ No missing weather data to fetch. File is up-to-date.")
        return

    logger.info(f"🌤️ Found {len(existing_keys)} existing sessions. Fetching {len(missing_sessions)} missing sessions...")

    new_weather_rows = []
    for meeting_key, session_key, country in tqdm(missing_sessions, desc="Fetching missing weather"):
        weather = fetch_weather_for_session(meeting_key, session_key)
        if weather:
            new_weather_rows.append({
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
        time.sleep(REQUEST_DELAY / 2)

    save_weather_data(new_weather_rows, OUTPUT_WEATHER_RAW)
    logger.info(f"✅ Finished weather fetch. Attempted: {len(missing_sessions)}, Saved: {len(new_weather_rows)}.")

if __name__ == "__main__":
    main()
