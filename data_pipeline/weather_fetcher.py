# File: data_pipeline/weather_fetcher.py
# Purpose: Fetch weather data for all F1 sessions using the OpenF1 API and save as structured CSV.
# Output: data/raw/weather_raw.csv

import os
import sys
import csv
import requests
import time # For rate limiting
from tqdm import tqdm

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

# Rate Limiting and Retry Parameters (consistent with other fetchers)
MAX_RETRIES = 5
INITIAL_DELAY = 1 # seconds
BACKOFF_FACTOR = 2
REQUEST_DELAY = 0.5 # seconds delay between successful requests
REQUEST_TIMEOUT = 30 # seconds

def fetch_with_retry(url: str):
    """Fetches data from a URL with retry logic for temporary errors."""
    retries = 0
    delay = INITIAL_DELAY
    while retries < MAX_RETRIES:
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            time.sleep(REQUEST_DELAY) # Wait after a successful request
            return response.json()
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code in [429, 500, 502, 503, 504]:
                retries += 1
                if retries >= MAX_RETRIES:
                    logger.error(f"Max retries reached for {url}. Error: {e}")
                    raise # Re-raise the exception after max retries
                logger.warning(f"Request failed ({status_code}), retrying in {delay} seconds... ({retries}/{MAX_RETRIES}) URL: {url}")
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
            elif status_code == 404:
                 logger.debug(f"Resource not found (404) at {url}. Assuming no data.")
                 return [] # Return empty list for 404
            else:
                logger.error(f"HTTP Request failed with non-retryable error: {e}")
                raise
    logger.error(f"Failed to fetch {url} after {MAX_RETRIES} retries without raising final exception.")
    return None

def fetch_weather_for_session(meeting_key, session_key):
    """Fetch weather data for a specific session using retry logic."""
    url = f"{OPENF1_API_URL}weather?meeting_key={meeting_key}&session_key={session_key}"
    try:
        data = fetch_with_retry(url)
        if data is None: # Persistent error
            logger.error(f"❌ Failed to fetch weather for session {session_key} after retries.")
            return None
        if not data: # No data found (e.g., 404 or empty list from API)
            logger.warning(f"No weather data found for meeting {meeting_key}, session {session_key}")
            return None
        
        # Use most recent entry if multiple are returned
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
        with open(csv_path, mode='r', encoding='utf-8') as file: # Corrected quotes
            reader = csv.DictReader(file)
            required_cols = ["meeting_key", "session_key", "country"]
            if not all(col in reader.fieldnames for col in required_cols):
                logger.error(f"Missing required columns ({required_cols}) in {csv_path}")
                return []
            for row in reader:
                sessions.append((row["meeting_key"], row["session_key"], row["country"]))
        logger.info(f"Loaded {len(sessions)} sessions from {csv_path}")
        return sessions
    except Exception as e:
        logger.exception(f"Error loading sessions from {csv_path}: {e}")
        return []

def save_weather_data(weather_data, output_path):
    """Save weather data to CSV."""
    if not weather_data:
        logger.warning("No weather data to save.")
        return
        
    file_exists = os.path.exists(output_path)
    try:
        with open(output_path, mode="a", newline='', encoding='utf-8') as file: # Corrected quotes
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            if not file_exists or os.path.getsize(output_path) == 0:
                writer.writeheader()
            writer.writerows(weather_data)
        logger.info(f"Appended weather data for {len(weather_data)} sessions to {output_path}")
    except Exception as e:
        logger.exception(f"Error saving weather data to {output_path}: {e}")

def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_WEATHER_RAW), exist_ok=True)
    
    # Clear existing weather file if starting fresh (optional, consider adding menu like lap_collector)
    # if os.path.exists(OUTPUT_WEATHER_RAW):
    #     logger.warning(f"Removing existing weather file: {OUTPUT_WEATHER_RAW}")
    #     os.remove(OUTPUT_WEATHER_RAW)
        
    sessions = load_sessions(INPUT_SESSIONS)
    if not sessions:
        logger.error("No sessions loaded, cannot fetch weather. Ensure session_raw.csv exists and is valid.")
        return
        
    weather_rows = []

    logger.info(f"🌤️ Starting weather fetch for {len(sessions)} sessions...")
    for meeting_key, session_key, country in tqdm(sessions, desc="Fetching weather"):
        weather = fetch_weather_for_session(meeting_key, session_key)
        if weather:
            weather_rows.append({
                "meeting_key": meeting_key,
                "session_key": session_key,
                "date": weather.get("date"),
                "country": country, # Get country from session data
                "air_temperature": weather.get("air_temperature"),
                "humidity": weather.get("humidity"),
                "pressure": weather.get("pressure"),
                "rainfall": weather.get("rainfall"),
                "track_temperature": weather.get("track_temperature"),
                "wind_direction": weather.get("wind_direction"),
                "wind_speed": weather.get("wind_speed"),
            })
        # Add a small delay between session weather requests
        time.sleep(REQUEST_DELAY / 2) # Shorter delay as weather calls might be less frequent
    
    save_weather_data(weather_rows, OUTPUT_WEATHER_RAW)
    logger.info(f"✅ Finished weather fetch. Attempted: {len(sessions)}, Saved: {len(weather_rows)}.")

if __name__ == "__main__":
    main()

