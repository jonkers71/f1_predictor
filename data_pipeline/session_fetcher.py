# File: data_pipeline/session_fetcher.py
# Description: Fetches F1 session data (2025 season by default) from OpenF1 API and saves it to a raw CSV.

import sys
import os
# Add project root to the sys.path to run scripts individually
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import requests
import csv
from config.settings import OPENF1_API_URL, DEFAULT_YEAR
from utils.logger import setup_logger  # Now we can import the logger

logger = setup_logger("SessionFetcher")

def fetch_sessions(year: int = DEFAULT_YEAR):
    logger.info(f"Fetching sessions for year {year}...")

    try:
        # Updated API endpoint to fetch sessions directly
        url = f"{OPENF1_API_URL}sessions?year={year}"
        response = requests.get(url)
        response.raise_for_status()

        meetings = response.json()

        # Debug: Print the raw meetings data to understand its structure
        logger.debug(f"Raw meetings data: {meetings}")

        logger.info(f"✅ Retrieved {len(meetings)} meetings.")

        # Extract session info (session_key, session_name) for each meeting
        sessions = []
        for meeting in meetings:
            if meeting.get("meeting_key") and meeting.get("country_name"):
                session = {
                    "meeting_key": meeting["meeting_key"],
                    "country": meeting["country_name"],
                    "date": meeting["date_start"],
                    "session_key": meeting["session_key"],
                    "session_name": meeting["session_name"]
                }
                sessions.append(session)

        # Save session data to CSV
        save_sessions_to_csv(sessions)

        logger.debug(f"Session data: {sessions[:3]}...")  # Show preview
        return sessions

    except Exception as e:
        logger.exception("❌ Failed to fetch session data.")
        return []


def save_sessions_to_csv(sessions):
    """ Saves the raw session data to session_raw.csv """
    os.makedirs("data/raw", exist_ok=True)
    file_path = "data/raw/session_raw.csv"
    
    # Writing to CSV
    try:
        with open(file_path, mode="w", newline='') as file:
            fieldnames = ["meeting_key", "country", "date", "session_key", "session_name"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sessions)
        logger.info(f"✅ Session data saved to {file_path}.")
    except Exception as e:
        logger.exception(f"❌ Failed to save session data to {file_path}.")
        
if __name__ == "__main__":
    sessions = fetch_sessions()
    print(f"Fetched {len(sessions)} sessions.")
    for s in sessions[:3]:
        print(s)
