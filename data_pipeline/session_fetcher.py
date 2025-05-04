# File: data_pipeline/session_fetcher.py
# Description: Fetches F1 session data for specified years from OpenF1 API and saves it to a raw CSV.

import sys
import os
import datetime # Needed to get current year
import time # Needed for rate limiting delays
import requests
import csv

# Add project root to the sys.path to run scripts individually
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from config.settings import OPENF1_API_URL
from utils.logger import setup_logger

logger = setup_logger("SessionFetcher")

# Define the range of years to fetch
START_YEAR = 2023
CURRENT_YEAR = datetime.datetime.now().year
YEARS_TO_FETCH = list(range(START_YEAR, CURRENT_YEAR + 1))

# Rate Limiting and Retry Parameters
MAX_RETRIES = 5
INITIAL_DELAY = 1 # seconds
BACKOFF_FACTOR = 2
REQUEST_DELAY = 0.5 # seconds delay between successful requests

def fetch_with_retry(url: str):
    """Fetches data from a URL with retry logic for temporary errors."""
    retries = 0
    delay = INITIAL_DELAY
    while retries < MAX_RETRIES:
        try:
            response = requests.get(url, timeout=30) # Added timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            time.sleep(REQUEST_DELAY) # Wait after a successful request
            return response.json()
        except requests.exceptions.RequestException as e:
            # Check for specific status codes that might indicate rate limiting or temporary server issues
            status_code = e.response.status_code if e.response is not None else None
            if status_code in [429, 500, 502, 503, 504]:
                retries += 1
                if retries >= MAX_RETRIES:
                    logger.error(f"Max retries reached for {url}. Error: {e}")
                    print(f"\n❌ Max retries reached for API request: {url}. See logs.") # User feedback
                    raise # Re-raise the exception after max retries
                logger.warning(f"Request failed ({status_code}), retrying in {delay} seconds... ({retries}/{MAX_RETRIES}) URL: {url}")
                print(f"   API request failed ({status_code}), retrying in {delay}s... ({retries}/{MAX_RETRIES})", end="\r") # User feedback
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
            else:
                # For other request errors (like connection errors, non-retryable HTTP errors), fail immediately
                logger.error(f"HTTP Request failed with non-retryable error: {e}")
                print(f"\n❌ Non-retryable API request error: {e}. See logs.") # User feedback
                raise
    return None # Should not be reached if raise is used properly

def fetch_sessions_for_year(year: int):
    """Fetches session data for a single specified year using retry logic."""
    # logger.info(f"Fetching sessions for year {year}...") # Moved print to main loop
    all_sessions_for_year = []
    try:
        url = f"{OPENF1_API_URL}sessions?year={year}"
        sessions_data = fetch_with_retry(url)
        
        if sessions_data is None:
             logger.error(f"Failed to fetch session data for {year} after multiple retries.")
             print(f"\n❌ Failed to fetch session data for {year} after retries.") # User feedback
             return []
             
        logger.info(f"Retrieved {len(sessions_data)} session entries for {year}.")
        print(f"   Retrieved {len(sessions_data)} session entries for {year}.") # User feedback

        # Extract relevant session info
        for session_entry in sessions_data:
            if all(k in session_entry for k in ["meeting_key", "country_name", "date_start", "session_key", "session_name"]):
                session = {
                    "meeting_key": session_entry["meeting_key"],
                    "country": session_entry["country_name"],
                    "date": session_entry["date_start"],
                    "session_key": session_entry["session_key"],
                    "session_name": session_entry["session_name"]
                }
                all_sessions_for_year.append(session)
            else:
                logger.warning(f"Skipping session entry due to missing keys: {session_entry.get('session_key', 'N/A')}")
        
        # logger.debug(f"First 3 sessions for {year}: {all_sessions_for_year[:3]}...") # Reduce verbosity
        return all_sessions_for_year

    except Exception as e:
        # Catch potential errors from fetch_with_retry or JSON processing
        logger.exception(f"Failed to fetch or process session data for {year}. Error: {e}")
        print(f"\n❌ Error processing data for year {year}: {e}") # User feedback
        return []

def save_sessions_to_csv(all_sessions, file_path="data/raw/session_raw.csv"):
    """ Saves the combined raw session data to session_raw.csv using absolute path. """
    if not all_sessions:
        logger.warning("No session data provided to save.")
        return
        
    # Construct absolute path based on project root
    absolute_file_path = os.path.join(PROJECT_ROOT, file_path)
    os.makedirs(os.path.dirname(absolute_file_path), exist_ok=True)
    
    try:
        print(f"\n💾 Saving {len(all_sessions)} sessions to {absolute_file_path}...") # User feedback
        with open(absolute_file_path, mode="w", newline="", encoding="utf-8") as file:
            fieldnames = ["meeting_key", "country", "date", "session_key", "session_name"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_sessions)
        logger.info(f"Combined session data saved to {absolute_file_path}. Total sessions: {len(all_sessions)}")
        print("   Save complete.") # User feedback
    except Exception as e:
        logger.exception(f"Failed to save session data to {absolute_file_path}.")
        print(f"\n❌ Error saving session data: {e}") # User feedback

def main():
    """Fetches session data for the defined range of years and saves it."""
    logger.info(f"--- Starting Session Fetcher for years: {YEARS_TO_FETCH} ---")
    print(f"\n🚀 Starting Session Fetcher for years: {YEARS_TO_FETCH}") # User feedback
    combined_sessions = []
    total_years = len(YEARS_TO_FETCH)
    
    for i, year in enumerate(YEARS_TO_FETCH):
        print(f"\n🔄 Fetching year {year} ({i+1}/{total_years})...") # User feedback: Progress per year
        sessions_for_year = fetch_sessions_for_year(year)
        if sessions_for_year:
            combined_sessions.extend(sessions_for_year)
        else:
            logger.warning(f"No sessions fetched for year {year}.")
            # User feedback already provided in fetch_sessions_for_year
        
        # Add a small delay between fetching different years, even if one failed
        if i < total_years - 1: # Don't delay after the last year
             time.sleep(REQUEST_DELAY * 2) 
            
    if combined_sessions:
        save_sessions_to_csv(combined_sessions)
        logger.info(f"Successfully fetched a total of {len(combined_sessions)} sessions across all years.")
        print(f"\n🏁 Session Fetcher Finished: Fetched {len(combined_sessions)} sessions total.") # User feedback
    else:
        logger.error("No sessions fetched for any year. Check API or configuration.")
        print("\n❌ Session Fetcher Finished: Failed to fetch any session data.") # User feedback
        
    # logger.info("--- Session Fetcher Finished ---") # Redundant with print above

if __name__ == "__main__":
    main()

