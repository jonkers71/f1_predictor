# File: data_pipeline/lap_collector.py
# Description: Fetches F1 lap data for sessions listed in session_raw.csv using a single call per session.

import requests
import time
import csv
import os
import logging
import sys
import threading # For file lock

# Add project root to the sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logger import setup_logger
from config.settings import OPENF1_API_URL

logger = setup_logger("LapCollector")

# --- Configuration --- 
# Use absolute paths based on project root
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/raw/lap_data.csv")
SESSION_CSV = os.path.join(PROJECT_ROOT, "data/raw/session_raw.csv")
DONE_FILE = os.path.join(PROJECT_ROOT, "data/raw/done_sessions.txt")
FAILED_FILE = os.path.join(PROJECT_ROOT, "data/raw/failed_sessions.txt")

# API Fetching Parameters
MAX_RETRIES = 5
INITIAL_DELAY = 2
BACKOFF_FACTOR = 2
REQUEST_DELAY = 1
REQUEST_TIMEOUT = 120

# Concurrency Control
START_CONCURRENT_REQUESTS = 1
MAX_CONCURRENT_REQUESTS = 3

# Ensure data/raw directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Headers for the API request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (LapCollector/1.20)", # Updated version
    "Accept": "application/json"
}

# Fields to be saved to the CSV file
FIELDNAMES = [
    "date_start", "driver_number", "duration_sector_1", "duration_sector_2", "duration_sector_3",
    "i1_speed", "i2_speed", "is_pit_out_lap", "lap_duration", "lap_number", "meeting_key",
    "segments_sector_1", "segments_sector_2", "segments_sector_3", "session_key", "st_speed"
]

# Lock for synchronizing access to FAILED_FILE
failed_file_lock = threading.Lock()

# --- End Configuration ---

def fetch_with_retry(session_requests, url: str):
    """Fetches data from a URL with retry logic using a requests.Session."""
    retries = 0
    delay = INITIAL_DELAY
    while retries < MAX_RETRIES:
        try:
            response = session_requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return response.json()
        except requests.exceptions.Timeout:
            retries += 1
            logger.warning(f"Request timed out for {url}, retrying in {delay} seconds... ({retries}/{MAX_RETRIES})")
            # Print warning for visibility if logging is off
            print(f"\r   WARN: Request timed out for {url}, retrying in {delay}s... ({retries}/{MAX_RETRIES})", end='', flush=True)
            if retries >= MAX_RETRIES:
                logger.error(f"Max retries reached due to timeout for {url}.")
                print(f"\n   ERROR: Max retries reached due to timeout for {url}.") # Newline for final error
                raise
            time.sleep(delay)
            delay *= BACKOFF_FACTOR
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code in [429, 500, 502, 503, 504]:
                retries += 1
                logger.warning(f"Request failed ({status_code}), retrying in {delay} seconds... ({retries}/{MAX_RETRIES}) URL: {url}")
                # Print warning for visibility if logging is off
                if retries >= MAX_RETRIES:
                    logger.error(f"Max retries reached for {url}. Error: {e}")
                    print(f"\n   ERROR: Max retries reached for {url}. Error: {e}") # Newline for final error
                    raise
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
            elif status_code == 404:
                 logger.debug(f"Resource not found (404) at {url}. Assuming no data.")
                 return []
            else:
                logger.error(f"HTTP Request failed with non-retryable error: {e}")
                print(f"\n   ERROR: Non-retryable HTTP Request error: {e}") # Newline for final error
                raise
    logger.error(f"Failed to fetch {url} after {MAX_RETRIES} retries without raising final exception.")
    print(f"\n   ERROR: Failed to fetch {url} after {MAX_RETRIES} retries.") # Newline for final error
    return None

def display_menu():
    """Asks user how to handle previously completed sessions."""
    print("\n=== F1 Lap Collector Menu ===")
    print("1. Add new sessions (continue from where left off using done_sessions.txt)")
    print("2. Run again from scratch (clear done_sessions.txt and potentially lap_data.csv)")
    choice = input("Select an option (1 or 2): ").strip()
    if choice == "2":
        confirm = input(f"This will clear {DONE_FILE} and start fresh. Optionally clear {OUTPUT_PATH}? (yes/no): ").strip().lower()
        if confirm == "yes":
            if os.path.exists(OUTPUT_PATH):
                try:
                    logger.warning(f"User chose to clear existing lap data file: {OUTPUT_PATH}")
                    os.remove(OUTPUT_PATH)
                except OSError as e:
                    logger.error(f"Failed to remove {OUTPUT_PATH}: {e}")
        return True
    return False

def load_done_sessions(clear=False):
    """Loads the set of already processed session keys, optionally clearing the file first."""
    if clear and os.path.exists(DONE_FILE):
        try:
            logger.info(f"Clearing done sessions file: {DONE_FILE}")
            os.remove(DONE_FILE)
        except OSError as e:
             logger.error(f"Failed to remove {DONE_FILE}: {e}")
    if os.path.exists(DONE_FILE):
        try:
            with open(DONE_FILE, "r") as f:
                done_keys = set(int(line.strip()) for line in f if line.strip().isdigit())
                logger.info(f"Loaded {len(done_keys)} completed session keys from {DONE_FILE}")
                return done_keys
        except Exception as e:
            logger.exception(f"Error loading done sessions file {DONE_FILE}: {e}. Starting fresh.")
            return set()
    return set()

def clean_lap_data(lap_data):
    """Ensures only specified FIELDNAMES are kept for each lap entry."""
    return [{k: row.get(k, None) for k in FIELDNAMES} for row in lap_data]

def save_lap_data_to_csv(lap_data):
    """Appends cleaned lap data to the output CSV file."""
    if not lap_data:
        return
    file_exists = os.path.exists(OUTPUT_PATH)
    try:
        with open(OUTPUT_PATH, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            if not file_exists or os.path.getsize(OUTPUT_PATH) == 0:
                logger.info(f"Writing headers to new/empty CSV: {OUTPUT_PATH}")
                writer.writeheader()
            writer.writerows(clean_lap_data(lap_data))
    except Exception as e:
        logger.exception(f"Error writing to CSV file {OUTPUT_PATH}: {e}")
        print(f"\n   ERROR: Failed to write lap data to CSV {OUTPUT_PATH}: {e}") # Newline for error

def remove_from_failed_sessions(session_key):
    """Removes a session key from the failed sessions file if it exists."""
    with failed_file_lock: # Ensure only one thread modifies the file at a time
        if not os.path.exists(FAILED_FILE):
            return # Nothing to remove if the file doesn\t exist
        
        try:
            with open(FAILED_FILE, "r") as f:
                failed_keys = [line.strip() for line in f if line.strip()]
            
            str_session_key = str(session_key)
            if str_session_key in failed_keys:
                updated_failed_keys = [key for key in failed_keys if key != str_session_key]
                
                # Rewrite the file with the updated list
                with open(FAILED_FILE, "w") as f:
                    for key in updated_failed_keys:
                        f.write(f"{key}\n")
                logger.info(f"Removed successfully processed session {session_key} from {FAILED_FILE}")
                
        except Exception as e:
            logger.exception(f"Error updating {FAILED_FILE} for session {session_key}: {e}")

def fetch_laps_for_session(session_key, session_requests):
    """Fetches all lap data for a given session_key using a single API call."""
    url = f"{OPENF1_API_URL}laps?session_key={session_key}"
    try:
        all_session_lap_data = fetch_with_retry(session_requests, url)
        if all_session_lap_data is None:
            logger.error(f"Persistent failure fetching laps for session {session_key}. Marking session as failed.")
            with failed_file_lock:
                with open(FAILED_FILE, "a") as f:
                    f.write(f"{session_key}\n")
            return False
        elif not all_session_lap_data:
            logger.info(f"No lap data found for session {session_key}. Marking as done.")
            with open(DONE_FILE, "a") as f:
                 f.write(f"{session_key}\n")
            remove_from_failed_sessions(session_key) # Clean up if it previously failed
            return True
        else:
            save_lap_data_to_csv(all_session_lap_data)
            with open(DONE_FILE, "a") as f:
                f.write(f"{session_key}\n")
            logger.info(f"✅ Saved {len(all_session_lap_data)} lap entries for session {session_key}.")
            remove_from_failed_sessions(session_key) # Clean up if it previously failed
            return True
    except Exception as e:
        logger.exception(f"Exception processing session {session_key}: {e}")
        print(f"\n   ERROR: Exception processing session {session_key}: {e}") # Newline for error
        with failed_file_lock:
            with open(FAILED_FILE, "a") as f:
                 f.write(f"{session_key}\n")
        return False

def fetch_sessions_from_csv(csv_path=SESSION_CSV):
    """Reads session keys from the session_raw.csv file."""
    sessions_to_process = []
    if not os.path.exists(csv_path):
        logger.error(f"Session CSV file not found: {csv_path}")
        return []
    try:
        with open(csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if "session_key" not in reader.fieldnames:
                logger.error(f"Column \"session_key\" not found in {csv_path}")
                return []
            for row in reader:
                try:
                    sessions_to_process.append(int(row["session_key"]))
                except (ValueError, TypeError):
                    logger.warning(f"Skipping invalid session_key in row: {row}")
        logger.info(f"Found {len(sessions_to_process)} sessions in {csv_path}.")
        return sessions_to_process
    except Exception as e:
        logger.exception(f"Failed to read session CSV {csv_path}: {e}")
        return []

def fetch_laps_concurrently(sessions_to_process, session_requests, max_concurrent_requests):
    """Manages concurrent fetching of lap data for a list of session keys with single-line progress."""
    failed_sessions_list = []
    successful_sessions_count = 0
    total_sessions = len(sessions_to_process)
    processed_count = 0

    print(f"Starting concurrent fetch for {total_sessions} sessions...", flush=True)
    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        future_to_session = {executor.submit(fetch_laps_for_session, sk, session_requests): sk for sk in sessions_to_process}
        
        for future in as_completed(future_to_session):
            session_key = future_to_session[future]
            processed_count += 1
            try:
                success = future.result()
                if success:
                    successful_sessions_count += 1
                else:
                    # Check if it was already marked failed within fetch_laps_for_session
                    # This avoids double-adding if fetch_laps_for_session logged the failure
                    # but the future itself raised an exception (less likely but possible)
                    if session_key not in failed_sessions_list: 
                        failed_sessions_list.append(session_key)
            except Exception as e:
                logger.exception(f"Critical exception processing future for session {session_key}: {e}")
                print(f"\n   ERROR: Critical exception processing future for session {session_key}: {e}") # Newline for error
                if session_key not in failed_sessions_list:
                    failed_sessions_list.append(session_key)
                    # Also log to file here if the exception prevented fetch_laps_for_session from doing so
                    with failed_file_lock:
                        with open(FAILED_FILE, "a") as f:
                            f.write(f"{session_key}\n")
            finally:
                # Update progress on the same line using carriage return
                progress_percent = (processed_count / total_sessions) * 100 if total_sessions > 0 else 0
                # Ensure the line is cleared with spaces at the end
                print(f"\r Progress: {processed_count}/{total_sessions} sessions ({progress_percent:.1f}%) | Success: {successful_sessions_count} | Failed: {len(failed_sessions_list)} ", end='', flush=True)
                
    print() # Print a newline after the loop finishes
    return successful_sessions_count, failed_sessions_list

# Main execution block
if __name__ == "__main__":
    logger.info("--- Starting Lap Collector (v1.20 - Failed Session Cleanup) --- ")
    print("--- Starting Lap Collector (v1.20) --- ", flush=True)
    session_requests = requests.Session()

    clear_done = display_menu()
    done_sessions = load_done_sessions(clear_done)
    all_session_keys = fetch_sessions_from_csv()

    if not all_session_keys:
        logger.error("No session keys found in CSV. Exiting.")
        print("ERROR: No session keys found in CSV. Exiting.", flush=True)
    else:
        sessions_to_fetch = [sk for sk in all_session_keys if sk not in done_sessions]
        logger.info(f"Total sessions in CSV: {len(all_session_keys)}")
        logger.info(f"Sessions already completed: {len(done_sessions)}")
        logger.info(f"Sessions remaining to fetch: {len(sessions_to_fetch)}")
        print(f"Total sessions in CSV: {len(all_session_keys)}", flush=True)
        print(f"Sessions already completed: {len(done_sessions)}", flush=True)
        print(f"Sessions remaining to fetch: {len(sessions_to_fetch)}", flush=True)

        if not sessions_to_fetch:
            logger.info("No new sessions to fetch.")
            print("INFO: No new sessions to fetch.", flush=True)
        else:
            current_concurrency = START_CONCURRENT_REQUESTS
            logger.info(f"Starting fetch with {current_concurrency} concurrent workers (max {MAX_CONCURRENT_REQUESTS})...")
            print(f"Starting fetch with {current_concurrency} concurrent workers (max {MAX_CONCURRENT_REQUESTS})...", flush=True)
            
            successful_count, failed_list = fetch_laps_concurrently(sessions_to_fetch, session_requests, current_concurrency)

            logger.info("--- Fetching Round Complete ---")
            print("--- Fetching Round Complete ---", flush=True)
            logger.info(f"Successfully processed sessions in this run: {successful_count}")
            print(f"Successfully processed sessions in this run: {successful_count}", flush=True)
            # Note: failed_list here might contain sessions that failed *during this run*
            # The remove_from_failed_sessions function cleans up *previously* failed ones that succeeded now.
            final_failed_count = 0
            if os.path.exists(FAILED_FILE):
                try:
                    with open(FAILED_FILE, "r") as f:
                        final_failed_count = len([line for line in f if line.strip()])
                except Exception as e:
                    logger.warning(f"Could not read final count from {FAILED_FILE}: {e}")

            logger.info(f"Total sessions currently marked as failed: {final_failed_count}")
            print(f"Total sessions currently marked as failed: {final_failed_count}", flush=True)
            if final_failed_count > 0:
                logger.warning(f"Check {FAILED_FILE} for list of currently failed sessions.")
                print(f"WARNING: Check {FAILED_FILE} for list of currently failed sessions.", flush=True)

    logger.info("--- Lap Collector Finished --- ")
    print("--- Lap Collector Finished --- ", flush=True)

