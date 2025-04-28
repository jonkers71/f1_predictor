import ssl
import requests
import time
import csv
import os
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
OUTPUT_PATH = "data/raw/lap_data.csv"
SESSION_CSV = "data/raw/session_raw.csv"
DONE_FILE = "data/raw/done_sessions.txt"
FAILED_FILE = "data/raw/failed_sessions.txt"
MAX_LAPS = 20
MAX_RETRIES = 3
INITIAL_DELAY = 5
BACKOFF_FACTOR = 2
MAX_DELAY = 60
START_CONCURRENT_REQUESTS = 2  # Starting with 1 concurrent request
MAX_CONCURRENT_REQUESTS = 5    # Maximum concurrent requests to attempt before backing off
CONCURRENT_INCREASE_STEP = 1   # Increase by 1 concurrent request after each successful session

os.makedirs("data/raw", exist_ok=True)
os.system('clear')

# Set up logging to the existing debug log file
LOG_FILE = "logs/debug.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Headers for the API request to simulate a browser user-agent.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (LapCollector/1.0)",
    "Accept": "application/json"
}

# Fields to be saved to the CSV file
FIELDNAMES = [
    "date_start", "driver_number", "duration_sector_1", "duration_sector_2", "duration_sector_3",
    "i1_speed", "i2_speed", "is_pit_out_lap", "lap_duration", "lap_number", "meeting_key",
    "segments_sector_1", "segments_sector_2", "segments_sector_3", "session_key", "st_speed"
]

def display_menu():
    print("\n=== F1 Lap Collector Menu ===")
    print("1. Add new sessions (continue)")
    print("2. Run again from scratch (clear done_sessions.txt)")
    return input("Select an option (1 or 2): ").strip()

def load_done_sessions(clear=False):
    if clear and os.path.exists(DONE_FILE):
        os.remove(DONE_FILE)
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE, "r") as f:
            return set(map(int, f.read().splitlines()))
    return set()

def clean_lap_data(lap_data):
    return [{k: row.get(k, None) for k in FIELDNAMES} for row in lap_data]

def save_lap_data_to_csv(lap_data):
    file_exists = os.path.exists(OUTPUT_PATH)
    with open(OUTPUT_PATH, mode="a", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not file_exists:
            logging.info("Writing headers to CSV...")
            writer.writeheader()  # Ensure headers are written when the file is created
        writer.writerows(clean_lap_data(lap_data))

# SSLAdapter to force TLS 1.2 (without specifying a cipher suite)
class SSLAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

# Function to fetch lap data with retries and backoff
def fetch_lap(session, session_key, lap_number):
    url = f"https://api.openf1.org/v1/laps?session_key={session_key}&lap_number={lap_number}"
    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, headers=HEADERS, timeout=20)
            if response.status_code == 429:
                logging.warning(f"⚠️ Rate limit for session {session_key}, lap {lap_number}, retrying in {delay}s...")
            elif response.status_code >= 500:
                logging.warning(f"⚠️ Server error {response.status_code} for session {session_key}, lap {lap_number}, retrying in {delay}s...")
            elif response.status_code != 200:
                logging.error(f"❌ Client error {response.status_code} for session {session_key}, lap {lap_number}")
                return None
            else:
                data = response.json()
                return data if data else []
        except Exception as e:
            logging.error(f"❌ Exception on session {session_key}, lap {lap_number}: {e}")
        time.sleep(delay)
        delay = min(delay * BACKOFF_FACTOR, MAX_DELAY)
    logging.error(f"❌ Failed session {session_key}, lap {lap_number} after {MAX_RETRIES} retries.")
    return None

# Function to fetch all laps for a session
def fetch_laps_for_session(session_key, session):
    all_data = []
    failed_count = 0

    for lap_number in range(1, MAX_LAPS + 1):
        data = fetch_lap(session, session_key, lap_number)
        if data is not None:
            all_data.extend(data)
        else:
            failed_count += 1

    if all_data:
        save_lap_data_to_csv(all_data)
        with open(DONE_FILE, "a") as f:
            f.write(f"{session_key}\n")
        logging.info(f"✅ Saved {len(all_data)} lap entries for session {session_key}.")
    else:
        logging.error(f"❌ Marking session {session_key} as failed.")
        with open(FAILED_FILE, "a") as f:
            f.write(f"{session_key}\n")

# Function to fetch session keys from CSV
def fetch_sessions_from_csv(csv_path=SESSION_CSV):
    try:
        with open(csv_path, mode="r") as file:
            reader = csv.DictReader(file)
            sessions = [int(row["session_key"]) for row in reader]
        logging.info(f"Found {len(sessions)} sessions in CSV.")
        return sessions
    except Exception as e:
        logging.error(f"❌ Failed to read session CSV: {e}")
        return []

# Function to fetch laps concurrently for multiple sessions
def fetch_laps_concurrently(session_keys, session, max_concurrent_requests=START_CONCURRENT_REQUESTS):
    all_data = []
    failed_sessions = []
    successful_sessions = 0

    # Set up tqdm for a nice progress bar
    with tqdm(total=len(session_keys), desc="Fetching laps", unit="session") as pbar:
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            future_to_session = {executor.submit(fetch_laps_for_session, session_key, session): session_key for session_key in session_keys}
            for future in as_completed(future_to_session):
                session_key = future_to_session[future]
                try:
                    future.result()
                    successful_sessions += 1
                    logging.info(f"✅ Successfully processed session {session_key}.")
                except Exception as e:
                    logging.error(f"❌ Error with session {session_key}: {e}")
                    failed_sessions.append(session_key)
                pbar.update(1)  # Update the progress bar

    return successful_sessions, failed_sessions

# Main function to execute the script
if __name__ == "__main__":
    # Initialize the session with SSLAdapter
    session = requests.Session()
    session.mount('https://', SSLAdapter())

    # Display the menu and decide whether to clear the done_sessions.txt
    clear = (display_menu() == "2")
    done_sessions = load_done_sessions(clear)
    session_keys = fetch_sessions_from_csv()

    if not session_keys:
        logging.error("No session keys found in CSV.")
    else:
        max_concurrent_requests = START_CONCURRENT_REQUESTS
        while True:
            logging.info(f"\nTrying with {max_concurrent_requests} concurrent requests...")  # Log the concurrent request attempt
            
            # Fetch laps concurrently for the sessions with the current max_concurrent_requests
            successful_sessions, failed_sessions = fetch_laps_concurrently(session_keys, session, max_concurrent_requests)

            # If we encountered errors, back off and reduce the number of concurrent requests
            if failed_sessions:
                logging.warning(f"❌ Encountered errors with {len(failed_sessions)} sessions. Reducing concurrent requests...")
                # Back off by halving the number of concurrent requests (minimum 1)
                max_concurrent_requests = max(START_CONCURRENT_REQUESTS, max_concurrent_requests // 2)
                logging.info(f"Max concurrent requests reduced to {max_concurrent_requests}")
                logging.info(f"✅ Successful sessions: {successful_sessions}")
                logging.info(f"❌ Failed sessions: {len(failed_sessions)}")
                time.sleep(10)  # Wait before retrying
            else:
                # Increase concurrent requests after successful batch (but don’t exceed MAX_CONCURRENT_REQUESTS)
                max_concurrent_requests = min(max_concurrent_requests + CONCURRENT_INCREASE_STEP, MAX_CONCURRENT_REQUESTS)
                logging.info(f"✅ All sessions processed successfully. Max concurrent requests increased to {max_concurrent_requests}")
                break
