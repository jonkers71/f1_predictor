import os

# Root project directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths for modules
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
SESSION_DATA_PATH = os.path.join(RAW_DATA_PATH, "session_raw.csv")
WEATHER_DATA_PATH = os.path.join(RAW_DATA_PATH, "weather_raw.csv")
LAP_DATA_PATH = os.path.join(RAW_DATA_PATH, "lap_data.csv")

# OpenF1 API URL
OPENF1_API_URL = "https://api.openf1.org/"

# Optional: Configure paths for logging or other components
LOGGING_PATH = os.path.join(PROJECT_ROOT, "logs")

# Optional: Configure other settings
LAPS_PER_SESSION = 20
RETRY_BASE_DELAY = 5
MAX_RETRIES = 5

# Optional: Add extra paths for import resolution in IDEs
EXTRA_PYTHON_PATHS = [DATA_PATH]
