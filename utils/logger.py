# File: utils/logger.py
import logging
import os
import sys

# Add project root to sys.path to allow importing config.settings
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

try:
    from config.settings import LOGGING_ENABLED
except ImportError:
    # Default to True if settings file or variable is missing
    LOGGING_ENABLED = True 

# Ensure logs directory exists
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, "debug.log")

def setup_logger(name: str):
    logger = logging.getLogger(name)
    
    # Prevent adding handlers multiple times if called repeatedly
    if logger.hasHandlers():
        # Check if logging state needs update (e.g., if LOGGING_ENABLED changed)
        # For simplicity, we assume logger is set up correctly on first call per script run.
        # A more complex setup might involve removing/adding handlers based on state change.
        return logger 

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if LOGGING_ENABLED:
        logger.setLevel(logging.DEBUG) # Enable all levels if logging is on

        # File Handler (always DEBUG level when enabled)
        fh = logging.FileHandler(LOG_FILE_PATH)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console Handler (INFO level when enabled)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO) # Console shows INFO and above
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # logger.info(f"Logging enabled for {name}. Level: DEBUG (File), INFO (Console)")
    else:
        # If logging is disabled, set level high and add only a NullHandler
        logger.setLevel(logging.CRITICAL + 1) # Effectively disable logging
        logger.addHandler(logging.NullHandler())
        # logger.info(f"Logging disabled for {name}.") # This won't be logged

    return logger

def update_logging_config(enabled: bool):
    """Updates the LOGGING_ENABLED setting in config/settings.py."""
    settings_path = os.path.join(PROJECT_ROOT, "config/settings.py")
    try:
        with open(settings_path, "r") as f:
            lines = f.readlines()
        
        updated = False
        with open(settings_path, "w") as f:
            for line in lines:
                if line.strip().startswith("LOGGING_ENABLED"):
                    f.write(f"LOGGING_ENABLED = {enabled}\n")
                    updated = True
                else:
                    f.write(line)
            if not updated:
                 # Append if not found (shouldn't happen with previous step)
                 f.write(f"\nLOGGING_ENABLED = {enabled}\n")
        return True
    except Exception as e:
        print(f"Error updating logging config: {e}")
        return False

