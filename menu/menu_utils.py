# File: menu/menu_utils.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import math
from utils.logger import setup_logger

logger = setup_logger("MenuUtils")

def load_driver_map(project_root, path="data/processed/driver_map.csv"):
    """Loads the driver number to name mapping from CSV relative to project root."""
    full_path = os.path.join(project_root, path)
    try:
        driver_df = pd.read_csv(full_path)
        # Create a dictionary for quick lookup: driver_number -> full_name
        driver_map = pd.Series(driver_df.full_name.values, index=driver_df.driver_number).to_dict()
        logger.info(f"Successfully loaded driver map from {full_path}. {len(driver_map)} drivers found.")
        return driver_map
    except FileNotFoundError:
        logger.warning(f"Driver map file not found at {full_path}. Driver names will not be displayed.")
        return {}
    except Exception as e:
        logger.exception(f"Error loading driver map from {full_path}: {e}")
        return {}

def format_lap_time(seconds):
    """Formats lap time in seconds to M:SS.fff format."""
    if seconds is None or not isinstance(seconds, (int, float)) or math.isnan(seconds):
        return "N/A"
    try:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}:{remaining_seconds:06.3f}"
    except Exception as e:
        logger.error(f"Error formatting time {seconds}: {e}")
        return "Error"

