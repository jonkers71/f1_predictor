# File: data_pipeline/driver_fetcher.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import pandas as pd
import logging
from utils.logger import setup_logger

logger = setup_logger("DriverFetcher")

# Configuration
API_BASE_URL = "https://api.openf1.org/v1"
# Remove year filtering, fetch all drivers
OUTPUT_FILE = "data/processed/driver_map.csv"

def fetch_all_drivers():
    """Fetches all driver data from the OpenF1 API base endpoint."""
    drivers_url = f"{API_BASE_URL}/drivers"
    logger.info(f"Fetching all drivers from {drivers_url}")
    try:
        response = requests.get(drivers_url)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        drivers_data = response.json()
        logger.info(f"Successfully fetched {len(drivers_data)} total driver entries.")
        return drivers_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching driver data: {e}")
        return []
    except Exception as e:
        logger.exception(f"An unexpected error occurred fetching drivers: {e}")
        return []

def main():
    """Fetches all driver data and saves a mapping."""
    logger.info("--- Starting Driver Fetcher --- ")
    
    drivers_list = fetch_all_drivers()

    if not drivers_list:
        logger.error("No driver data fetched. Aborting.")
        return

    # Convert list of dicts to DataFrame
    all_drivers_df = pd.DataFrame(drivers_list)

    if all_drivers_df.empty:
        logger.error("Driver DataFrame is empty after fetching. Aborting.")
        return

    # Select relevant columns and remove duplicates based on driver_number
    # Keep the latest entry if duplicates exist (though less likely without year filter)
    if "driver_number" in all_drivers_df.columns and "full_name" in all_drivers_df.columns:
        driver_map_df = all_drivers_df[["driver_number", "full_name"]].drop_duplicates(subset=["driver_number"], keep="last")
    else:
        logger.error("Required columns (\"driver_number\", \"full_name\") not found in fetched data.")
        logger.debug(f"Available columns: {all_drivers_df.columns.tolist()}")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Save the mapping
    try:
        driver_map_df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Driver mapping saved successfully to {OUTPUT_FILE}. Shape: {driver_map_df.shape}")
    except Exception as e:
        logger.exception(f"Failed to save driver map to {OUTPUT_FILE}: {e}")

    logger.info("--- Driver Fetcher Finished --- ")

if __name__ == "__main__":
    main()

