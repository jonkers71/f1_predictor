# File: data_pipeline/dataset_builder.py

import os
import sys

# Add project root to the sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
import requests # Added for Ergast schedule fetching
import time     # Added for Ergast schedule fetching delay
from utils.logger import setup_logger

logger = setup_logger("DatasetBuilder")

# Define paths using absolute paths based on project root
INPUT_LAPS = os.path.join(PROJECT_ROOT, "data/raw/lap_data.csv")
INPUT_SESSIONS = os.path.join(PROJECT_ROOT, "data/raw/session_raw.csv")
INPUT_WEATHER = os.path.join(PROJECT_ROOT, "data/raw/weather_raw.csv")
INPUT_ERGAST_RESULTS = os.path.join(PROJECT_ROOT, "data/raw/ergast_race_results.csv") # Added Ergast results path
OUTPUT_CLEAN_DATA = os.path.join(PROJECT_ROOT, "data/processed/cleaned_f1_data.csv")
SUMMARY_FILE = os.path.join(PROJECT_ROOT, "data/processed/summary.txt")

# --- Feature Engineering Configuration ---
ROLLING_LAP_WINDOW = 3 # Window size for rolling average lap time
# --- End Configuration ---

# --- Ergast API Configuration ---
ERGAST_BASE_URL = "http://ergast.com/api/f1"
ERGAST_REQUEST_DELAY = 0.2 # Seconds to wait between requests
# --- End Configuration ---

def fetch_ergast_schedule(year):
    """Fetches the race schedule for a given year from Ergast."""
    url = f"{ERGAST_BASE_URL}/{year}.json"
    schedule_data = []
    try:
        logger.debug(f"Fetching Ergast schedule for {year} from {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        for race in races:
            schedule_data.append({
                "year": int(year),
                "round": int(race.get("round")),
                "race_name": race.get("raceName"),
                "circuit_id": race.get("Circuit", {}).get("circuitId"),
                "date": race.get("date") # Keep date for potential matching
            })
        logger.debug(f"Fetched {len(schedule_data)} schedule entries for {year}.")
        return pd.DataFrame(schedule_data)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Ergast schedule for {year}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        logger.error(f"Unexpected error fetching Ergast schedule for {year}: {e}")
        return pd.DataFrame()

def infer_session_type(name):
    """Infer session type based on session name keywords."""
    name = str(name).lower()
    if "qualifying" in name:
        if "sprint" in name:
            return "sprint_qualifying"
        return "qualifying"
    elif "practice" in name or name.startswith("fp") or "day" in name:
        return "practice"
    elif "race" in name:
        if "sprint" in name:
            return "sprint"
        return "race"
    # Adjust based on actual naming conventions if needed
    elif "sprint shootout" in name: # Example alternative name
        return "sprint_qualifying"
    return "unknown"

def add_race_weekend_info(df):
    """Adds Year and RaceWeekend columns based on date and country/session name."""
    # Determine which date column to use for year extraction
    date_col_to_use = None
    if "session_date" in df.columns:
        date_col_to_use = "session_date"
    elif "date" in df.columns: # Fallback if session_date was missing earlier
        date_col_to_use = "date"
        logger.warning("Column 'session_date' not found, falling back to 'date' for year extraction.")
    
    if date_col_to_use:
        # Ensure the date column exists before trying to use it
        if date_col_to_use in df.columns:
            df[f"{date_col_to_use}_dt"] = pd.to_datetime(df[date_col_to_use], errors="coerce")
            df["year"] = df[f"{date_col_to_use}_dt"].dt.year
            df.drop(columns=[f"{date_col_to_use}_dt"], inplace=True)
        else:
             logger.error(f"Date column '{date_col_to_use}' not found, cannot extract year.")
             df["year"] = None # Set year to None if date column is missing
    else:
        logger.error("Columns 'session_date' or 'date' not found, cannot extract year.")
        df["year"] = None

    # Prioritize country_name if available and not all null
    if "country_name" in df.columns and df["country_name"].notna().any():
        df["race_weekend"] = df["year"].fillna(0).astype(int).astype(str) + "_" + df["country_name"].fillna("Unknown").astype(str)
    elif "session_name" in df.columns:
        logger.warning("Attempting to infer race weekend from session_name as country_name is missing/unreliable.")
        # Regex to extract potential country name from session name (needs refinement)
        df["race_weekend"] = df["year"].fillna(0).astype(int).astype(str) + "_" + df["session_name"].str.extract(r"(\d{4})?\s?([A-Za-z\s]+?)\s(Grand Prix|Practice|Qualifying|Race|Day|Sprint)", expand=False)[1].str.strip().fillna("Unknown")
    else:
        logger.error("Cannot determine Race Weekend identifier (missing country_name and session_name).")
        df["race_weekend"] = None

    # Clean up the race_weekend identifier
    if "race_weekend" in df.columns and df["race_weekend"] is not None:
        df["race_weekend"] = df["race_weekend"].str.replace(" ", "_", regex=False).str.replace("[^A-Za-z0-9_]+", "", regex=True)

    logger.info("Added 'year' and 'race_weekend' columns.")
    return df

def add_rolling_lap_features(df, window=ROLLING_LAP_WINDOW):
    """Adds rolling average lap time features."""
    logger.info(f"Adding rolling lap time features with window size {window}...")
    required_cols = ["session_key", "driver_number", "lap_number", "lap_duration"]
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Required columns for rolling features missing ({required_cols}). Skipping.")
        return df

    # Ensure correct sorting for rolling calculation
    df_sorted = df.sort_values(by=["session_key", "driver_number", "lap_number"])

    # Calculate rolling average of previous N laps for each driver within each session
    group_cols = ["session_key", "driver_number"]
    rolling_col_name = f"rolling_avg_lap_time_{window}"
    df_sorted[rolling_col_name] = df_sorted.groupby(group_cols)["lap_duration"].shift(1).rolling(window=window, min_periods=1).mean()

    overall_median_lap_time = df_sorted["lap_duration"].median()
    if pd.isna(overall_median_lap_time):
        overall_median_lap_time = df_sorted["lap_duration"].mean() # Fallback to mean
        logger.warning(f"Overall median lap duration is NaN, using mean ({overall_median_lap_time:.3f}) for filling rolling avg NaNs.")
    if pd.isna(overall_median_lap_time):
         overall_median_lap_time = 90 # Absolute fallback
         logger.error(f"Overall mean and median lap duration are NaN, using fallback value ({overall_median_lap_time}) for rolling avg NaNs.")

    # FIX: Avoid chained assignment warning
    df_sorted[rolling_col_name] = df_sorted[rolling_col_name].fillna(overall_median_lap_time)
    logger.info(f"Calculated {rolling_col_name}. Filled NaNs with {overall_median_lap_time:.3f}.")

    return df_sorted

def clean_data(df):
    """Cleans the merged dataframe."""
    logger.info(f"Starting cleaning. Initial rows: {len(df)}")

    # FIX: Handle missing date_start/session_date (drop rows)
    # Identify the date column used (it should be 'session_date' after merge_data)
    date_col_final = "session_date"
    if date_col_final in df.columns:
        initial_rows = len(df)
        df.dropna(subset=[date_col_final], inplace=True)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows due to missing values in '{date_col_final}' column.")
    else:
        logger.error(f"Critical date column '{date_col_final}' not found before cleaning. Cannot proceed with date-dependent cleaning.")
        # Depending on requirements, you might return None or raise an error here
        # return None 

    # Infer session_type if missing or needs update
    if "session_name" in df.columns:
        logger.info("Inferring/updating 'session_type' from 'session_name' during cleaning.")
        df["session_type"] = df["session_name"].apply(infer_session_type)
    elif "session_type" not in df.columns or df["session_type"].isnull().all():
        logger.warning("Cannot infer 'session_type' as 'session_name' is missing.")
        df["session_type"] = "unknown"

    # Filter for relevant session types (including sprint types)
    valid_sessions = ["practice", "qualifying", "race", "sprint", "sprint_qualifying"]
    df = df[df["session_type"].isin(valid_sessions)]
    logger.info(f"Rows after filtering session types ({valid_sessions}): {len(df)}")

    # Filter invalid lap durations
    if "lap_duration" in df.columns:
        original_rows = len(df)
        df = df[df["lap_duration"] > 0]
        filtered_rows = original_rows - len(df)
        if filtered_rows > 0:
            logger.info(f"Filtered out {filtered_rows} rows with invalid lap_duration (<= 0). Rows remaining: {len(df)}")
    else:
        logger.warning("Column 'lap_duration' not found. Skipping duration filtering.")

    # Define columns to fill and their strategies (median for most)
    numeric_cols_median = [
        "air_temperature", "humidity", "pressure", "rainfall", "track_temperature",
        "wind_direction", "wind_speed", "duration_sector_1", "duration_sector_2",
        "duration_sector_3", "i1_speed", "i2_speed", "st_speed",
        f"rolling_avg_lap_time_{ROLLING_LAP_WINDOW}"
    ]
    fill_values = {}
    for col in numeric_cols_median:
        if col in df.columns:
            # Check if column is numeric before calculating median/mean
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                if pd.notna(median_val):
                    fill_values[col] = median_val
                else:
                    mean_val = df[col].mean()
                    if pd.notna(mean_val):
                        fill_values[col] = mean_val
                        logger.warning(f"Median for {col} is NaN, filling with mean ({mean_val:.3f}).")
                    else:
                        fill_values[col] = 0
                        logger.warning(f"Median and Mean for {col} are NaN, filling with 0.")
            else:
                logger.warning(f"Column {col} is not numeric, cannot calculate median/mean. Skipping fill for this column.")
        else:
             fill_values[col] = 0
             logger.warning(f"Column {col} not found, adding and filling with 0.")
             if col not in df.columns: df[col] = 0 # Add the column first

    # Fill other specific columns
    if "session_type" in df.columns:
        fill_values["session_type"] = "unknown"
    else:
        df["session_type"] = "unknown"

    if "is_pit_out_lap" in df.columns:
        fill_values["is_pit_out_lap"] = False
    else:
        df["is_pit_out_lap"] = False
        logger.warning("Column is_pit_out_lap not found, adding and filling with False.")

    # Handle race_position NaNs (fill with a high value like 99 to indicate not finished or not applicable)
    if "race_position" in df.columns:
        fill_values["race_position"] = 99
    else:
        # If race_position wasn't added (e.g., Ergast merge failed), add it and fill
        df["race_position"] = 99
        logger.warning("Column race_position not found, adding and filling with 99.")

    df.fillna(value=fill_values, inplace=True)
    logger.info("Filled NaN values using median/mean for numeric or default for others.")

    # Convert race_position to integer after filling NaNs
    if "race_position" in df.columns:
        # Ensure it's numeric first before converting to int
        df["race_position"] = pd.to_numeric(df["race_position"], errors="coerce").fillna(99).astype(int)

    # Final check for NaNs
    missing_after = df.isnull().sum()
    missing_after = missing_after[missing_after > 0]
    if not missing_after.empty:
        logger.warning(f"Still missing values after cleaning: {missing_after.sum()}")
        logger.warning(f"Columns with remaining NaNs:\n{missing_after}")
    else:
        logger.info("All missing values handled.")

    logger.info(f"Cleaning complete. Final rows: {len(df)}")
    return df

def merge_data(lap_data, weather_data, session_data, ergast_results):
    """Merges lap, session, weather, and Ergast race results data."""
    logger.info("Starting data merging...")

    # --- Pre-checks --- 
    if lap_data is None or lap_data.empty:
        logger.error("Lap data is empty/None. Cannot proceed.")
        return None
    if session_data is None or session_data.empty:
        logger.error("Session data is empty/None. Cannot proceed.")
        return None
    if "session_key" not in lap_data.columns:
        logger.error("Missing 'session_key' in lap_data. Cannot merge.")
        return None

    # --- Prepare Session Data --- 
    # Determine which date column exists in session_data
    session_date_col = None
    if "date_start" in session_data.columns:
        session_date_col = "date_start"
    elif "date" in session_data.columns:
        session_date_col = "date"
        logger.warning("Using 'date' column from session data as 'date_start' is missing.")
    else:
        logger.error("Neither 'date_start' nor 'date' column found in session_data. Cannot determine session date.")
        return None # Critical for year/race_weekend

    # Ensure year column exists (needed for meeting_key placeholder and Ergast merge)
    if "year" not in session_data.columns:
        logger.info(f"Extracting 'year' from '{session_date_col}' in session_data.")
        session_data["year"] = pd.to_datetime(session_data[session_date_col], errors="coerce").dt.year
        # Handle potential NaNs in year if date conversion failed
        if session_data["year"].isnull().any():
            logger.warning("Some years could not be extracted from session dates.")

    # Handle missing circuit_short_name
    if "circuit_short_name" not in session_data.columns:
        logger.warning("Column 'circuit_short_name' missing in session_data. Adding placeholder based on country.")
        if "country" in session_data.columns:
            session_data["circuit_short_name"] = session_data["country"].str.lower().str.replace(" ", "_", regex=False) + "_circuit"
        else:
            logger.error("Cannot create placeholder for 'circuit_short_name' as 'country' column is also missing.")
            session_data["circuit_short_name"] = "unknown_circuit"
            
    # FIX: Handle missing meeting_key - Ensure it exists before selecting columns
    if "meeting_key" not in session_data.columns:
        logger.warning("Column 'meeting_key' missing in session_data. Adding placeholder.")
        if "year" in session_data.columns and "country" in session_data.columns:
             # Ensure year is not NaN before creating key
             session_data["meeting_key"] = session_data["year"].fillna(0).astype(int).astype(str) + "_" + session_data["country"].str.lower().str.replace(" ", "_", regex=False)
        else:
             logger.error("Cannot create reliable placeholder for 'meeting_key'. Using -1.")
             session_data["meeting_key"] = -1 # Use a distinct placeholder

    # Define columns to keep from session_data - ENSURE meeting_key is included
    session_cols_to_keep = ["session_key", "meeting_key", "country", session_date_col, "session_name", "circuit_short_name", "year"]
    
    # Check if all required columns exist in session_data NOW
    missing_session_cols = [col for col in session_cols_to_keep if col not in session_data.columns]
    if missing_session_cols:
         logger.error(f"Required columns missing in session_data before merge: {missing_session_cols}. Cannot proceed.")
         return None
             
    # --- 1. Merge Lap and Session Data --- 
    logger.info("Merging lap_data with session_data on 'session_key'...")
    try:
        # Select subset and rename columns for clarity after merge
        session_data_subset = session_data[session_cols_to_keep].rename(columns={
            "country": "country_name", 
            session_date_col: "session_date" # Standardize date column name
        })
        merged_data = pd.merge(lap_data, session_data_subset, on="session_key", how="left")
        logger.info(f"Rows after merging laps and sessions: {len(merged_data)}")
        # Verify meeting_key is present after merge
        if "meeting_key" not in merged_data.columns:
             logger.error("CRITICAL: 'meeting_key' was lost during lap/session merge!")
             # Attempt to re-merge just the key if possible, or fail
             if "session_key" in merged_data.columns and "meeting_key" in session_data.columns:
                 logger.info("Attempting to re-merge meeting_key...")
                 merged_data = pd.merge(merged_data, session_data[["session_key", "meeting_key"]].drop_duplicates(), on="session_key", how="left")
                 if "meeting_key" not in merged_data.columns:
                     logger.error("Re-merge failed. Cannot proceed without meeting_key.")
                     return None
             else:
                 return None
        elif merged_data["meeting_key"].isnull().any():
             logger.warning("Some rows have null 'meeting_key' after lap/session merge.")

    except Exception as e:
        logger.exception(f"Error merging lap_data and session_data: {e}")
        return None

    # --- 2. Merge with Weather Data --- 
    weather_cols_expected = ["air_temperature", "humidity", "pressure", "rainfall", "track_temperature", "wind_direction", "wind_speed"]
    
    # Add expected weather columns with NaNs if they don't exist yet
    for col in weather_cols_expected:
        if col not in merged_data.columns:
            merged_data[col] = np.nan
            
    if weather_data is None or weather_data.empty:
        logger.warning("Weather data is empty or None. Skipping weather merge. Weather columns will be NaN.")
    elif "meeting_key" not in merged_data.columns or merged_data["meeting_key"].isnull().all():
        logger.error("Column 'meeting_key' missing or all null in merged data. Cannot merge weather.")
    elif "meeting_key" not in weather_data.columns:
         logger.error("Column 'meeting_key' missing in weather_data. Cannot merge weather.")
    else:
        logger.info("Merging with weather_data on 'meeting_key'...")
        try:
            # Prepare weather data: keep relevant columns, drop duplicates
            weather_cols_to_keep = ["meeting_key"] + [col for col in weather_cols_expected if col in weather_data.columns]
            weather_data_unique = weather_data[weather_cols_to_keep].drop_duplicates(subset=["meeting_key"], keep="last")
            
            # Perform the merge
            merged_data = pd.merge(merged_data, weather_data_unique, on="meeting_key", how="left", suffixes=("", "_weather"))
            
            # Handle potential duplicate columns if suffixes were added (e.g., if a weather col already existed)
            for col in weather_cols_expected:
                if f"{col}_weather" in merged_data.columns:
                    # Prioritize the original column, fill NaNs with the _weather version
                    merged_data[col] = merged_data[col].fillna(merged_data[f"{col}_weather"])
                    merged_data.drop(columns=[f"{col}_weather"], inplace=True)
            logger.info(f"Rows after merging with weather: {len(merged_data)}")
        except Exception as e:
            logger.exception(f"Error merging with weather_data: {e}")
            # Weather columns should already exist with NaNs from the pre-addition step

    # --- 3. Merge with Ergast Race Results --- 
    # Ensure race_position column exists, even if merge fails
    if "race_position" not in merged_data.columns:
        merged_data["race_position"] = np.nan
        
    if ergast_results is None or ergast_results.empty:
        logger.warning("Ergast results data is empty or None. Skipping race results merge. Race positions will be NaN.")
    elif not all(k in merged_data.columns for k in ["year", "driver_number"]):
        logger.error("Missing 'year' or 'driver_number' in main data. Cannot merge Ergast results.")
    else:
        logger.info("Preparing to merge with Ergast race results...")
        try:
            # Ensure Ergast columns have correct types
            ergast_results["year"] = pd.to_numeric(ergast_results["year"], errors="coerce")
            ergast_results["round"] = pd.to_numeric(ergast_results["round"], errors="coerce")
            ergast_results["driver_number"] = pd.to_numeric(ergast_results["driver_number"], errors="coerce")
            ergast_results["race_position_ergast"] = pd.to_numeric(ergast_results["position"], errors="coerce") # Use new name
            ergast_results.dropna(subset=["year", "round", "driver_number"], inplace=True)
            ergast_results["year"] = ergast_results["year"].astype(int)
            ergast_results["round"] = ergast_results["round"].astype(int)
            ergast_results["driver_number"] = ergast_results["driver_number"].astype(int)

            ergast_to_merge = ergast_results[["year", "round", "driver_number", "race_position_ergast"]].copy()

            # --- Fetch Ergast Schedules to get 'round' number for main data ---
            unique_years = merged_data["year"].dropna().unique().astype(int)
            all_schedules = pd.DataFrame()
            logger.info(f"Fetching Ergast schedules for years: {unique_years}")
            for year in unique_years:
                schedule_df = fetch_ergast_schedule(year)
                if not schedule_df.empty:
                    all_schedules = pd.concat([all_schedules, schedule_df], ignore_index=True)
                time.sleep(ERGAST_REQUEST_DELAY)

            if all_schedules.empty:
                logger.error("Failed to fetch any Ergast schedules. Cannot map 'round' number for Ergast merge.")
            elif "circuit_short_name" not in merged_data.columns:
                 logger.error("Missing 'circuit_short_name' in main data. Cannot map 'round' number for Ergast merge.")
            else:
                # --- Map circuit_short_name to circuit_id ---
                logger.warning("Attempting to merge schedule using 'circuit_short_name' as 'circuit_id'. This might be inaccurate.")
                # Ensure year types match before merge
                all_schedules["year"] = all_schedules["year"].astype(int)
                merged_data["year"] = merged_data["year"].astype(int)
                
                merged_data_with_schedule = pd.merge(
                    merged_data,
                    all_schedules[["year", "round", "circuit_id"]],
                    left_on=["year", "circuit_short_name"],
                    right_on=["year", "circuit_id"],
                    how="left"
                )
                round_missing_count = merged_data_with_schedule["round"].isnull().sum()
                if round_missing_count > 0:
                    logger.warning(f"{round_missing_count} rows could not be mapped to an Ergast round based on year/circuit_short_name.")

                # --- Perform the final merge with Ergast results ---
                logger.info("Merging main data with Ergast results on year, round, driver_number...")
                # Ensure key types match
                merged_data_with_schedule["round"] = pd.to_numeric(merged_data_with_schedule["round"], errors="coerce").astype("Int64") # Use nullable Int
                merged_data_with_schedule["driver_number"] = pd.to_numeric(merged_data_with_schedule["driver_number"], errors="coerce").astype("Int64")
                ergast_to_merge["driver_number"] = ergast_to_merge["driver_number"].astype("Int64")
                ergast_to_merge["round"] = ergast_to_merge["round"].astype("Int64")
                
                final_merged_data = pd.merge(
                    merged_data_with_schedule,
                    ergast_to_merge,
                    on=["year", "round", "driver_number"],
                    how="left"
                    # No suffixes needed as we renamed ergast column
                )

                # Prioritize the race_position from Ergast if merge was successful
                if "race_position_ergast" in final_merged_data.columns:
                    # Update existing race_position only where ergast data is available
                    final_merged_data["race_position"] = final_merged_data["race_position_ergast"].fillna(final_merged_data["race_position"])
                    final_merged_data.drop(columns=["race_position_ergast", "circuit_id", "round"], inplace=True, errors="ignore")
                else:
                    logger.warning("'race_position_ergast' column not found after merge.")
                    # race_position column should already exist from pre-addition

                logger.info(f"Rows after merging with Ergast results: {len(final_merged_data)}")
                merged_data = final_merged_data # Update merged_data for subsequent steps

        except Exception as e:
            logger.exception(f"Error merging with Ergast results data: {e}")
            # race_position column should already exist

    # --- 4. Infer session_type (after all merges) ---
    if "session_name" in merged_data.columns:
        merged_data["session_type"] = merged_data["session_name"].apply(infer_session_type)
        logger.info("Inferred/updated 'session_type' from 'session_name'.")
    else:
        logger.warning("Column 'session_name' not found after merges. Cannot infer 'session_type'.")
        if "session_type" not in merged_data.columns:
             merged_data["session_type"] = "unknown"

    logger.info("Data merging complete.")
    return merged_data

def generate_summary(df, output_path=SUMMARY_FILE):
    """Generates a summary of the cleaned dataframe."""
    if df is None or df.empty:
        logger.error("Cannot generate summary for empty or None dataframe.")
        return
        
    logger.info(f"Generating summary report to {output_path}...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"--- Data Summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n\n")
            f.write(f"Total rows: {len(df)}\n")
            f.write(f"Total columns: {len(df.columns)}\n\n")

            f.write("Columns and Data Types:\n")
            f.write(df.dtypes.to_string() + "\n\n")

            f.write("Missing Values per Column:\n")
            missing_values = df.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            if not missing_values.empty:
                f.write(missing_values.to_string() + "\n\n")
            else:
                f.write("No missing values found.\n\n")

            f.write("Basic Statistics for Numeric Columns:\n")
            # Select only numeric columns for describe
            numeric_df = df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                f.write(numeric_df.describe().to_string() + "\n\n")
            else:
                f.write("No numeric columns found.\n\n")

            f.write("Value Counts for Key Categorical Columns:\n")
            categorical_cols = ["session_type", "driver_number", "team_name", "country_name", "year", "race_weekend", "race_position"]
            for col in categorical_cols:
                if col in df.columns:
                    f.write(f"\n--- {col} ---\n")
                    # Show top 20 most frequent values
                    # Check if column is categorical or object type before value_counts
                    if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                        f.write(df[col].value_counts().head(20).to_string() + "\n")
                    else:
                        # For numeric identifiers like driver_number, show counts differently if needed
                        f.write(df[col].value_counts().head(20).to_string() + "\n") # Keep as is for now

        logger.info("Summary report generated successfully.")
    except Exception as e:
        logger.exception(f"Error generating summary report: {e}")

def main():
    """Main function to load, merge, clean, and save data."""
    logger.info("Starting dataset builder pipeline...")

    # Load data
    try:
        logger.info(f"Loading lap data from {INPUT_LAPS}")
        lap_data = pd.read_csv(INPUT_LAPS, low_memory=False)
        logger.info(f"Loaded {len(lap_data)} rows from lap data.")
    except FileNotFoundError:
        logger.error(f"Lap data file not found: {INPUT_LAPS}")
        return
    except Exception as e:
        logger.exception(f"Error loading lap data: {e}")
        return

    try:
        logger.info(f"Loading session data from {INPUT_SESSIONS}")
        session_data = pd.read_csv(INPUT_SESSIONS, low_memory=False)
        logger.info(f"Loaded {len(session_data)} rows from session data.")
    except FileNotFoundError:
        logger.error(f"Session data file not found: {INPUT_SESSIONS}")
        return
    except Exception as e:
        logger.exception(f"Error loading session data: {e}")
        return

    try:
        logger.info(f"Loading weather data from {INPUT_WEATHER}")
        weather_data = pd.read_csv(INPUT_WEATHER, low_memory=False)
        logger.info(f"Loaded {len(weather_data)} rows from weather data.")
    except FileNotFoundError:
        logger.warning(f"Weather data file not found: {INPUT_WEATHER}. Proceeding without weather data.")
        weather_data = None
    except Exception as e:
        logger.exception(f"Error loading weather data: {e}")
        weather_data = None # Treat as missing if error occurs

    try:
        logger.info(f"Loading Ergast results data from {INPUT_ERGAST_RESULTS}")
        ergast_results = pd.read_csv(INPUT_ERGAST_RESULTS, low_memory=False)
        logger.info(f"Loaded {len(ergast_results)} rows from Ergast results data.")
    except FileNotFoundError:
        logger.warning(f"Ergast results file not found: {INPUT_ERGAST_RESULTS}. Race positions will be missing.")
        ergast_results = None # Set to None if not found
    except Exception as e:
        logger.exception(f"Error loading Ergast results data: {e}")
        ergast_results = None

    # Merge data
    merged_df = merge_data(lap_data, weather_data, session_data, ergast_results)

    if merged_df is None or merged_df.empty:
        logger.error("Merging failed or resulted in empty dataframe. Exiting.")
        return

    # Add race weekend info (depends on columns from session merge)
    merged_df = add_race_weekend_info(merged_df)

    # Feature Engineering
    merged_df = add_rolling_lap_features(merged_df)

    # Clean data
    cleaned_df = clean_data(merged_df)

    if cleaned_df is None or cleaned_df.empty:
        logger.error("Cleaning failed or resulted in empty dataframe. Exiting.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CLEAN_DATA), exist_ok=True)

    # Save cleaned data
    try:
        logger.info(f"Saving cleaned data to {OUTPUT_CLEAN_DATA}...")
        cleaned_df.to_csv(OUTPUT_CLEAN_DATA, index=False, encoding='utf-8') # FIX: Corrected encoding string
        logger.info(f"Successfully saved {len(cleaned_df)} rows to {OUTPUT_CLEAN_DATA}.")
    except Exception as e:
        logger.exception(f"Error saving cleaned data: {e}")

    # Generate summary
    generate_summary(cleaned_df, SUMMARY_FILE)

    logger.info("Dataset builder pipeline finished.")

if __name__ == "__main__":
    main()


