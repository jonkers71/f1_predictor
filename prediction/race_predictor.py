# File: prediction/race_predictor.py

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging

# Add project root to the sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.logger import setup_logger

logger = setup_logger("RacePredictor")

# --- Configuration ---
MODEL_DIR = os.path.join(PROJECT_ROOT, "models/race_predictor")
MODEL_FILE = os.path.join(MODEL_DIR, "race_predictor_model.joblib")
FEATURES_FILE = os.path.join(MODEL_DIR, "race_predictor_features.txt")
SCALER_FILE = os.path.join(MODEL_DIR, "race_predictor_scaler.joblib")
ENCODERS_FILE = os.path.join(MODEL_DIR, "race_predictor_encoders.joblib")
DRIVER_MAP_FILE = os.path.join(PROJECT_ROOT, "data/processed/driver_map.csv") # To get driver list

# --- End Configuration ---

def load_prediction_assets():
    """Loads the trained model, scaler, encoders, and feature list."""
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        encoders = joblib.load(ENCODERS_FILE)
        with open(FEATURES_FILE, "r") as f:
            features = [line.strip() for line in f if line.strip()]
        logger.info("Successfully loaded prediction assets.")
        return model, scaler, encoders, features
    except FileNotFoundError as e:
        logger.error(f"Error loading prediction asset: {e}. Model or supporting files not found. Please train the race predictor model first.")
        return None, None, None, None
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading prediction assets: {e}")
        return None, None, None, None

def get_driver_list():
    """Loads the list of drivers from the driver map."""
    try:
        driver_map_df = pd.read_csv(DRIVER_MAP_FILE)
        # Assuming columns are 'driver_number', 'full_name', 'team_name' etc.
        # Return relevant info, maybe a list of dicts or a simplified DataFrame
        drivers = driver_map_df[['driver_number', 'full_name', 'team_name']].to_dict('records')
        logger.info(f"Loaded {len(drivers)} drivers from driver map.")
        return drivers
    except FileNotFoundError:
        logger.error(f"Driver map file not found: {DRIVER_MAP_FILE}")
        return []
    except Exception as e:
        logger.exception(f"Error loading driver map: {e}")
        return []

def gather_prediction_input(features, drivers):
    """Gathers the necessary input data for prediction (simplified: user prompts)."""
    logger.info("Gathering prediction input data...")
    input_data = []
    
    # --- Get General Race Info (Simplified) ---
    try:
        year = int(input("Enter the race year (e.g., 2024): "))
        circuit_short_name = input("Enter the circuit short name (e.g., bahrain): ").lower()
        # Simplified weather - potentially fetch real forecast later
        air_temp = float(input("Enter forecast Air Temperature (°C): "))
        humidity = float(input("Enter forecast Humidity (%): "))
        pressure = float(input("Enter forecast Pressure (hPa): "))
        rainfall = int(input("Enter forecast Rainfall (0=No, 1=Yes): ")) # Simplified
        track_temp = float(input("Enter forecast Track Temperature (°C): "))
        wind_dir = int(input("Enter forecast Wind Direction (degrees): "))
        wind_speed = float(input("Enter forecast Wind Speed (m/s): "))
    except ValueError:
        logger.error("Invalid numeric input for race/weather info.")
        return None

    # --- Get Per-Driver Info ---
    print("\nEnter data for each driver:")
    if not drivers:
        print("No drivers found. Cannot proceed.")
        return None
        
    for driver in drivers:
        driver_num = driver['driver_number']
        driver_name = driver['full_name']
        team_name = driver['team_name']
        print(f"\nDriver: {driver_name} ({driver_num}) - Team: {team_name}")
        try:
            # These would ideally come from fetched quali/practice/sprint data
            quali_pos = int(input(f"  Enter Qualifying Position: "))
            avg_practice_lap_time = float(input(f"  Enter Average Practice Lap Time (seconds): "))
            sprint_pos_input = input(f"  Enter Sprint Race Position (or leave blank if none/NA): ")
            sprint_pos = int(sprint_pos_input) if sprint_pos_input else 99 # Default to 99 if no sprint

            driver_data = {
                "year": year,
                "circuit_short_name": circuit_short_name,
                "team_name": team_name,
                "driver_number": driver_num,
                "air_temperature": air_temp,
                "humidity": humidity,
                "pressure": pressure,
                "rainfall": rainfall,
                "track_temperature": track_temp,
                "wind_direction": wind_dir,
                "wind_speed": wind_speed,
                "quali_pos": quali_pos,
                "avg_practice_lap_time": avg_practice_lap_time,
                "sprint_pos": sprint_pos
            }
            # Ensure only features expected by the model are included
            filtered_driver_data = {f: driver_data.get(f) for f in features}
            input_data.append(filtered_driver_data)
            
        except ValueError:
            logger.error(f"Invalid numeric input for driver {driver_name}. Skipping driver.")
            continue # Skip this driver if input is bad
            
    if not input_data:
        logger.error("No valid driver data entered.")
        return None
        
    return pd.DataFrame(input_data)

def preprocess_input_data(df, scaler, encoders, features):
    """Preprocesses the input DataFrame using loaded assets."""
    logger.info("Preprocessing input data...")
    if df is None:
        return None
        
    # Ensure DataFrame has all required feature columns, even if empty initially
    for feature in features:
        if feature not in df.columns:
            df[feature] = np.nan # Add missing feature columns
            
    # Reorder columns to match the training order
    df = df[features]

    # Apply Label Encoding
    categorical_features = [f for f in encoders.keys() if f in df.columns]
    for col in categorical_features:
        le = encoders[col]
        # Handle unseen labels: Assign a default value (e.g., -1 or len(classes))
        # Or map them to an 'unknown' category if trained that way.
        # Simple approach: convert to str, use transform, ignore errors for unseen.
        df[col] = df[col].astype(str)
        try:
            # Get known classes from the encoder
            known_classes = list(le.classes_)
            # Apply transform, mapping unknown values to a specific category (e.g., the first class or a dedicated 'unknown' class if available)
            # A simple approach is to replace unknowns with a common value like the most frequent class or handle them post-prediction.
            # Here, we'll try transforming and let errors occur for unknowns, then potentially handle NaNs.
            # A safer approach: map known values, mark others as NaN or a specific code.
            df[col] = df[col].apply(lambda x: x if x in known_classes else known_classes[0]) # Map unknown to first known class
            df[col] = le.transform(df[col])
        except Exception as e:
            logger.warning(f"Could not apply label encoding for column '{col}': {e}. Filling with 0.")
            df[col] = 0 # Fill with a default encoded value if transform fails

    # Apply Scaling
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_features:
        logger.error("No numeric features found in input data for scaling.")
        return None
        
    try:
        df[numeric_features] = scaler.transform(df[numeric_features])
        logger.info("Applied scaling to numeric features.")
    except Exception as e:
        logger.exception(f"Error applying scaling: {e}")
        return None
        
    # Handle any remaining NaNs (e.g., from missing inputs or failed encoding)
    # Simple strategy: fill with 0 after scaling (mean of scaled data)
    df.fillna(0, inplace=True)
    logger.info("Filled remaining NaNs with 0.")

    return df

def predict_upcoming_race():
    """Loads model, gets input, preprocesses, predicts, and displays results."""
    logger.info("--- Starting Upcoming Race Prediction ---")
    
    model, scaler, encoders, features = load_prediction_assets()
    if not all([model, scaler, encoders, features]):
        print("\n❌ Prediction assets could not be loaded. Please ensure the race predictor model is trained.")
        return

    drivers = get_driver_list()
    if not drivers:
        print("\n❌ Could not load driver list. Cannot proceed with prediction.")
        return
        
    # Gather input data (simplified via prompts)
    input_df_raw = gather_prediction_input(features, drivers)
    if input_df_raw is None:
        print("\n❌ Failed to gather valid input data for prediction.")
        return

    # Preprocess the data
    input_df_processed = preprocess_input_data(input_df_raw.copy(), scaler, encoders, features)
    if input_df_processed is None:
        print("\n❌ Failed to preprocess input data.")
        return

    # Make Predictions
    try:
        logger.info("Making predictions...")
        predictions = model.predict(input_df_processed)
        logger.info("Predictions generated.")
    except Exception as e:
        logger.exception(f"Error during model prediction: {e}")
        print("\n❌ An error occurred while making predictions.")
        return

    # Format and Display Results
    results_df = input_df_raw[['driver_number']].copy() # Start with driver numbers from raw input
    # Merge driver names back for display
    driver_map_df = pd.DataFrame(drivers)
    results_df = pd.merge(results_df, driver_map_df[['driver_number', 'full_name', 'team_name']], on='driver_number', how='left')
    
    results_df['predicted_position'] = predictions
    # Round predictions and convert to integer
    results_df['predicted_position'] = results_df['predicted_position'].round().astype(int)
    # Sort by predicted position
    results_df = results_df.sort_values(by='predicted_position')
    # Assign rank based on sorted order
    results_df['predicted_rank'] = range(1, len(results_df) + 1)

    print("\n--- Predicted Race Results ---")
    print(results_df[['predicted_rank', 'full_name', 'team_name', 'predicted_position']].to_string(index=False))
    logger.info("--- Upcoming Race Prediction Finished ---")

if __name__ == '__main__':
    # Example of how to run it directly (for testing)
    predict_upcoming_race()

