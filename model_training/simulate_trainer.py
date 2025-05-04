# File: model_training/simulate_trainer.py

import pandas as pd
import logging
import xgboost as xgb
import os

# Logger setup
logger = logging.getLogger("SimulateTrainer")

# Define desired features (should match trainer.py)
# Ideally, this list should be loaded from a shared config or saved during training
# Added rolling_avg_lap_time_3 to match trainer.py
desired_features = [
    "duration_sector_1", "duration_sector_2", "duration_sector_3",
    "i1_speed", "i2_speed", "st_speed",
    # "segments_sector_1", "segments_sector_2", "segments_sector_3", # Complex/list features excluded
    "air_temperature", "humidity", "pressure", "rainfall",
    "track_temperature", "wind_direction", "wind_speed",
    "lap_number", "is_pit_out_lap",
    "rolling_avg_lap_time_3", # <-- Added new feature
    # Categorical features to be one-hot encoded
    "driver_number", # Keep driver ID
    # Features not found in current data based on trainer logs:
    # "team_name", 
    # "circuit_key", 
    # "compound", 
    "session_type" # Categorical feature
]

# --- Placeholder for loading training columns --- 
training_columns = None # Placeholder

def load_training_columns(path="models/training_columns.txt"):
    """Loads the list of training columns from a file."""
    global training_columns
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                training_columns = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Loaded {len(training_columns)} training columns from {path}")
        except Exception as e:
            logger.exception(f"Error reading training columns file {path}: {e}")
            training_columns = []
    else:
        logger.error(f"Training columns file not found at {path}. Preprocessing might fail.")
        training_columns = [] # Ensure it's an empty list if file not found

def load_trained_model():
    """
    Load the trained XGBoost model.
    """
    logger.info("Loading trained model...")
    model_path = "models/xgb_best_model.json"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None
    model = xgb.XGBRegressor()
    try:
        model.load_model(model_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.exception(f"Failed to load model from {model_path}: {e}")
        return None

def preprocess_data_for_simulation(session_data):
    """
    Preprocess session data for simulation, ensuring consistency with training.
    """
    logger.info("Preprocessing session data for simulation...")
    
    if training_columns is None:
        load_training_columns() # Attempt to load columns if not already loaded
        if not training_columns: # Check if loading failed or was empty
             logger.error("Training columns are not available. Cannot preprocess data.")
             return None

    # 1. Filter to desired features available in the input data
    features_present = [f for f in desired_features if f in session_data.columns]
    missing_in_input = set(desired_features) - set(features_present)
    if missing_in_input:
        # Log only if the missing feature was actually expected based on training columns
        expected_missing = [f for f in missing_in_input if f in training_columns]
        if expected_missing:
            logger.warning(f"Expected features missing in simulation input data: {expected_missing}")
        else:
            logger.info(f"Features missing in input but not in training columns: {missing_in_input}")
    
    X_sim = session_data[features_present].copy()

    # 2. Impute NaNs before encoding (using same strategy as trainer - median for numeric, mode for categorical)
    for col in features_present:
        if pd.api.types.is_numeric_dtype(X_sim[col]):
            if X_sim[col].isnull().any():
                # Use median from the input simulation data itself - might differ slightly from training median
                # A more robust approach would save training medians/modes
                median_val = X_sim[col].median()
                if pd.isna(median_val):
                    median_val = 0 # Fallback
                    logger.warning(f"Median for '{col}' in simulation data is NaN, filling with 0.") # Fixed f-string
                X_sim[col].fillna(median_val, inplace=True)
                logger.info(f"Filled NaNs in numeric feature '{col}' with median ({median_val})") # Fixed f-string
        elif pd.api.types.is_object_dtype(X_sim[col]) or pd.api.types.is_categorical_dtype(X_sim[col]):
             if X_sim[col].isnull().any():
                mode_val = X_sim[col].mode()[0] if not X_sim[col].mode().empty else "Unknown"
                X_sim[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled NaNs in categorical feature '{col}' with mode ('{mode_val}')") # Fixed f-string

    # 3. Apply get_dummies (consistent with training: drop_first=True)
    X_sim_encoded = pd.get_dummies(X_sim, drop_first=True)

    # 4. Sanitize feature names (consistent with training)
    X_sim_encoded.columns = X_sim_encoded.columns.str.replace("[", "_", regex=False).str.replace("]", "_", regex=False).str.replace("<", "_", regex=False)

    # 5. Reindex to match training columns
    # This adds missing columns (with value 0) and removes extra columns
    X_sim_reindexed = X_sim_encoded.reindex(columns=training_columns, fill_value=0)
    
    logger.info(f"Preprocessing complete. Data shape: {X_sim_reindexed.shape}")
    
    # Final check for NaNs after reindexing (should ideally be 0)
    if X_sim_reindexed.isnull().sum().sum() > 0:
        logger.warning("NaN values found after reindexing. Filling with 0.")
        X_sim_reindexed = X_sim_reindexed.fillna(0)
        
    return X_sim_reindexed

def simulate_past_quali(session_data):
    """
    Simulate past qualifying results based on available session data.
    Requires session_data to contain necessary features.
    """
    logger.info("Simulating past qualifying results...")
    
    # Ensure driver_number exists for grouping results
    if "driver_number" not in session_data.columns:
         logger.error("Input data for simulation must contain \"driver_number\".")
         return None
    driver_id_col = "driver_number"

    # Preprocess data into features
    features = preprocess_data_for_simulation(session_data)
    if features is None:
        logger.error("Preprocessing failed. Cannot simulate.")
        return None
        
    # Load trained model
    model = load_trained_model()
    if model is None:
        logger.error("Failed to load model. Cannot simulate.")
        return None
    
    # Predict lap times for qualifying
    try:
        predictions = model.predict(features)
        logger.info(f"Predictions generated successfully. Shape: {predictions.shape}")
    except Exception as e:
        logger.exception(f"Error during prediction: {e}")
        return None
    
    # Add predictions back to the original relevant data subset
    # Ensure index alignment if session_data was modified
    # Use the index from the features DataFrame which aligns with predictions
    result_df = session_data.loc[features.index, [driver_id_col]].copy()
    result_df["predicted_lap_time"] = predictions
    
    # Find the best predicted lap time for each driver
    best_laps = result_df.loc[result_df.groupby(driver_id_col)["predicted_lap_time"].idxmin()]
    
    # Sort drivers by their best predicted lap time (ascending)
    sorted_drivers = best_laps.sort_values(by="predicted_lap_time")
    
    # Return the sorted qualifying order 
    qualifying_order = sorted_drivers[[driver_id_col, "predicted_lap_time"]].reset_index(drop=True)
    
    logger.info(f"Simulated qualifying order:\n{qualifying_order.head()}") # Log head only
    return qualifying_order

# Removed simulate_past_race function as it was redundant with menu logic

# Example usage (for testing - requires model and training columns)
if __name__ == "__main__":
    # Ensure logger is configured for standalone run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # --- This test requires the model and training columns file to exist --- 
    # --- Run trainer.py first to generate these --- 
    
    try:
        # Load a sample of the cleaned data for testing
        full_session_data = pd.read_csv("data/processed/cleaned_f1_data.csv")
        logger.info("Test data loaded.")
        
        # Simulate for a specific race weekend found in the data
        # Example: Find the first race weekend
        if "race_weekend" in full_session_data.columns:
            test_weekend = full_session_data["race_weekend"].iloc[0]
            logger.info(f"Simulating for test weekend: {test_weekend}")
            weekend_data = full_session_data[full_session_data["race_weekend"] == test_weekend].copy()
            
            # Filter to practice sessions for quali simulation
            practice_data = weekend_data[weekend_data["session_type"] == "practice"]
            
            if not practice_data.empty:
                result = simulate_past_quali(practice_data)
                if result is not None:
                    print(f"\nSimulated qualifying order for {test_weekend}:")
                    print(result)
                else:
                    print(f"Simulation failed for {test_weekend}.")
            else:
                print(f"No practice data found for test weekend {test_weekend}.")
        else:
            logger.error("Test data does not contain \"race_weekend\". Cannot run test.")
                 
    except FileNotFoundError:
        logger.error("Could not find cleaned data file for testing.")
    except Exception as e:
        logger.exception(f"An error occurred during testing: {e}")

