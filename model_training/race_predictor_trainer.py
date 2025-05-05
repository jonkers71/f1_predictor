# File: model_training/race_predictor_trainer.py

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import logging

# Add project root to the sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.logger import setup_logger

logger = setup_logger("RacePredictorTrainer")

# --- Configuration ---
INPUT_DATA = os.path.join(PROJECT_ROOT, "data/processed/cleaned_f1_data.csv")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models/race_predictor")
MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "race_predictor_model.joblib")
FEATURES_FILE = os.path.join(MODEL_OUTPUT_DIR, "race_predictor_features.txt")
SCALER_FILE = os.path.join(MODEL_OUTPUT_DIR, "race_predictor_scaler.joblib")
ENCODERS_FILE = os.path.join(MODEL_OUTPUT_DIR, "race_predictor_encoders.joblib")

TARGET_VARIABLE = "race_position"

# Define features potentially available *before* the race starts
# Excludes race lap times, sector times during the race, etc.
# Includes qualifying results, practice performance summaries, sprint results, etc.
# This list needs careful refinement based on data exploration and availability
FEATURES = [
    # Session/Track Info
    "year",
    "circuit_short_name", # Categorical
    "session_type", # Categorical (used for filtering/grouping, maybe feature)
    # Driver/Team Info
    "driver_number", # Keep as identifier, maybe feature if stable
    "team_name", # Categorical
    # Weather (potentially average/last known before race)
    "air_temperature",
    "humidity",
    "pressure",
    "rainfall",
    "track_temperature",
    "wind_direction",
    "wind_speed",
    # Performance Metrics (Aggregated from Practice/Quali/Sprint before Race)
    # Need to be engineered - e.g., avg_practice_lap_time, quali_position, sprint_finish_pos
    # Placeholder - these need to be created in a preprocessing step
    "quali_pos", # Placeholder - Needs to be added/engineered
    "avg_practice_lap_time", # Placeholder - Needs to be added/engineered
    "sprint_pos", # Placeholder - Needs to be added/engineered
    f"rolling_avg_lap_time_{3}" # Example rolling average from previous sessions
]

CATEGORICAL_FEATURES = [
    "circuit_short_name",
    "team_name",
    "session_type" # If used as feature
]

# --- End Configuration ---

def engineer_pre_race_features(df):
    """Engineers features available before the race starts."""
    logger.info("Engineering pre-race features...")
    df = df.sort_values(by=["race_weekend", "date", "driver_number"])

    # --- 1. Get Qualifying Position --- 
    # Find the qualifying session for each race weekend
    quali_sessions = df[df["session_type"] == "qualifying"].copy()
    # Assuming lap_duration is the qualifying time, find the best lap per driver
    # Note: This is simplified. Real quali has Q1/Q2/Q3. Using best lap overall.
    if not quali_sessions.empty and "lap_duration" in quali_sessions.columns:
        quali_sessions["quali_lap_time"] = quali_sessions.groupby(["race_weekend", "driver_number"])["lap_duration"].transform("min")
        # Rank drivers based on their best lap time within each qualifying session
        quali_results = quali_sessions.loc[quali_sessions.groupby(["race_weekend", "driver_number"])["lap_duration"].idxmin()]
        quali_results["quali_pos"] = quali_results.groupby("race_weekend")["quali_lap_time"].rank(method="min").astype(int)
        # Merge quali_pos back to the main dataframe
        df = pd.merge(df, quali_results[["race_weekend", "driver_number", "quali_pos"]], on=["race_weekend", "driver_number"], how="left")
        logger.info("Engineered 'quali_pos' feature.")
    else:
        logger.warning("Could not engineer 'quali_pos'. Qualifying data missing or lap_duration missing.")
        df["quali_pos"] = np.nan

    # --- 2. Get Average Practice Lap Time --- 
    practice_laps = df[df["session_type"] == "practice"].copy()
    if not practice_laps.empty and "lap_duration" in practice_laps.columns:
        # Calculate average lap time per driver per race weekend during practice
        avg_practice_times = practice_laps.groupby(["race_weekend", "driver_number"])["lap_duration"].mean().reset_index()
        avg_practice_times.rename(columns={"lap_duration": "avg_practice_lap_time"}, inplace=True)
        # Merge back
        df = pd.merge(df, avg_practice_times, on=["race_weekend", "driver_number"], how="left")
        logger.info("Engineered 'avg_practice_lap_time' feature.")
    else:
        logger.warning("Could not engineer 'avg_practice_lap_time'. Practice data missing or lap_duration missing.")
        df["avg_practice_lap_time"] = np.nan

    # --- 3. Get Sprint Race Position (if applicable) ---
    sprint_races = df[df["session_type"] == "sprint"].copy()
    if not sprint_races.empty and "race_position" in sprint_races.columns:
        # Use the existing race_position column, but only for sprint sessions
        sprint_results = sprint_races[["race_weekend", "driver_number", "race_position"]].rename(columns={"race_position": "sprint_pos"})
        # Merge back
        df = pd.merge(df, sprint_results, on=["race_weekend", "driver_number"], how="left")
        logger.info("Engineered 'sprint_pos' feature.")
    else:
        logger.warning("Could not engineer 'sprint_pos'. Sprint race data or race_position missing.")
        df["sprint_pos"] = np.nan

    # Fill NaNs for engineered features (e.g., if a driver missed quali)
    # Use a high value for positions (e.g., 99) and maybe mean/median for times
    df["quali_pos"].fillna(99, inplace=True)
    df["sprint_pos"].fillna(99, inplace=True)
    median_practice_time = df["avg_practice_lap_time"].median()
    df["avg_practice_lap_time"].fillna(median_practice_time if pd.notna(median_practice_time) else 999, inplace=True)

    logger.info("Finished engineering pre-race features.")
    return df

def main():
    """Loads data, trains the race predictor model, and saves it."""
    logger.info("Starting Race Predictor Training Pipeline...")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # --- 1. Load Data ---
    try:
        logger.info(f"Loading data from {INPUT_DATA}")
        df = pd.read_csv(INPUT_DATA, low_memory=False)
        logger.info(f"Loaded {len(df)} rows.")
    except FileNotFoundError:
        logger.error(f"Input data file not found: {INPUT_DATA}")
        return
    except Exception as e:
        logger.exception(f"Error loading data: {e}")
        return

    # --- 2. Feature Engineering ---
    df = engineer_pre_race_features(df)

    # --- 3. Prepare Data for Modeling ---
    # Filter for actual race sessions only, as we predict race outcomes
    df_race = df[df["session_type"] == "race"].copy()
    logger.info(f"Filtered for race sessions. Rows remaining: {len(df_race)}")

    # Drop rows where target is missing (e.g., race not yet run or result unknown)
    df_race.dropna(subset=[TARGET_VARIABLE], inplace=True)
    # Also drop rows where key engineered features might be missing if crucial
    df_race.dropna(subset=["quali_pos", "avg_practice_lap_time"], inplace=True)
    logger.info(f"Rows after dropping NaNs in target/key features: {len(df_race)}")

    if df_race.empty:
        logger.error("No valid race data available for training after filtering. Exiting.")
        return

    # Define final feature list including engineered ones
    final_features = [
        "year", "circuit_short_name", "team_name", "driver_number", # Identifiers/Categorical
        "air_temperature", "humidity", "pressure", "rainfall", "track_temperature", "wind_direction", "wind_speed", # Weather
        "quali_pos", "avg_practice_lap_time", "sprint_pos" # Engineered performance
    ]
    # Ensure all selected features exist in the dataframe
    missing_cols = [f for f in final_features if f not in df_race.columns]
    if missing_cols:
        logger.error(f"Missing required feature columns after processing: {missing_cols}. Exiting.")
        return
    
    X = df_race[final_features]
    y = df_race[TARGET_VARIABLE]

    # --- 4. Preprocessing ---
    logger.info("Preprocessing data (Label Encoding, Scaling)...")
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            le = LabelEncoder()
            # Handle potential NaN/new categories during transform
            X[col] = X[col].astype(str) # Convert to string to handle mixed types/NaNs
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
            logger.debug(f"Label encoded column: {col}")
        else:
            logger.warning(f"Categorical feature {col} not found in data. Skipping encoding.")

    # Scaling numerical features
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    logger.info("Scaled numeric features.")

    # Save encoders and scaler
    try:
        joblib.dump(encoders, ENCODERS_FILE)
        joblib.dump(scaler, SCALER_FILE)
        logger.info(f"Saved encoders to {ENCODERS_FILE} and scaler to {SCALER_FILE}")
    except Exception as e:
        logger.exception("Error saving encoders/scaler.")

    # Save the list of features used by the model
    try:
        with open(FEATURES_FILE, "w") as f:
            for feature in final_features:
                f.write(f"{feature}\n")
        logger.info(f"Saved feature list to {FEATURES_FILE}")
    except IOError as e:
        logger.error(f"Error saving feature list: {e}")

    # --- 5. Train/Test Split ---
    # Simple random split for now, consider time-based split later
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train/Test split: {len(X_train)} train, {len(X_test)} test samples.")

    # --- 6. Model Training (XGBoost Regressor) ---
    logger.info("Training XGBoost Regressor model...")
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }
    
    # Use Mean Absolute Error as the scoring metric
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                               scoring=mae_scorer, cv=3, verbose=1, n_jobs=-1)
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        logger.info(f"GridSearchCV complete. Best parameters: {grid_search.best_params_}")
        logger.info(f"Best MAE score on CV: {-grid_search.best_score_:.4f}") # Score is negative MAE
    except Exception as e:
        logger.exception(f"Error during GridSearchCV: {e}")
        # Fallback to default model if grid search fails
        logger.warning("GridSearchCV failed. Training with default parameters.")
        best_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        best_model.fit(X_train, y_train)

    # --- 7. Evaluation ---
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Model evaluation on test set: MAE = {mae:.4f}")
    # Note: Lower MAE is better. An MAE of X means the model's position prediction is off by X positions on average.

    # --- 8. Save Model ---
    try:
        joblib.dump(best_model, MODEL_FILE)
        logger.info(f"Trained model saved to {MODEL_FILE}")
    except Exception as e:
        logger.exception(f"Error saving model: {e}")

    logger.info("Race Predictor Training Pipeline finished.")

if __name__ == "__main__":
    main()

