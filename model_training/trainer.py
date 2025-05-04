# File: model_training/trainer.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import joblib # Using joblib might be better for python objects, but let's stick to text for columns for simplicity

# --- Configuration --- 
# Ideally, use paths from a central config like settings.py
# Assuming script is run from project root or paths are relative to project root
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_best_model.json')
TRAINING_COLUMNS_PATH = os.path.join(MODEL_DIR, 'training_columns.txt')
VISUALS_DIR = 'visuals'
LOG_DIR = 'logs'
LOG_PATH = os.path.join(LOG_DIR, 'debug.log')
DATA_PATH = 'data/processed/cleaned_f1_data.csv'

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logger Setup --- 
# Use the shared logger setup if available, otherwise configure basic logging
try:
    from utils.logger import setup_logger
    logger = setup_logger("XGBoostTrainer")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("XGBoostTrainer")

def save_training_columns(columns, path):
    """Saves the list of training columns to a text file."""
    try:
        with open(path, 'w') as f:
            for col in columns:
                f.write(f"{col}\n")
        logger.info(f"Training columns saved to {path}")
    except Exception as e:
        logger.exception(f"Failed to save training columns to {path}: {e}")

def train_model(data_path=DATA_PATH):
    """Loads data, trains an XGBoost model with hyperparameter tuning, and saves it."""
    
    # --- Load Data --- 
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return False # Indicate failure

    logger.info(f"Loading data from: {data_path}")
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
    except Exception as e:
        logger.exception(f"Failed to load data from {data_path}: {e}")
        return False
        
    # Check for empty data
    if data.empty:
        logger.error("Loaded data is empty. Cannot train model.")
        return False

    # --- Feature Engineering & Selection --- 
    # Define all desired features (consider moving to config)
    # Added 'rolling_avg_lap_time_3' based on dataset_builder update
    desired_features = [
        'duration_sector_1', 'duration_sector_2', 'duration_sector_3',
        'i1_speed', 'i2_speed', 'st_speed',
        # 'segments_sector_1', 'segments_sector_2', 'segments_sector_3', # Example: Assume these are complex objects/lists and need specific handling or exclusion
        'air_temperature', 'humidity', 'pressure', 'rainfall',
        'track_temperature', 'wind_direction', 'wind_speed',
        'lap_number', 'is_pit_out_lap',
        'rolling_avg_lap_time_3', # <-- Added new feature
        # Categorical features to be one-hot encoded
        'driver_number', # Keep driver ID if useful for grouping/analysis later, but maybe not as direct feature?
        'team_name', 
        'session_type', 
        'circuit_key', # Example: Add circuit info if available and relevant
        'compound', # Example: Add tire compound if available
        # 'car_model' # Example: If car_model is too granular or redundant with team_name
    ]
    
    target = 'lap_duration'

    # Validate target column
    if target not in data.columns:
        logger.error(f"Target column '{target}' not found in the dataset.")
        return False
        
    # Filter to features present in the loaded data
    features_present = []
    missing_in_data = []
    for f in desired_features:
        if f in data.columns:
            features_present.append(f)
        else:
            missing_in_data.append(f)
            logger.warning(f"Desired feature '{f}' not found in data and will be skipped.")

    if not features_present:
        logger.error("No valid features found in the dataset after filtering.")
        return False
        
    logger.info(f"Using features: {features_present}")
    
    # Handle missing values in features and target before splitting
    # Simple strategy: drop rows with missing target
    data.dropna(subset=[target], inplace=True)
    logger.info(f"Rows after dropping NaN in target: {len(data)}")
    
    # Impute or handle NaNs in features (Example: fill numeric with median, categorical with mode or 'Unknown')
    for col in features_present:
        if pd.api.types.is_numeric_dtype(data[col]):
            if data[col].isnull().any():
                median_val = data[col].median()
                # Check if median is NaN (can happen if all values are NaN)
                if pd.isna(median_val):
                    median_val = 0 # Fallback value
                    logger.warning(f"Median for numeric feature '{col}' is NaN, filling NaNs with 0.")
                data[col].fillna(median_val, inplace=True)
                logger.info(f"Filled NaNs in numeric feature '{col}' with median ({median_val})")
        elif pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]):
             if data[col].isnull().any():
                mode_val = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                data[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled NaNs in categorical feature '{col}' with mode ('{mode_val}')")

    # Prepare features (X) and target (y)
    X = data[features_present]
    y = data[target]
    
    # Apply one-hot encoding (after handling NaNs)
    logger.info("Applying one-hot encoding...")
    X_encoded = pd.get_dummies(X, drop_first=True) # drop_first=True helps reduce multicollinearity

    # Sanitize feature names for XGBoost (after encoding)
    X_encoded.columns = X_encoded.columns.str.replace('[', '_', regex=False).str.replace(']', '_', regex=False).str.replace('<', '_', regex=False)
    final_training_columns = X_encoded.columns.tolist()
    logger.info(f"Final features after encoding and sanitization ({len(final_training_columns)}): {final_training_columns[:10]}...")

    # Save the final training columns
    save_training_columns(final_training_columns, TRAINING_COLUMNS_PATH)

    # --- Data Splitting --- 
    logger.info("Splitting data into training and testing sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # --- Model Training & Tuning --- 
    # Define parameter grid for GridSearchCV
    param_grid = {
        'max_depth': [5, 7], # Reduced complexity slightly
        'learning_rate': [0.05, 0.1], # Focused range
        'n_estimators': [100, 200],
        'colsample_bytree': [0.5, 0.7], # Fraction of features used per tree
        'subsample': [0.7, 0.9], # Fraction of samples used per tree
        'gamma': [0, 0.1], # Minimum loss reduction to make a split
        'reg_alpha': [0, 0.1] # L1 regularization
    }

    logger.info("Running GridSearchCV for hyperparameter tuning...")
    xgb_estimator = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1) # Use all cores within XGBoost
    
    # Use fewer CV folds if dataset is very large or time is critical
    cv_folds = 3 
    grid_search = GridSearchCV(
        estimator=xgb_estimator,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error', # Use RMSE directly
        cv=cv_folds,
        verbose=1, # Display progress
        n_jobs=1 # Let XGBoost handle internal parallelization (n_jobs=-1 in XGBRegressor)
    )

    try:
        grid_search.fit(X_train, y_train)
        logger.info(f"GridSearchCV completed.")
    except Exception as e:
        logger.exception(f"Error during GridSearchCV: {e}")
        return False

    # Log the best parameters and score
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    best_score = -grid_search.best_score_ # Score is negative RMSE
    logger.info(f"Best CV RMSE: {best_score:.4f}")

    # --- Evaluate Best Model --- 
    best_xg_reg = grid_search.best_estimator_
    y_pred_test = best_xg_reg.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    logger.info(f"Test Set RMSE of Best Model: {test_rmse:.4f}")

    # --- Save Model --- 
    logger.info(f"Saving the best model to {MODEL_PATH}...")
    try:
        best_xg_reg.save_model(MODEL_PATH)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.exception(f"Failed to save model to {MODEL_PATH}: {e}")
        # Continue to plotting even if saving fails

    # --- Visualizations --- 
    try:
        # Feature Importance Plot
        logger.info("Generating feature importance plot...")
        fig, ax = plt.subplots(figsize=(10, max(6, len(final_training_columns) // 3))) # Adjust height based on num features
        xgb.plot_importance(best_xg_reg, ax=ax, max_num_features=20) # Show top 20 features
        plt.title('XGBoost Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, 'feature_importance.png'))
        plt.close(fig)
        logger.info(f"Feature importance plot saved to {os.path.join(VISUALS_DIR, 'feature_importance.png')}")

        # Prediction Error Distribution Plot
        logger.info("Generating prediction error distribution plot...")
        errors = y_test - y_pred_test
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors, bins=50, edgecolor='black')
        ax.set_title("Prediction Error Distribution (Test Set)")
        ax.set_xlabel("Error (Actual - Predicted Lap Duration in s)")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, 'error_distribution.png'))
        plt.close(fig)
        logger.info(f"Error distribution plot saved to {os.path.join(VISUALS_DIR, 'error_distribution.png')}")

    except Exception as e:
        logger.exception(f"Failed to generate or save plots: {e}")

    logger.info("Model training process completed.")
    return True # Indicate success

# --- Main Execution Guard --- 
if __name__ == "__main__":
    logger.info("Starting model training script...")
    success = train_model()
    if success:
        logger.info("Model training finished successfully.")
        print("\nModel training finished successfully.")
    else:
        logger.error("Model training failed.")
        print("\nModel training failed. Check logs/debug.log for details.")

