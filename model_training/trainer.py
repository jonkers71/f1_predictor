# File: model/trainer.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import joblib

os.system('clear')

# Directories
MODEL_PATH = 'models/xgb_best_model.json'
VISUALS_DIR = 'visuals'
LOG_PATH = 'logs/debug.log'

os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("XGBoostTrainer")

# Load the cleaned dataset
data_path = 'data/processed/cleaned_f1_data.csv'
if not os.path.exists(data_path):
    logger.error(f"Data file not found: {data_path}")
    exit(1)

logger.info("Loading data from: %s", data_path)
data = pd.read_csv(data_path)

# Define all desired features
desired_features = [
    'duration_sector_1', 'duration_sector_2', 'duration_sector_3',
    'i1_speed', 'i2_speed', 'st_speed',
    'segments_sector_1', 'segments_sector_2', 'segments_sector_3',
    'air_temperature', 'humidity', 'pressure', 'rainfall',
    'track_temperature', 'wind_direction', 'wind_speed',
    'lap_number', 'is_pit_out_lap',
    'car_model', 'team_name', 'session_type'
]

# Filter valid features
features = []
missing_features = set()

for f in desired_features:
    if f in data.columns:
        features.append(f)
    else:
        missing_features.add(f)
        logger.warning(f"Feature '{f}' not found in data and will be skipped.")

if missing_features:
    logger.warning(f"Missing features: {missing_features}, skipping them.")

if not features:
    logger.error("No valid features found in the dataset.")
    exit(1)

target = 'lap_duration'
if target not in data.columns:
    logger.error(f"Target column '{target}' not found in the dataset.")
    exit(1)

# Prepare features and target
X = pd.get_dummies(data[features], drop_first=True)
y = data[target]

# Sanitize feature names to ensure they are valid for XGBoost
X.columns = X.columns.str.replace('[', '_').str.replace(']', '_').str.replace('<', '_')

# Split the data
logger.info("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train default model
logger.info("Training base XGBoost model...")
xg_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=5,
    alpha=10,
    n_estimators=100
)

logger.info("Fitting the base model...")
xg_reg.fit(X_train, y_train)
logger.info("Base Model RMSE: %.4f", np.sqrt(mean_squared_error(y_test, xg_reg.predict(X_test))))

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'colsample_bytree': [0.3, 0.5]
}

logger.info("Running GridSearchCV for hyperparameter tuning...")

# Parallelize GridSearchCV to use all cores
grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror'),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,  # Display progress
    n_jobs=-1  # Use all available CPU cores for faster processing
)

# Run GridSearchCV
grid_search.fit(X_train, y_train)

# Log the best parameters found
logger.info(f"Best parameters found: {grid_search.best_params_}")

# Evaluate the best model
best_xg_reg = grid_search.best_estimator_
y_pred_best = best_xg_reg.predict(X_test)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
logger.info(f"Best Model RMSE: {rmse_best:.4f}")

# Save the best model
logger.info(f"Saving the best model to {MODEL_PATH}...")
best_xg_reg.save_model(MODEL_PATH)

# Feature importance plot
logger.info("Generating feature importance plot...")
xgb.plot_importance(best_xg_reg)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'feature_importance.png'))
plt.close()

# Error distribution plot
logger.info("Generating prediction error distribution plot...")
errors = y_test - y_pred_best
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, edgecolor='black')
plt.title("Prediction Error Distribution")
plt.xlabel("Error (s)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'error_distribution.png'))
plt.close()

# Example prediction
logger.info("Making a prediction for a new lap...")
new_data = pd.DataFrame({
    'air_temperature': [30],
    'humidity': [60],
    'pressure': [1013],
    'track_temperature': [45],
    'wind_speed': [5],
    'lap_number': [5],
    'is_pit_out_lap': [0],
    'car_model': ['model_x'],
    'team_name': ['team_a'],
    'session_type': ['qualifying']
})

new_data_encoded = pd.get_dummies(new_data, drop_first=True)
new_data_encoded = new_data_encoded.reindex(columns=X.columns, fill_value=0)
predicted_lap_time = best_xg_reg.predict(new_data_encoded)
logger.info(f"Predicted lap time: {predicted_lap_time[0]:.4f}")

# Final message to indicate completion
logger.info("Model training and prediction completed successfully.")
print("Model training and prediction completed successfully.")
