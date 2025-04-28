import pandas as pd
import logging
import xgboost as xgb

# Logger setup
logger = logging.getLogger("SimulateTrainer")

def load_trained_model():
    """
    Load the trained model for simulating past qualifying.
    """
    logger.info("Loading trained model...")
    model_path = 'models/xgb_best_model.json'  # Path to the trained model
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

def preprocess_data(session_data):
    """
    Preprocess session data into the feature set for prediction.
    """
    logger.info("Preprocessing session data for simulation...")
    # Example: Apply necessary preprocessing steps like encoding, feature scaling, etc.
    features = pd.get_dummies(session_data, drop_first=True)
    return features

def simulate_past_quali(session_data):
    """
    Simulate past qualifying results based on available session data (P1, P2, P3).
    """
    logger.info("Simulating past qualifying results...")
    
    # Preprocess data into features
    features = preprocess_data(session_data)
    
    # Load trained model
    model = load_trained_model()
    
    # Predict lap times for qualifying
    predictions = model.predict(features)
    
    # Add predictions to the session data
    session_data['predicted_lap_time'] = predictions
    
    # Sort data by predicted lap times (ascending)
    sorted_data = session_data.sort_values(by='predicted_lap_time')
    
    # Return the sorted qualifying order with predicted lap times
    qualifying_order = sorted_data[['driver_name', 'predicted_lap_time']]
    
    logger.info(f"Simulated qualifying order:\n{qualifying_order}")
    return qualifying_order

def simulate_past_race(race_name, session_data):
    """
    Simulate the qualifying for a past race by using available session data (P1, P2, P3).
    """
    logger.info(f"Simulating past qualifying for race: {race_name}")
    
    # Filter session data for the specified race and P1, P2, P3 sessions
    past_race_data = session_data[session_data['race_name'] == race_name]
    past_race_data = past_race_data[past_race_data['session_type'].isin(['P1', 'P2', 'P3'])]
    
    if past_race_data.empty:
        logger.warning(f"No qualifying data found for race: {race_name}.")
        return None
    
    # Simulate the qualifying order
    simulated_order = simulate_past_quali(past_race_data)
    
    return simulated_order

# Example usage (for testing)
if __name__ == "__main__":
    race_name = "Spain 2024"
    session_data = pd.read_csv('data/processed/cleaned_f1_data.csv')  # Replace with actual path
    result = simulate_past_race(race_name, session_data)
    if result is not None:
        print(f"Simulated qualifying order for {race_name}:")
        print(result)
