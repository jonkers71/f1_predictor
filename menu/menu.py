# File: menu/menu.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import setup_logger
import pandas as pd
import math # Needed for time formatting

# Make sure to import the correct simulation function
from model_training.simulate_trainer import simulate_past_quali # Assuming this is the target for now

logger = setup_logger("Menu")

# --- Helper Functions ---

def load_driver_map(path="data/processed/driver_map.csv"):
    """Loads the driver number to name mapping from CSV."""
    try:
        driver_df = pd.read_csv(path)
        # Create a dictionary for quick lookup: driver_number -> full_name
        driver_map = pd.Series(driver_df.full_name.values, index=driver_df.driver_number).to_dict()
        logger.info(f"Successfully loaded driver map from {path}. {len(driver_map)} drivers found.")
        return driver_map
    except FileNotFoundError:
        logger.warning(f"Driver map file not found at {path}. Driver names will not be displayed.")
        return {}
    except Exception as e:
        logger.exception(f"Error loading driver map from {path}: {e}")
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

# --- Menu Display and Options ---

def display_menu():
    """Displays the main menu and handles user choices."""
    # Load driver map once when menu starts (or reload if needed)
    driver_map = load_driver_map()
    
    while True:
        print("\n📊 F1 Prediction System Menu")
        print("1. Predict Upcoming Qualifying (Not Implemented)")
        print("2. Predict Upcoming Race (Not Implemented)")
        print("3. Simulate Past Qualifying")
        print("4. Update Training Data & Retrain Model")
        print("5. Exit")

        choice = input("Select an option (1–5): ").strip()

        if choice == "1":
            predict_qualifying()
        elif choice == "2":
            predict_race()
        elif choice == "3":
            # Pass the loaded driver map to the simulation menu
            simulate_past_qualifying_menu(driver_map)
        elif choice == "4":
            update_and_train()
        elif choice == "5":
            print("Exiting...")
            logger.info("User exited program.")
            break
        else:
            print("Invalid selection. Please try again.")

def predict_qualifying():
    logger.info("User selected: Predict Upcoming Qualifying")
    print("🔧 Predicting upcoming qualifying... (module not yet implemented)")

def predict_race():
    logger.info("User selected: Predict Upcoming Race")
    print("🔧 Predicting upcoming race... (module not yet implemented)")

def simulate_past_qualifying_menu(driver_map):
    """Handles the user interaction for simulating past qualifying."""
    logger.info("User selected: Simulate Past Qualifying")
    data_path = "data/processed/cleaned_f1_data.csv"
    
    try:
        logger.info(f"Loading data from {data_path}")
        session_data = pd.read_csv(data_path)
        logger.info(f"Data loaded. Shape: {session_data.shape}")
    except FileNotFoundError:
        print(f"❌ Could not find cleaned session data at \n{data_path}\n. ")
        print("Please run the data pipeline (e.g., using option 4) or ensure the file exists.")
        logger.error(f"Cleaned data file not found: {data_path}")
        return
    except Exception as e:
        print(f"❌ An error occurred while loading data: {e}")
        logger.exception(f"Failed to load data from {data_path}")
        return

    # --- Validate Data --- 
    required_cols = ["race_weekend", "session_type", "date"]
    missing_cols = [col for col in required_cols if col not in session_data.columns]
    if missing_cols:
        print(f"❌ Required columns missing in the data: {missing_cols}. Cannot proceed.")
        logger.error(f"Missing required columns for simulation: {missing_cols}")
        return
        
    session_data["date_dt"] = pd.to_datetime(session_data["date"], errors="coerce")
    session_data = session_data.dropna(subset=["date_dt", "race_weekend"])
    
    if session_data.empty:
        print("❌ No valid session data found after handling dates and race weekends.")
        logger.error("Data is empty after initial validation.")
        return

    # --- Select Race Weekend --- 
    unique_weekends = session_data.loc[session_data.groupby("race_weekend")["date_dt"].idxmax()]
    unique_weekends = unique_weekends.sort_values("date_dt", ascending=False)
    available_weekends = unique_weekends["race_weekend"].unique().tolist()

    if not available_weekends:
        print("❌ No valid race weekends found in the dataset.")
        logger.error("No unique race weekends could be extracted.")
        return

    num_to_display = min(10, len(available_weekends))
    print("\n📅 Available Race Weekends for Simulation (most recent first):")
    display_list = available_weekends[:num_to_display]
    for i, weekend in enumerate(display_list, start=1):
        print(f"{i}. {weekend}")

    try:
        weekend_choice_idx = int(input(f"\nEnter the number of the race weekend to simulate (1–{num_to_display}): ")) - 1
        if 0 <= weekend_choice_idx < num_to_display:
            selected_weekend = display_list[weekend_choice_idx]
            logger.info(f"User selected race weekend: {selected_weekend}")
        else:
            print("Invalid selection.")
            logger.warning(f"Invalid weekend choice index: {weekend_choice_idx + 1}")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        logger.warning("Non-integer input for weekend choice.")
        return
        
    # --- Filter Data for Selected Weekend and Qualifying Simulation --- 
    print(f"\n🔧 Preparing data for {selected_weekend} Qualifying simulation...")
    weekend_data = session_data[session_data["race_weekend"] == selected_weekend].copy()
    quali_sim_sessions = ["practice"]
    simulation_input_data = weekend_data[weekend_data["session_type"].isin(quali_sim_sessions)]
    
    if simulation_input_data.empty:
        print(f"❌ No relevant session data (Practice) found for {selected_weekend}. Cannot simulate qualifying.")
        logger.warning(f"No practice session data found for {selected_weekend}")
        return
        
    logger.info(f"Filtered data for {selected_weekend} qualifying simulation. Input rows: {len(simulation_input_data)}")

    # --- Call Simulation Function --- 
    print("🚀 Running simulation...")
    simulated_order = simulate_past_quali(simulation_input_data)

    # --- Display Results --- 
    if simulated_order is not None and not simulated_order.empty:
        print(f"\n🏁 Simulated Qualifying Order for {selected_weekend} (Top 10):")
        
        top_10_results = simulated_order.head(10)
        
        print("Pos. | No. | Driver Name        | Predicted Time")
        print("---- | --- | ------------------ | --------------")
        
        for index, row in top_10_results.iterrows():
            position = index + 1
            driver_num = int(row["driver_number"])
            # Look up driver name from the map
            driver_name = driver_map.get(driver_num, "(Unknown)") # Use map, default to (Unknown)
            formatted_time = format_lap_time(row["predicted_lap_time"])
            
            # Print with driver name, ensure alignment (adjust padding if needed)
            print(f"{position:<4} | {driver_num:<3} | {driver_name:<18} | {formatted_time}")
            
        logger.info(f"Simulation successful for {selected_weekend}. Displayed top 10.")
        
    elif simulated_order is not None and simulated_order.empty:
        print("Simulation ran but produced no results (empty order).")
        logger.warning(f"Simulation for {selected_weekend} resulted in an empty DataFrame.")
    else:
        print(f"❌ Simulation failed for {selected_weekend}. Check logs for details.")
        logger.error(f"Simulation function returned None for {selected_weekend}.")

def update_and_train():
    """Triggers the data building and model retraining process."""
    # Note: This does NOT run the raw data fetchers (session_fetcher, lap_collector)
    # Those should be run separately when needed.
    logger.info("User selected: Update Training Data & Retrain Model")
    print("\n🔄 Processing Existing Raw Data & Retraining Model...")
    print("   NOTE: This option processes data currently in the 'data/raw' folder.")
    print("   To fetch NEW or UPDATED data from the OpenF1 API (e.g., recent races or older years),")
    print("   you must MANUALLY run the following scripts from the project root directory")
    print("   BEFORE using this option:")
    print("     1. python3.11 data_pipeline/session_fetcher.py")
    print("     2. python3.11 data_pipeline/lap_collector.py")
    # Optional: Add weather fetcher if needed
    # print("     3. python3.11 data_pipeline/weather_fetcher.py")
    print("-" * 20)
    
    print(" -> Running dataset builder...")
    try:
        import subprocess
        process = subprocess.run([sys.executable, "data_pipeline/dataset_builder.py"], capture_output=True, text=True, check=True, encoding="utf-8")
        logger.info("Dataset builder output:\n" + process.stdout)
        if process.stderr:
            logger.warning("Dataset builder stderr:\n" + process.stderr)
        print(" -> Dataset builder finished.")
    except FileNotFoundError:
        print("❌ Error: dataset_builder.py not found.")
        logger.error("dataset_builder.py not found during update_and_train")
        return 
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dataset_builder.py: {e}")
        logger.error(f"Dataset builder failed:\nStdout: {e.stdout}\nStderr: {e.stderr}")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred running dataset_builder: {e}")
        logger.exception("Unexpected error during dataset_builder execution")
        return
        
    print(" -> Running model trainer...")
    try:
        # Ensure trainer is imported AFTER dataset builder runs
        # This assumes trainer.py reads the updated cleaned_f1_data.csv
        from model_training import trainer 
        # Need to reload the module if it was imported previously to pick up changes
        import importlib
        importlib.reload(trainer)
        
        success = trainer.train_model() 
        if success:
            print("✅ Model successfully updated and retrained.")
            logger.info("Model retraining successful.")
        else:
            print("❌ Model retraining failed. Check logs.")
            logger.error("trainer.train_model() returned False.")
            
    except ImportError:
         print("❌ Error importing trainer module.")
         logger.error("Failed to import model_training.trainer")
    except Exception as e:
        print(f"❌ An unexpected error occurred during training: {e}")
        logger.exception("Unexpected error during model training execution")

# --- Main Execution Guard ---
if __name__ == "__main__":
    display_menu()

