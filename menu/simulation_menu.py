# File: menu/simulation_menu.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from utils.logger import setup_logger
from menu.menu_utils import format_lap_time # Import from utils

# Import the simulation function (adjust path if needed)
from model_training.simulate_trainer import simulate_past_quali

logger = setup_logger("SimulationMenu")

def simulate_past_qualifying_menu(project_root, driver_map):
    """Handles the user interaction for simulating past qualifying."""
    logger.info("User selected: Simulate Past Qualifying")
    data_path = os.path.join(project_root, "data/processed/cleaned_f1_data.csv")
    
    try:
        logger.info(f"Loading data from {data_path}")
        session_data = pd.read_csv(data_path)
        logger.info(f"Data loaded. Shape: {session_data.shape}")
    except FileNotFoundError:
        print(f"\n❌ Could not find cleaned session data at \n{data_path}\n. ")
        print("Please run the data processing (Option 4) first.")
        logger.error(f"Cleaned data file not found: {data_path}")
        return
    except Exception as e:
        print(f"\n❌ An error occurred while loading data: {e}")
        logger.exception(f"Failed to load data from {data_path}")
        return

    # --- Validate Data --- 
    required_cols = ["race_weekend", "session_type", "date"]
    missing_cols = [col for col in required_cols if col not in session_data.columns]
    if missing_cols:
        print(f"\n❌ Required columns missing in the data: {missing_cols}. Cannot proceed.")
        logger.error(f"Missing required columns for simulation: {missing_cols}")
        return
        
    try:
        session_data["date_dt"] = pd.to_datetime(session_data["date"], errors="coerce")
        session_data = session_data.dropna(subset=["date_dt", "race_weekend"])
    except Exception as e:
        print(f"\n❌ Error processing date column: {e}")
        logger.exception("Error during date conversion or dropping NA.")
        return
        
    if session_data.empty:
        print("\n❌ No valid session data found after handling dates and race weekends.")
        logger.error("Data is empty after initial validation.")
        return

    # --- Select Race Weekend --- 
    try:
        unique_weekends = session_data.loc[session_data.groupby("race_weekend")["date_dt"].idxmax()]
        unique_weekends = unique_weekends.sort_values("date_dt", ascending=False)
        available_weekends = unique_weekends["race_weekend"].unique().tolist()
    except Exception as e:
        print(f"\n❌ Error identifying unique race weekends: {e}")
        logger.exception("Error grouping or sorting race weekends.")
        return

    if not available_weekends:
        print("\n❌ No valid race weekends found in the dataset.")
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
    try:
        weekend_data = session_data[session_data["race_weekend"] == selected_weekend].copy()
        # Define sessions relevant for *predicting* qualifying
        quali_sim_sessions = ["Practice 1", "Practice 2", "Practice 3"]
        simulation_input_data = weekend_data[weekend_data["session_name"].isin(quali_sim_sessions)] # Use session_name if available
        
        # Fallback if session_name isn't reliable or present
        if simulation_input_data.empty and 'session_type' in weekend_data.columns:
             logger.warning("Falling back to using 'session_type' = 'practice' for filtering.")
             simulation_input_data = weekend_data[weekend_data["session_type"].str.lower() == 'practice']

    except KeyError as e:
        print(f"\n❌ Error accessing column for filtering: {e}. Check your 'cleaned_f1_data.csv'.")
        logger.exception(f"KeyError during data filtering for simulation: {e}")
        return
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during data filtering: {e}")
        logger.exception("Unexpected error during simulation data filtering.")
        return
        
    if simulation_input_data.empty:
        print(f"\n❌ No relevant practice session data found for {selected_weekend}. Cannot simulate qualifying.")
        logger.warning(f"No practice session data found for {selected_weekend} using session_name or session_type.")
        return
        
    logger.info(f"Filtered data for {selected_weekend} qualifying simulation. Input rows: {len(simulation_input_data)}")

    # --- Call Simulation Function --- 
    print("🚀 Running simulation...")
    try:
        simulated_order = simulate_past_quali(simulation_input_data)
    except Exception as e:
        print(f"\n❌ An error occurred during the simulation execution: {e}")
        logger.exception(f"Error calling simulate_past_quali for {selected_weekend}")
        simulated_order = None # Ensure it's None if simulation fails

    # --- Display Results --- 
    if simulated_order is not None and not simulated_order.empty:
        print(f"\n🏁 Simulated Qualifying Order for {selected_weekend} (Top 10):")
        
        top_10_results = simulated_order.head(10)
        
        print("Pos. | No. | Driver Name        | Predicted Time")
        print("---- | --- | ------------------ | --------------")
        
        for index, row in top_10_results.iterrows():
            position = index + 1
            try:
                driver_num = int(row["driver_number"])
                # Look up driver name from the map
                driver_name = driver_map.get(driver_num, "(Unknown)") # Use map, default to (Unknown)
                formatted_time = format_lap_time(row["predicted_lap_time"])
                
                # Print with driver name, ensure alignment (adjust padding if needed)
                print(f"{position:<4} | {driver_num:<3} | {driver_name:<18} | {formatted_time}")
            except KeyError as e:
                print(f"Error accessing result column: {e}")
                logger.error(f"KeyError accessing simulation results columns: {e}")
            except ValueError as e:
                 print(f"Error converting driver number: {e}")
                 logger.error(f"ValueError converting driver number in results: {e}")
            
        logger.info(f"Simulation successful for {selected_weekend}. Displayed top 10.")
        
    elif simulated_order is not None and simulated_order.empty:
        print("\nℹ️ Simulation ran but produced no results (empty order).")
        logger.warning(f"Simulation for {selected_weekend} resulted in an empty DataFrame.")
    else:
        # Error message already printed in the simulation execution block
        print(f"\n❌ Simulation failed for {selected_weekend}. Check logs for details.")
        # Logger message already logged in the simulation execution block

