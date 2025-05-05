# File: menu/main_menu.py
import os
import sys
import importlib # Needed to reload settings

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Import necessary functions and the config setting
from utils.logger import setup_logger, update_logging_config
from config import settings # Import the settings module
from menu.menu_utils import load_driver_map
from menu.simulation_menu import simulate_past_qualifying_menu
# Import the data management and the NEW model training sub-menus
from menu.data_menu import data_management_menu, model_training_menu 
from prediction.race_predictor import predict_upcoming_race # Import the new race predictor function

# Initial logger setup respects the setting from config file
logger = setup_logger("MainMenu")

def display_main_menu(project_root):
    """Displays the main menu and handles user choices."""
    # Load driver map once when menu starts
    driver_map = load_driver_map(project_root)
    
    while True:
        # Reload settings module to get the current LOGGING_ENABLED status
        importlib.reload(settings)
        current_logging_status = "Enabled" if settings.LOGGING_ENABLED else "Disabled"
        
        print("\n--- 📊 F1 Prediction System Main Menu ---")
        print("1. Predict Upcoming Qualifying (Not Implemented)")
        print("2. Predict Upcoming Race")
        print("3. Simulate Past Qualifying")
        print("4. Data Management & Processing")
        print("5. Train Models") # Updated text slightly
        print(f"6. Toggle Logging (Currently: {current_logging_status})")
        print("7. Exit")

        choice = input("Select an option (1–7): ").strip()

        if choice == "1":
            predict_qualifying()
        elif choice == "2":
            # Call the imported race prediction function
            predict_upcoming_race()
        elif choice == "3":
            # Pass the loaded driver map and project root
            simulate_past_qualifying_menu(project_root, driver_map)
        elif choice == "4":
            # Call the data management sub-menu
            data_management_menu(project_root)
        elif choice == "5":
            # Call the NEW model training sub-menu
            model_training_menu(project_root)
        elif choice == "6":
            # Toggle logging
            toggle_logging()
        elif choice == "7":
            print("Exiting...")
            logger.info("User exited program.")
            break
        else:
            print("Invalid selection. Please try again.")

def predict_qualifying():
    logger.info("User selected: Predict Upcoming Qualifying")
    print("\n🔧 Predicting upcoming qualifying... (module not yet implemented)")

# Removed the placeholder predict_race() function

def toggle_logging():
    """Toggles the logging status and updates the config file."""
    # Reload settings to ensure we have the latest state before toggling
    importlib.reload(settings)
    new_status = not settings.LOGGING_ENABLED
    if update_logging_config(new_status):
        status_str = "Enabled" if new_status else "Disabled"
        print(f"\nLogging has been {status_str}.")
        logger.info(f"Logging status changed to: {new_status}")
        print("Note: Logging changes will take full effect when scripts are run next.")
    else:
        print("\nError updating logging configuration.")
        logger.error("Failed to update logging configuration in settings.py")


