# File: menu/data_menu.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import subprocess
# import select # No longer needed for this approach
from utils.logger import setup_logger

logger = setup_logger("DataMenu")

def run_script(script_path, project_root):
    """Runs a Python script using subprocess, allowing real-time output directly to console."""
    full_script_path = os.path.join(project_root, script_path)
    logger.info(f"Attempting to run script: {full_script_path} from {project_root}")
    try:
        # Use sys.executable to ensure the same Python interpreter is used
        # Use Popen and let it inherit stdout/stderr for real-time output
        process = subprocess.Popen(
            [sys.executable, "-u", full_script_path], # -u for unbuffered output
            cwd=project_root, # Run from the project root directory
            # stdout=subprocess.PIPE, # Remove to inherit stdout
            # stderr=subprocess.PIPE, # Remove to inherit stderr
            text=True,
            encoding="utf-8"
            # bufsize=1 # Not needed when not capturing
        )

        # Wait for the process to complete and get the return code
        return_code = process.wait()

        if return_code == 0:
            logger.info(f"Script {script_path} finished successfully. RC: {return_code}")
            # No need to log stdout/stderr here as it went directly to console
            print(f"\n✅ Script \t{script_path}\t finished successfully.")
            return True
        else:
            logger.error(f"Script {script_path} failed with return code {return_code}")
            # Error messages should have been printed directly by the script
            print(f"\n❌ Error running script {script_path}. Return Code: {return_code}")
            return False

    except FileNotFoundError:
        print(f"\n❌ Error: Script not found at {full_script_path}")
        logger.error(f"Script not found: {full_script_path}")
        return False
    except Exception as e:
        print(f"\n❌ An unexpected error occurred running {script_path}: {e}")
        logger.exception(f"Unexpected error during {script_path} execution")
        return False

def data_management_menu(project_root):
    """Displays the data management sub-menu and handles user choices."""
    while True:
        print("\n--- Data Management & Processing --- ")
        print("1. Fetch Session Data (API, 2023-Present)")
        print("2. Fetch Lap Data (API, 2023-Present - Long Process)")
        print("3. Fetch Weather Data (API, 2023-Present)")
        print("4. Process Raw Data (Build Dataset)")
        print("5. Back to Main Menu")
        
        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            print("\n⏳ Running Session Fetcher...")
            run_script("data_pipeline/session_fetcher.py", project_root)
        elif choice == "2":
            print("\n⏳ Running Lap Collector (This may take a long time!)...")
            run_script("data_pipeline/lap_collector.py", project_root)
        elif choice == "3":
            print("\n⏳ Running Weather Fetcher...")
            run_script("data_pipeline/weather_fetcher.py", project_root)
        elif choice == "4":
            print("\n⏳ Running Dataset Builder...")
            # Check if dataset_builder.py exists, if not, notify user
            builder_path = os.path.join(project_root, "data_pipeline/dataset_builder.py")
            if os.path.exists(builder_path):
                run_script("data_pipeline/dataset_builder.py", project_root)
            else:
                print(f"\n⚠️ Warning: Dataset builder script not found at {builder_path}. Skipping.")
                logger.warning(f"Dataset builder script not found at {builder_path}. Skipping execution.")
        elif choice == "5":
            print("Returning to main menu...")
            break
        else:
            print("Invalid selection. Please try again.")

def train_model_menu_option(project_root):
    """Handles the user interaction for training the model."""
    logger.info("User selected: Train Model")
    print("\n🏋️ Running Model Trainer...")
    print("   NOTE: This uses the data previously processed by the Dataset Builder.")
    print("   Ensure you have run \'Process Raw Data\' (Option 4) if you have fetched new raw data.")
    print("-" * 20)
    
    # Check if trainer.py exists
    trainer_path = os.path.join(project_root, "model_training/trainer.py")
    if not os.path.exists(trainer_path):
         print(f"\n❌ Error: Model training script not found at {trainer_path}. Cannot proceed.")
         logger.error(f"Model training script not found at {trainer_path}. Cannot proceed.")
         return

    success = run_script("model_training/trainer.py", project_root)
    
    if success:
        print("\n✅ Model training script finished.")
    else:
        print("\n❌ Model training script failed. Check logs or output above for details.")

