# File: menu/data_menu.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import subprocess
from utils.logger import setup_logger

logger = setup_logger("DataMenu")

def run_script(script_path, project_root):
    """Runs a Python script using subprocess, allowing real-time output directly to console."""
    full_script_path = os.path.join(project_root, script_path)
    logger.info(f"Attempting to run script: {full_script_path} from {project_root}")
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", full_script_path],
            cwd=project_root,
            text=True,
            encoding="utf-8"
        )
        return_code = process.wait()

        if return_code == 0:
            logger.info(f"Script {script_path} finished successfully. RC: {return_code}")
            print(f"\n✅ Script \t{script_path}\t finished successfully.")
            return True
        else:
            logger.error(f"Script {script_path} failed with return code {return_code}")
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
        print("\n--- 📊 Data Management & Processing --- ")
        print("1. Fetch Session Data (API, 2023-Present)")
        print("2. Fetch Lap Data (API, 2023-Present - Long Process)")
        print("3. Fetch Weather Data (API, 2023-Present)")
        print("4. Fetch Race Results (Ergast API)")
        print("5. Process Raw Data (Build Dataset & Optionally Retrain Models)")
        print("6. Back to Main Menu")
        
        choice = input("Select an option (1-6): ").strip()

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
            print("\n⏳ Running Ergast Fetcher (Race Results)...")
            run_script("data_pipeline/ergast_fetcher.py", project_root)
        elif choice == "5":
            print("\n⏳ Running Dataset Builder...")
            builder_path = os.path.join(project_root, "data_pipeline/dataset_builder.py")
            if os.path.exists(builder_path):
                builder_success = run_script("data_pipeline/dataset_builder.py", project_root)
                if builder_success:
                    retrain_race_choice = input("\nDataset built successfully. Retrain the Race Predictor model now? (y/n): ").strip().lower()
                    if retrain_race_choice == "y":
                        print("\n⏳ Running Race Predictor Model Trainer...")
                        race_trainer_path = os.path.join(project_root, "model_training/race_predictor_trainer.py")
                        if os.path.exists(race_trainer_path):
                            run_script("model_training/race_predictor_trainer.py", project_root)
                        else:
                            print(f"\n❌ Error: Race Predictor trainer script not found at {race_trainer_path}. Cannot retrain.")
                            logger.error(f"Race Predictor trainer script not found at {race_trainer_path}. Cannot retrain.")
                    else:
                        print("Skipping Race Predictor model retraining.")

                    retrain_sim_choice = input("\nRetrain the Simulation (Qualifying) model now? (y/n): ").strip().lower()
                    if retrain_sim_choice == "y":
                        print("\n⏳ Running Simulation Model Trainer...")
                        sim_trainer_path = os.path.join(project_root, "model_training/trainer.py")
                        if os.path.exists(sim_trainer_path):
                            run_script("model_training/trainer.py", project_root)
                        else:
                            print(f"\n❌ Error: Simulation trainer script not found at {sim_trainer_path}. Cannot retrain.")
                            logger.error(f"Simulation trainer script not found at {sim_trainer_path}. Cannot retrain.")
                    else:
                        print("Skipping Simulation model retraining.")
            else:
                print(f"\n⚠️ Warning: Dataset builder script not found at {builder_path}. Skipping.")
                logger.warning(f"Dataset builder script not found at {builder_path}. Skipping execution.")
        elif choice == "6":
            print("Returning to main menu...")
            break
        else:
            print("Invalid selection. Please try again.")

def model_training_menu(project_root):
    """Displays the model training sub-menu and handles user choices."""
    logger.info("User entered Model Training Menu")
    while True:
        print("\n--- 🏋️ Model Training --- ")
        print("   NOTE: Training uses data processed by the Dataset Builder.")
        print("   Ensure you have run \"Process Raw Data\" if you fetched new raw data.")
        print("-" * 20)
        print("1. Train Simulation Model (for Past Qualifying)")
        print("2. Train Race Predictor Model (for Upcoming Race)")
        print("3. Back to Main Menu")
        
        choice = input("Select an option (1-3): ").strip()

        script_to_run = None
        if choice == "1":
            print("\n⏳ Running Simulation Model Trainer...")
            script_to_run = "model_training/trainer.py"
        elif choice == "2":
            print("\n⏳ Running Race Predictor Model Trainer...")
            script_to_run = "model_training/race_predictor_trainer.py"
        elif choice == "3":
            print("Returning to main menu...")
            break
        else:
            print("Invalid selection. Please try again.")
            continue

        trainer_path = os.path.join(project_root, script_to_run)
        if not os.path.exists(trainer_path):
            print(f"\n❌ Error: Model training script not found at {trainer_path}. Cannot proceed.")
            logger.error(f"Model training script not found at {trainer_path}. Cannot proceed.")
            continue

        success = run_script(script_to_run, project_root)
        if success:
            print(f"\n✅ Model training script ({script_to_run}) finished.")
        else:
            print(f"\n❌ Model training script ({script_to_run}) failed. Check logs or output above for details.")
