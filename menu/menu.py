import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import pandas as pd
from model_training.simulate_trainer import simulate_past_race

logger = setup_logger("Menu")

def display_menu():
    while True:
        print("\n📊 F1 Prediction System Menu")
        print("1. Predict Upcoming Qualifying")
        print("2. Predict Upcoming Race")
        print("3. Simulate Past Race")
        print("4. Update Training Data & Retrain Model")
        print("5. Exit")

        choice = input("Select an option (1–5): ").strip()

        if choice == "1":
            predict_qualifying()
        elif choice == "2":
            predict_race()
        elif choice == "3":
            simulate_past()
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

def simulate_past():
    logger.info("User selected: Simulate Past Race")
    
    try:
        session_data = pd.read_csv('data/processed/cleaned_f1_data.csv')  # Historical data up to P3
    except FileNotFoundError:
        print("❌ Could not find cleaned session data. Make sure 'cleaned_f1_data.csv' exists in 'data/processed'.")
        return

    # Debugging: print columns and first few rows of the dataset
    print("Columns in session_data:", session_data.columns)
    print("First few rows in session_data:")
    print(session_data.head())

    # Check unique country names before dropping NaNs
    print("\nUnique country names in the dataset:")
    print(session_data['country_name'].unique())
    
    # Remove rows where 'country_name' is NaN
    session_data = session_data.dropna(subset=['country_name'])
    
    # List all available countries from the 'country_name' column, sorted alphabetically
    available_countries = sorted(session_data['country_name'].unique())

    # Check if there are any available countries
    if not available_countries:
        print("No valid countries found in the dataset. Please check the data.")
        return
    
    print("\nAvailable countries for simulation:")
    for i, country in enumerate(available_countries, start=1):
        print(f"{i}. {country}")
    
    # Ask user to select a country to simulate
    try:
        country_choice = int(input(f"\nEnter the number of the country you want to simulate (1–{len(available_countries)}): "))
        if 1 <= country_choice <= len(available_countries):
            selected_country = available_countries[country_choice - 1]
            print(f"\nSimulating qualifying for country: {selected_country}")
            
            # Filter session data by selected country
            country_sessions = session_data[session_data['country_name'] == selected_country]
            
            # Call the simulate function with the filtered data
            simulated_order = simulate_past_race(selected_country, country_sessions)
            
            if simulated_order is not None:
                print(f"\nSimulated qualifying order for {selected_country}:")
                print(simulated_order)
            else:
                print(f"No qualifying data found for country: {selected_country}.")
        else:
            print("Invalid selection. Please try again.")
    
    except ValueError:
        print(f"Invalid input. Please enter a number between 1 and {len(available_countries)}.")

def update_and_train():
    from model_training import model_updater
    success = model_updater.update_and_retrain()
    if success:
        print("✅ Model successfully updated and retrained.")
    else:
        print("❌ Something went wrong during update.")
