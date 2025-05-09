# F1 Predictor Project

## Overview

This project aims to predict Formula 1 race outcomes using historical data and machine learning models. It includes scripts for fetching data from various sources (OpenF1 API, Ergast API), processing the data, training predictive models, and running predictions.

## Features

*   **Data Fetching:** Scripts to fetch session data, lap times, and weather information from the OpenF1 API. It also includes functionality to fetch historical race results from the Ergast API.
    *   `session_fetcher.py`: Fetches basic session information (meetings, sessions).
    *   `lap_collector.py`: Fetches detailed lap-by-lap data for specified sessions. Only fetches data for sessions missing from the existing `lap_data.csv`.
    *   `weather_fetcher.py`: Fetches weather data for specified sessions. Only fetches data for sessions missing from the existing `weather_raw.csv`.
    *   `ergast_fetcher.py`: Fetches historical race results from the Ergast database.
*   **Data Processing:**
    *   `dataset_builder.py`: Merges data from various sources (laps, sessions, weather, Ergast results), performs cleaning, feature engineering (e.g., rolling lap averages), and saves the final processed dataset (`cleaned_f1_data.csv`) and a summary report (`summary.txt`).
*   **Model Training:**
    *   `trainer.py`: (Placeholder/Not fully implemented for qualifying simulation in this version)
    *   `race_predictor_trainer.py`: Trains an XGBoost model to predict race finishing positions based on the processed data. Saves the trained model.
*   **Prediction:**
    *   `race_predictor.py`: Loads the trained race predictor model and necessary data to predict the outcome of an upcoming race (currently requires manual setup of prediction data).
*   **Interactive Menu:**
    *   `main.py`: The main entry point, providing a menu-driven interface to access different functionalities.
    *   **Main Menu Options:**
        1.  Predict Upcoming Qualifying: *Currently Not Implemented*
        2.  Predict Upcoming Race: Implemented - Uses `race_predictor.py`.
        3.  Run Past Qualifying Simulation: *Currently Not Implemented*
        4.  Data Management & Processing: Sub-menu for fetching and processing data.
        5.  Model Training: Sub-menu for training models.
        6.  Exit
    *   **Data Management Sub-Menu Options:**
        1.  Fetch Session Data
        2.  Fetch Lap Data (Incremental)
        3.  Fetch Weather Data (Incremental)
        4.  Process Raw Data (Build Dataset & Optionally Retrain Race Predictor)
        5.  Back to Main Menu
    *   **Model Training Sub-Menu Options:**
        1.  Train Simulation Model: *Currently Not Implemented*
        2.  Train Race Predictor Model
        3.  Back to Main Menu

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jonkers71/f1_predictor.git
    cd f1_predictor
    ```
2.  **Install dependencies:** Ensure you have Python 3.11 or later installed.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the main application:**
    ```bash
    python main.py
    ```
2.  **Navigate the Menus:** Use the number keys to select options.
3.  **Data Pipeline Workflow:**
    *   Use the "Data Management & Processing" menu (Option 4 from Main Menu).
    *   Fetch necessary raw data using options 1, 2, and 3. Note that lap and weather fetching are incremental and will only get missing data.
    *   Run "Process Raw Data" (Option 4 in Data Management) to build the `cleaned_f1_data.csv`. You will be prompted if you want to retrain the Race Predictor model after the dataset is built.
4.  **Model Training:**
    *   Use the "Model Training" menu (Option 5 from Main Menu).
    *   Select Option 2 to train/retrain the Race Predictor model using the latest `cleaned_f1_data.csv`.
5.  **Prediction:**
    *   Use the "Predict Upcoming Race" option (Option 2 from Main Menu).

## Data Sources

*   **OpenF1 API:** Used for fetching session, lap, and weather data (primarily for recent seasons, 2023 onwards).
*   **Ergast Developer API:** Used for fetching historical race results to supplement the main dataset, particularly for race finishing positions.

## File Structure

```
f1_predictor/
├── config/             # Configuration files (e.g., API URLs)
├── data/
│   ├── processed/      # Processed data (cleaned_f1_data.csv, summary.txt)
│   └── raw/            # Raw data fetched from APIs (lap_data.csv, session_raw.csv, etc.)
├── data_pipeline/      # Scripts for fetching and processing data
├── logs/               # Log files generated by the application
├── menu/               # Scripts defining the user interface menus
├── models/             # Saved trained machine learning models
├── model_training/     # Scripts for training models
├── prediction/         # Scripts for making predictions
├── utils/              # Utility scripts (e.g., logger setup)
├── main.py             # Main application entry point
├── requirements.txt    # Python package dependencies
└── README.md           # This file
```

## Recent Changes (May 2025)

*   Implemented Main Menu Option 2: Predict Upcoming Race.
*   Updated Data Management Option 4: Added prompt to optionally retrain the Race Predictor model after building the dataset.
*   Updated `weather_fetcher.py` to only fetch data for sessions missing from the existing raw file.
*   Fixed various bugs and improved robustness in the `dataset_builder.py` script, including handling missing columns (`date_start`, `circuit_short_name`, `meeting_key`) and resolving data merging issues.
*   Removed unnecessary `__pycache__` directories and log files from the repository.
*   Verified `requirements.txt` is up-to-date.

