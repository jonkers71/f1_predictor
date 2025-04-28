import pandas as pd
import logging
from model_training.simulate_trainer import simulate_past_race  # Importing from the correct path
from prediction.simulator_utils import build_simulation_dataset  # Utility function for dataset construction

logger = logging.getLogger("Simulator")

def simulate_past_quali(selected_country: str):
    try:
        # Load all session data
        session_data = pd.read_csv("data/raw/session_raw.csv", parse_dates=["date"])

        # Get all sessions for the selected country
        race_sessions = session_data[session_data['country'] == selected_country]

        if race_sessions.empty:
            logger.warning(f"No sessions found for {selected_country}")
            return None

        # Get qualifying session for that country to use as a cutoff
        quali_session = race_sessions[race_sessions['session_type'] == 'Qualifying']
        if quali_session.empty:
            logger.warning(f"No qualifying session found for {selected_country}")
            return None

        quali_date = pd.to_datetime(quali_session['date'].min())

        logger.info(f"Simulating qualifying for {selected_country} using data before {quali_date}")

        # 🔧 Build dataset from all prior sessions (excluding Quali) using simulator_utils
        sim_dataset = build_simulation_dataset(selected_country)

        if sim_dataset.empty:
            logger.warning("Simulation dataset is empty after preprocessing.")
            return None

        # ✅ Run simulation (predict qualifying)
        simulated_order = simulate_past_race(selected_country, sim_dataset)

        if simulated_order is not None:
            logger.info(f"Simulated qualifying order for {selected_country}:")
            print(simulated_order)
            return simulated_order
        else:
            logger.warning(f"No simulated qualifying order for {selected_country}.")
            return None

    except FileNotFoundError as e:
        logger.error(f"Missing file: {e.filename}")
        return None
    except Exception as e:
        logger.exception("Unexpected error in simulate_past_quali")
        return None
