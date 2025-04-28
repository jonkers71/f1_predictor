import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger("SimulatorUtils")


def build_simulation_dataset(selected_country: str) -> pd.DataFrame:
    """
    Prepares a simulation dataset using all data up to (and including) P3 of the selected country's most recent race.
    It excludes the qualifying session of that race.
    """

    try:
        # Load raw session data
        session_df = pd.read_csv("data/raw/session_raw.csv", parse_dates=["date"])
        lap_df = pd.read_csv("data/raw/laps_raw.csv")
        weather_df = pd.read_csv("data/raw/weather_raw.csv")

        # 1. Get the selected race's sessions
        race_sessions = session_df[session_df['country'] == selected_country]

        if race_sessions.empty:
            logger.error(f"No sessions found for {selected_country}")
            return pd.DataFrame()

        # 2. Get the Qualifying session datetime for that race
        quali_sessions = race_sessions[session_df["session_type"] == "Qualifying"]
        if quali_sessions.empty:
            logger.error(f"No qualifying session found for {selected_country}")
            return pd.DataFrame()

        quali_date = quali_sessions.iloc[0]["date"]

        logger.info(f"Using all sessions before {quali_date} for {selected_country}")

        # 3. Filter all sessions that happened before the qualifying session (any country)
        allowed_sessions = session_df[session_df["date"] < quali_date]

        # 4. Get session keys of those sessions
        valid_session_keys = allowed_sessions["session_key"].unique()

        # 5. Filter lap data for those sessions
        lap_data = lap_df[lap_df["session_key"].isin(valid_session_keys)]

        if lap_data.empty:
            logger.warning("No lap data found for valid sessions.")
            return pd.DataFrame()

        # 6. Merge with weather and session metadata
        weather_df["session_key"] = weather_df["session_key"].astype(str)
        lap_data["session_key"] = lap_data["session_key"].astype(str)

        merged_df = lap_data.merge(weather_df, on="session_key", how="left")
        merged_df = merged_df.merge(session_df[["session_key", "circuit_short_name", "session_type", "date"]],
                                    on="session_key", how="left")

        # 7. Optional: drop any incomplete rows or fill NAs
        merged_df.dropna(subset=["lap_time"], inplace=True)

        logger.info(f"Simulation dataset built: {merged_df.shape[0]} laps from {len(valid_session_keys)} sessions")

        return merged_df

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return pd.DataFrame()

    except Exception as e:
        logger.exception(f"Error building simulation dataset: {e}")
        return pd.DataFrame()
