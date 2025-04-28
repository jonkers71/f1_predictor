from utils.logger import setup_logger
from data_pipeline import session_fetcher, lap_collector, weather_fetcher, team_metrics, dataset_builder
from model_training import trainer

logger = setup_logger("ModelUpdater")

def update_and_retrain():
    logger.info("Starting update and retrain process.")

    try:
        logger.info("Fetching current sessions...")
        session_data = session_fetcher.fetch_sessions()

        logger.info("Collecting lap data...")
        lap_data = lap_collector.get_lap_times(session_data)

        logger.info("Fetching weather data...")
        weather_data = weather_fetcher.get_weather(session_data)

        logger.info("Calculating team metrics...")
        team_data = team_metrics.get_team_strength(session_data)

        logger.info("Building dataset...")
        dataset = dataset_builder.build_dataset(lap_data, weather_data, team_data)

        logger.info("Training model...")
        trainer.train_model(dataset)

        logger.info("✅ Update and retraining complete.")
        return True

    except Exception as e:
        logger.exception("❌ Update and retrain failed.")
        return False
