from menu.menu import display_menu
from utils.logger import setup_logger

logger = setup_logger("Main")

def main():
    logger.info("Starting F1 ML Project")
    try:
        display_menu()
    except Exception as e:
        logger.exception(f"Unexpected error in main: {e}")

if __name__ == "__main__":
    main()
