# File: main.py
import os
import sys
os.system('clear')

# Determine the project root directory (the directory containing main.py)
project_root = os.path.dirname(os.path.abspath(__file__))
# Add project root to sys.path to allow imports from subdirectories
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from menu.main_menu import display_main_menu # Import from the refactored main_menu module
from utils.logger import setup_logger

logger = setup_logger("Main")

def main():
    logger.info(f"Starting F1 ML Project from root: {project_root}")
    try:
        # Pass the project_root to the main menu function
        display_main_menu(project_root)
    except Exception as e:
        logger.exception(f"Unexpected error in main: {e}")

if __name__ == "__main__":
    main()

