from src.train_pipeline import run_training_pipeline
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Executing main script...")
    run_training_pipeline()
    logging.info("Main script execution finished.") 