import pandas as pd
from sklearn.utils import shuffle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str, random_state: int = 42) -> pd.DataFrame:

    logging.info(f"Loading data from {file_path}")
    try:

        table = pd.read_csv(file_path, header=0) 
        
        logging.info("Filling NaN values with 0")
        table = table.fillna(0)

        logging.info("Shuffling data")
        table = shuffle(table, random_state=random_state)
        table.reset_index(inplace=True, drop=True)
        
        logging.info(f"Data loaded successfully. Shape: {table.shape}")
        return table
    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


if __name__ == '__main__':
    data_path = 'data/39_FullDataset_normal_faullt_attack.csv' 
    df = load_data(data_path)
    print(df.head())
    print(df.info()) 