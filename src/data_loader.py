import pandas as pd
import logging

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_data(data_path="data/train.csv"):
    try:
        logging.info("Loading dataset...")
        df = pd.read_csv(data_path)
        logging.info(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise e