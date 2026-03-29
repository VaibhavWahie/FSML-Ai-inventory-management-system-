from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.features import create_features
from src.train import train_model

def run_pipeline():
    df = load_data()
    df = preprocess_data(df)
    df = create_features(df)
    train_model(df)

if __name__ == "__main__":
    run_pipeline()