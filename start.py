from src.models.train_model import train_model

data_path = r"data\processed\processed_df.csv"
if __name__ == "__main__":
    train_model(data_path)
