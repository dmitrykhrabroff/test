import torch
import torch.nn as nn


class ConfigModel:
    def __init__(self) -> None:
        self.num_workers = 0
        self.test_size = 0.15
        self.epochs = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = nn.BCELoss()
        with open("data/processed/processed_df.csv") as f:
            self.label_names = f.readline().strip().split(",")[2:]
        self.n_classes = 20
        self.model_name = "bert-base-cased"
        self.grid = {
            "learning_rate": [1e-5, 1e-6],
            "batch_size": [8],
            "warmup_partition": [5, 10]
        }
