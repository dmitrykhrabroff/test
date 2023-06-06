import torch
import torch.nn as nn
import pandas as pd


class ConfigModel:
    def __init__(self) -> None:
        self.num_workers = 2
        self.test_size = 0.15
        self.epochs = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = nn.BCELoss()
        self.n_classes = 20
        self.model_name = 'bert-base-cased'
        with open('data/processed/processed_df.csv') as f:
           self.label_names = f.readline().strip().split(',')[2:]
        self.grid = {
            "learning_rate": [1e-4, 5e-4, 1e-5],
            "batch_size": [2, 4],
            "warmup_partition" : [5, 10]
        }
