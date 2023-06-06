import torch
import torch.nn as nn


class ConfigModel:
    def __init__(self) -> None:
        self.num_workers = 2
        self.test_size = 0.15
        self.epochs = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = nn.BCELoss()
        self.n_warmup_steps = None
        self.n_training_steps = None


        self.grid = {
            "learning_rate": [1e-3, 1e-2, 1e-4],
            "batch_size": [8, 16],
            "modelname" : ['bert-base-cased']
        }
