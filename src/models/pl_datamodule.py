import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from transformers import AutoTokenizer
from skmultilearn.model_selection import iterative_train_test_split
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.models.config import ConfigModel
from src.models.utils import MultiLabelDataset


class MyDataModule(pl.LightningDataModule):
    def __init__(self, input_filepath, model_name, batch_size):
        super().__init__()
        self.data_path = input_filepath
        processed_df = pd.read_csv(input_filepath)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dset = MultiLabelDataset(processed_df, tokenizer)
        self.config = ConfigModel()
        # self.size_data = self.dset.size_data[0]
        self.labels_name = self.dset.labels_name
        self.batch_size = batch_size
        self.prepare_data()

    def prepare_data(self):
        df = pd.read_csv(self.data_path)
        labels = df[self.labels_name].values  # загружаем целевой признак для стратификации выборки
        indices = list(range(labels.shape[0]))
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1,
                                                    test_size=self.config.test_size,
                                                    random_state=139)

        ind_train,self.ind_test = next(splitter.split(X=indices, y=labels))
        self.ind_train,self.ind_val = next(splitter.split(X=ind_train, y=labels[ind_train]))


    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = Subset(self.dset, self.ind_train)
            self.val_set = Subset(self.dset, self.ind_val)
            self.test_set = Subset(self.dset, self.ind_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
