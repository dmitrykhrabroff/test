import pandas as pd
import pytorch_lightning as pl
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from transformers import AutoTokenizer

from src.models.config import ConfigModel
from src.models.utils import MultiLabelDataset


class MyDataModule(pl.LightningDataModule):
    def __init__(self, processed_df_path, model_name, batch_size):
        """_summary_

        Args:
            processed_df_path (_type_): путь к DataFrame готовому к обучению
            model_name (_type_): имя модели
            batch_size (_type_): размер батча
        """ """"""
        super().__init__()
        self.data_path = processed_df_path
        processed_df = pd.read_csv(processed_df_path)
        print(processed_df.shape, 'processed_df.shape')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dset = MultiLabelDataset(processed_df, tokenizer)
        self.config = ConfigModel()
        self.labels_name = self.dset.labels_name
        self.batch_size = batch_size
        self.prepare_data()

    def prepare_data(self):
        df = pd.read_csv(self.data_path)
        labels = df[
            self.labels_name
        ].values  # загружаем целевой признак для стратификации выборки
        indices = list(range(labels.shape[0]))
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=self.config.test_size, random_state=139
        )

        ind_train, self.ind_test = next(splitter.split(X=indices, y=labels))
        self.ind_train, self.ind_val = next(
            splitter.split(X=ind_train, y=labels[ind_train])
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = Subset(self.dset, self.ind_train)
            self.val_set = Subset(self.dset, self.ind_val)
        elif stage == "test":
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
