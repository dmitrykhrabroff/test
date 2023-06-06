import itertools
import os
import warnings

import click
import pytorch_lightning as pl
from dotenv import load_dotenv
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from src.models.config import ConfigModel
from src.models.models import TransformersTextClassifier
from src.models.pl_datamodule import MyDataModule

# ранние остановки
# сохранение весов

# ранние остановки
# сохранение весов

warnings.filterwarnings("ignore")
load_dotenv()
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")



def train_model(dataset_path: str):
    """Тренировка нашей модели с перебором гиперпараметров

    Args:
        dataset_path (str): путь к датасету
        logger_path (str): путь к папке куда записываем лучшие модели
        is_demo (bool, optional): при значении True выполняется пробный
        прогон модели для одного батча одной эпохи, для отлова багов.
    """

    config = ConfigModel()
    epochs = config.epochs
    grid = config.grid  # гиперпараметры для обучения модели
    device = config.device
    model_name = config.model_name
    for values in itertools.product(*grid.values()):  # перебор гиперпараметров
        lr_rate, batch_size, warmup_partition = values
        
        name_experiment = (
            f"{model_name}_{lr_rate}_{batch_size}"
        )
        dm = MyDataModule(dataset_path, model_name, batch_size)  # создаем DataModule-экземпляр
        dm.setup('fit')
        steps_per_epoch = len(dm.train_dataloader()) 
        total_training_steps = steps_per_epoch * config.epochs
        warmup_steps = total_training_steps // warmup_partition
        
        model = TransformersTextClassifier(model_name = model_name, lr_rate = lr_rate,
                                           n_training_steps = total_training_steps,
                                           n_warmup_steps =  warmup_steps)

        # список для отслеживания lr, ранних остановок, сохранения весов
        callbacks = [
            ModelCheckpoint(
                dirpath=f"models/{name_experiment}",
                filename=model_name,
                save_top_k=2,
                monitor="valid_loss",
                mode="min",
            ),
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(
                monitor="valid_loss",
                min_delta=2e-4,
                patience=10,
                verbose=False,
                mode="min",
            ),
        ]
        logger = TensorBoardLogger("lightning_logs", name="multilabel_classification")


        trainer = pl.Trainer(
            accelerator=device,
            max_epochs=epochs,
            callbacks=callbacks,
            logger = logger,
            enable_checkpointing=True,
            fast_dev_run=True,
)

        trainer.fit(model, dm)

@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def main(dataset_path):
    train_model(dataset_path)

if __name__ == "__main__":
    main()
