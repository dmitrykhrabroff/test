import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import AUROC
from transformers import AdamW
from transformers import AutoModel
from transformers import get_linear_schedule_with_warmup

from src.models.config import ConfigModel


class TransformersTextClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name="bert-base-cased",
        lr_rate=2e-5,
        n_training_steps=None,
        n_warmup_steps=None,
    ):
        """
    Args:
        model_name (str, optional): имя модели. Defaults to 'bert-base-cased'.
        lr_rate: коэф скорости обучения. Defaults to 2e-5.
        n_training_steps: общее кол-во шагов при обучении, нужно для schedular.
        n_warmup_steps: сколько шагов прогревается наша модель, нужно для schedular.
    """ """"""
        super().__init__()
        config = ConfigModel()
        self.bert = AutoModel.from_pretrained(model_name, return_dict=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(self.bert.config.hidden_size, config.n_classes),
        )

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCEWithLogitsLoss(reduce="mean")
        self.lr_rate = lr_rate
        self.metrcis = AUROC(task="multilabel", num_labels=config.n_classes)
        self.label_names = config.label_names
        self.epoch_prediction = {"train": [], "valid": [], "test": []}
        self.epoch_labels = {"train": [], "valid": [], "test": []}
        self.automatic_optimization = False

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(output.last_hidden_state, 1)
        output = self.classifier(pooled_output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.type(torch.float))
        return loss, output

    def shared_step(self, batch, stage):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        if stage == "train":
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        self.log(f"{stage}_loss", loss, prog_bar=True, logger=True)
        self.epoch_prediction[stage].append(outputs)
        self.epoch_labels[stage].append(labels)
        return loss

    def training_step(self, batch, batch_idx):
        result = self.shared_step(batch, "train")
        return result

    def validation_step(self, batch, batch_idx):
        result = self.shared_step(batch, "valid")
        return result

    def test_step(self, batch, batch_idx):
        result = self.shared_step(batch, "test")
        return result

    def on_shared_epoch_end(self, stage):
        labels = torch.vstack(self.epoch_labels[stage]).type(torch.LongTensor).to("cpu")
        predictions = torch.vstack(self.epoch_prediction[stage]).to("cpu")
        class_roc_auc = self.metrcis(predictions, labels)
        self.logger.experiment.add_scalar(
            f"roc_auc/{stage}", class_roc_auc, self.current_epoch
        )
        self.epoch_labels[stage].clear()
        self.epoch_prediction[stage].clear()

    def on_train_epoch_end(self):
        self.on_shared_epoch_end("train")

    def on_validation_epoch_end(self):
        self.on_shared_epoch_end("valid")

    def on_test_epoch_end(self):
        self.on_shared_epoch_end("test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )
        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )
