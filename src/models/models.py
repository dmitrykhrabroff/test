import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from torchmetrics import AUROC

from src.models.config import ConfigModel

class TransformersTextClassifier(pl.LightningModule):
  def __init__(self, model_name = 'bert-base-cased', lr_rate = 2e-5, 
               n_training_steps = None, n_warmup_steps = None):
    super().__init__()
    config = ConfigModel()
    self.bert = BertModel.from_pretrained(model_name, return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, config.n_classes)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = config.loss_fn
    self.lr_rate = lr_rate
    self.metrcis = AUROC(task = 'multilabel', num_labels = config.n_classes)
    self.label_names = config.label_names
    self.epoch_prediction = {'train' : [], 'valid' : [], 'test' : []}
    self.epoch_labels = {'train' : [], 'valid' : [], 'test' : []}

  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.pooler_output)
    output = torch.sigmoid(output)
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output
  
  def shared_step(self, batch, stage):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log(f"{stage}_loss", loss, prog_bar=True, logger=True)
    self.epoch_prediction[stage].append(outputs)
    self.epoch_labels[stage].append(labels)
    return loss
  
  def training_step(self, batch, batch_idx):
    result = self.shared_step(batch, 'train')
    return result

  def validation_step(self, batch, batch_idx):
    result = self.shared_step(batch, 'valid')
    return result
  
  def test_step(self, batch, batch_idx):
    result = self.shared_step(batch, 'test')
    return result
  
  def on_shared_epoch_end(self, stage):
    labels = torch.stack(self.epoch_labels[stage]).type(torch.LongTensor)
    print(labels, 'labels')
    predictions = torch.stack(self.epoch_prediction[stage])
    for i, name in enumerate(self.label_names):
      class_roc_auc = self.metrcis(predictions[:, i], labels[:, i])
      self.logger.experiment.add_scalar(f"{name}_roc_auc/{stage}", class_roc_auc, self.current_epoch)
    self.epoch_labels[stage].clear()
    self.epoch_prediction[stage].clear()


  def on_train_epoch_end(self):
    self.on_shared_epoch_end('train')

  def on_validation_epoch_end(self):
    self.on_shared_epoch_end('valid')

  def on_train_epoch_end(self):
    self.on_shared_epoch_end('train')

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.lr_rate)
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )
    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )