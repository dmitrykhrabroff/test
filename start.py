import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.bert_models import TransformersTextClassifier
from src.models.pl_datamodule import MyDataModule

data_path = r"data\processed\processed_df.csv"
model_name = "smallbenchnlp/bert-small"
dm = MyDataModule(data_path, model_name, 8)

dm.setup("fit")
steps_per_epoch = len(dm.train_dataloader())
total_training_steps = steps_per_epoch * 5
warmup_steps = total_training_steps // 5
model = TransformersTextClassifier(
    model_name, n_training_steps=total_training_steps, n_warmup_steps=warmup_steps
)

logger = TensorBoardLogger("lightning_logs", name="multilabel_classification")
trainer = pl.Trainer(
    logger=logger,
    accelerator="cuda",
    max_epochs=5,
    enable_checkpointing=True,
    fast_dev_run=False,
)
torch.cuda.empty_cache()
if __name__ == "__main__":
    trainer.fit(model, dm)
    # for i, batch in enumerate(dm.train_dataloader()):
    #     batch = {k:v.cuda() for k,v in batch.items()}
    #     # input_ids = batch["input_ids"]
    #     # attention_mask = batch["attention_mask"]
    #     # labels = batch["labels"]
    #     model.cuda()
    #     model.training_step(batch, i)
    #     if (i+1) % 10 == 0:
    #         model.on_train_epoch_end()
    #     # loss, outputs = model(input_ids, attention_mask, labels)
