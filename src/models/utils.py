import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast as BertTokenizer
import pandas as pd


class MultiLabelDataset(Dataset):
  def __init__(
    self,
    data: pd.DataFrame,
    tokenizer: BertTokenizer,
    max_token_len: int = 512
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.size_data = data.shape
    self.labels_name = [col for col in data.columns if 'encodded_label' in col]
    # self.labels = data.loc[:, self.labels_name]
    self.max_token_len = max_token_len

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]
    text = data_row.summary
    labels = data_row[self.labels_name]
    # print(labels)
    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return dict(
      # comment_text=text,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.ShortTensor(labels)
    )