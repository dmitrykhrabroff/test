import os
from tqdm import tqdm
import pandas as pd
import click
from config import ConfigData

@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("save_path", type=click.Path())
def clean_data(data_path: str, save_path: str):
    df = pd.read_csv(data_path)
    config = ConfigData()
    clean_df = df.loc[df.summary.notna(), :]
    clean_df.drop_duplicates(clean_df[['summary', 'label']], inplace = True) #EDA показал, что много дубликатов текста 
    labels = clean_df.label.unique()
    remove_index = []
    curr = pd.get_dummies(clean_df.label)
    labels = curr.columns
    clean_df = pd.concat([clean_df, pd.get_dummies(clean_df.label)], axis = 1)
    ununique_text = clean_df.summary.value_counts()[clean_df.summary.value_counts() > 1].index
    for text in tqdm(ununique_text):
        curr_ind = clean_df.loc[clean_df.summary == text, :].index
        curr_labels = clean_df.loc[curr_ind, labels].sum(0)
        remove_index.extend(curr_ind[1:])
        clean_df.loc[curr_ind[0], labels] = curr_labels
    clean_df.drop(remove_index, inplace = True)
    clean_df.to_csv(save_path)


if __name__ == "__main__":
    clean_data()
