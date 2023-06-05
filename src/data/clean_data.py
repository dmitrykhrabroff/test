from sklearn.preprocessing import LabelEncoder
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
    clean_df.drop_duplicates(['summary', 'label'], inplace = True)
    le = LabelEncoder()
    clean_df['text_id'] = le.fit_transform(clean_df.summary)
    mask = clean_df.duplicated('text_id', keep = False)
    curr_df = pd.get_dummies(clean_df.label)
    labels = curr_df.columns
    clean_df = pd.concat([clean_df, curr_df], axis = 1)
    duplicated_df = clean_df.loc[mask, :]
    non_duplicated_df = clean_df.loc[~mask, :]
    duplicate_target = duplicated_df.groupby('text_id')[labels].sum()
    duplicated_df.loc[:, labels] = duplicate_target
    duplicated_df.drop_duplicates('text_id', inplace = True)
    clean_df = pd.concat([non_duplicated_df, duplicated_df])
    config.labels = labels
    clean_df.to_csv(save_path)


if __name__ == "__main__":
    clean_data()
