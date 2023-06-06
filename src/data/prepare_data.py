import os
from tqdm import tqdm
import pandas as pd
import click
from sklearn.preprocessing import LabelEncoder

from config import ConfigData


def _parse_data(data_path: str):
    flag = False
    for filename in tqdm(os.listdir(data_path)):
        if filename.endswith('.csv'):
            if not flag:
                full_df = pd.read_csv(os.path.join(data_path, filename))
                full_df['label'] = filename.replace('.csv', '')
                flag = True
            else:
                curr_df = pd.read_csv(os.path.join(data_path, filename))
                curr_df['label'] = filename.replace('.csv', '')
                full_df = pd.concat([full_df, curr_df])
    return full_df


def _clean_data(full_df: pd.DataFrame):
    clean_df = full_df.loc[full_df.summary.notna(), :]
    # Leave repeat of texts only if they have different targets
    clean_df.drop_duplicates(['summary', 'label'], inplace = True)
    le = LabelEncoder()
    clean_df.loc[:, 'text_id'] = le.fit_transform(clean_df.summary) # Unique id for unique text
    clean_df = pd.get_dummies(clean_df, columns = ['label'], prefix='encodded_label') # label to OneHot vector
    labels = [col for col in clean_df.columns if 'encodded_label' in col]
    non_labels = [col for col in clean_df.columns if 'encodded_label' not in col]
    target_df = clean_df.groupby('text_id')[labels].sum().reset_index()
    clean_df = clean_df.loc[:, non_labels].merge(target_df, on='text_id')
    clean_df.drop_duplicates('text_id', inplace = True)
    return clean_df
    # clean_df = pd.concat([clean_df, target_df], axis = 1)
    # target_df = target_df.groupby('text_id').sum()
    # duplicated_df = clean_df.loc[mask, :]
    # duplicate_target = duplicated_df.groupby('text_id')[labels].sum()
    # duplicated_df.drop_duplicates('text_id', inplace = True)
    # duplicated_df = duplicated_df.drop(labels, axis =1)
    # clean_df.loc[mask, labels] = clean_df.loc[mask, 'text_id'].map(duplicate_target)
    # clean_df.drop_duplicates('text_id', inplace = True)
    # clean_df = clean_df.loc[:, ['summary'] + labels]
    # return clean_df



@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("save_path", type=click.Path())
def prepare_data(data_path: str, save_path: str):
    full_df = _parse_data(data_path)
    clean_df = _clean_data(full_df)
    clean_df.to_csv(save_path, index= False)

if __name__ == "__main__":
    prepare_data()
