import os
from tqdm import tqdm
import pandas as pd
import click

@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("save_path", type=click.Path())
def parse_data(data_path: str, save_path: str):
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
    full_df.to_csv(save_path)




if __name__ == "__main__":
    parse_data()
