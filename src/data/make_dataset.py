import re

import click
import pandas as pd


def cut_texts(text: str, num_words=512) -> str:
    """Обрезаем текст до нужного кол-ва символов для экономии памяти

    Args:
        text (str): исходный текст
        num_words (int, optional): требуемое кол-во слов. Defaults to 512.

    Returns:
        str: обработанный текст
    """ """"""
    lowertext = text.lower()
    split = re.split("[\\s.,!?:;'\"]+", lowertext)
    split = [x for x in split if x != ""]
    return " ".join(split[:num_words])


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def make_dataset(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    interim_df = pd.read_csv(input_filepath)
    interim_df["summary"] = interim_df.summary.map(cut_texts)
    labels_name = [col for col in interim_df.columns if "encodded_label" in col]
    interim_df[["summary"] + labels_name].to_csv(output_filepath, index=False)


if __name__ == "__main__":
    make_dataset()
