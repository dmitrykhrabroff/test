import re

import click
import pandas as pd


def cut_texts(text: str, num_words: int) -> str:
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
@click.option("--num_tokens", required=False, type=click.INT)
def make_dataset(input_filepath: str,
                  output_filepath: str, num_tokens: int=512):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    interim_df = pd.read_csv(input_filepath)
    interim_df["summary"] = interim_df.summary.apply(lambda x: cut_texts(x, num_tokens))
    labels_name = [col for col in interim_df.columns if "encodded_label" in col]
    interim_df[["summary"] + labels_name].to_csv(output_filepath, index=False)


if __name__ == "__main__":
    make_dataset()
