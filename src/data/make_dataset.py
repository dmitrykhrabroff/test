# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import re

def cut_texts(text, num_words=512):
    lowerText = text.lower()
    split = re.split("[\s.,!?:;'\"]+",lowerText)
    split = [x for x in split if x != '']  # <- list comprehension
    return ' '.join(split[:num_words])



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    interim_df = pd.read_csv(input_filepath)
    interim_df['summary'] = interim_df.summary.apply(lambda x: cut_texts(x))
    labels_name = [col for col in interim_df.columns if 'encodded_label' in col]
    interim_df[['summary'] + 
               labels_name].to_csv(output_filepath, index = False)
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
