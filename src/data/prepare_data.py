import os

import click
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def parse_data(data_path: str) -> pd.DataFrame:
    """Функция объединяет все csv файлы в заданной папке в один файл

    Args:
        data_path (str): папка с csv файлами

    Returns:
        pd.DataFrame: _description_
    """ """"""
    flag = False
    for filename in tqdm(os.listdir(data_path)):
        if filename.endswith(".csv"):
            if not flag:
                full_df = pd.read_csv(os.path.join(data_path, filename))
                full_df["label"] = filename.replace(".csv", "")
                flag = True
            else:
                curr_df = pd.read_csv(os.path.join(data_path, filename))
                curr_df["label"] = filename.replace(".csv", "")
                full_df = pd.concat([full_df, curr_df])
    return full_df


def clean_data(full_df: pd.DataFrame) -> pd.DataFrame:
    """Очищаем DataFrame от дубликатов в толбце summary,
    приводим целевые метки к OneHotEncoding вектору

    Args:
        full_df (pd.DataFrame): данные для обработки

    Returns:
        pd.DataFrame: очищенные данные
    """ """"""
    clean_df = full_df.loc[full_df.summary.notna(), :]
    # Оставляем дублткаты текста, только если он принадлежит разным лейблам
    clean_df.drop_duplicates(["summary", "label"], inplace=True)
    le = LabelEncoder()
    clean_df.loc[:, "text_id"] = le.fit_transform(
        clean_df.summary
    )  # Unique id for unique text
    clean_df = pd.get_dummies(clean_df, columns=["label"], prefix="encodded_label")
    labels = [col for col in clean_df.columns if "encodded_label" in col]
    non_labels = [col for col in clean_df.columns if "encodded_label" not in col]
    target_df = clean_df.groupby("text_id")[labels].sum().reset_index()
    clean_df = clean_df.loc[:, non_labels].merge(target_df, on="text_id")
    clean_df.drop_duplicates("text_id", inplace=True)
    clean_df["num_labels"] = clean_df[labels].sum(1)
    return clean_df


def sample_sparse_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downsampling данных из-за дисбаланса классов. Тексты принадлежащие нескольким
    классам оставляем неизменными, тексты которые относятся только к одному классу
    сэмплируем до достижения доли в 5% в общем балансе классов

    Args:
        df (pd.DataFrame): не обработанные данные

    Returns:
        pd.DataFrame: обработанные данные
    """ """"""
    labels = [col for col in df.columns if "encodded_label" in col]
    min_value = df[labels].sum().min()  # кол-во самого редкого целевого класса
    total = min_value // 0.02  # кол-во данных которые нужно сэмлировать
    # чтобы распределение всех классов было > 2%
    # Для каждого класса как часто он принадлежит еще другим классам
    non_unique_num_labels = df.loc[df.num_labels != 1, :][labels].sum().to_dict()
    # Для каждого класса как часто он только одному классу
    unique_num_labels = df.loc[df.num_labels == 1, :][labels].sum().to_dict()

    non_unique_df = df.loc[df.num_labels > 1, :]
    unique_df = df.loc[df.num_labels == 1, :]
    for label in labels:
        if non_unique_num_labels[label] > total * 0.05:
            continue
        else:
            curr_df = unique_df.loc[unique_df[label] != 0, :]
            n_sample = min(
                int(total * 0.05 - non_unique_num_labels[label]),
                unique_num_labels[label],
            )

            non_unique_df = pd.concat([non_unique_df, curr_df.sample(n_sample)])
    return non_unique_df


@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("save_path", type=click.Path())
def prepare_data(data_path: str, save_path: str):
    full_df = parse_data(data_path)
    clean_df = clean_data(full_df)
    clean_df = sample_sparse_row(clean_df)
    clean_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    prepare_data()
