# Тестовое задание.
## Создание мультиклассового классификатора (один текст может иметь разные метки)
==============================

 ### Описание задачи
* Имеется 20 csv файлов, каждый описывает некоторую технологию (название файла – название технологии). Технология описывается новостями за последние 3 года.
* Надо создать классификатор, который будет предсказывать по входному тексту, к какой технологии относится новость.
* Классификатор должен быть мультиклассовым – т.е. один текст может иметь метки разных технологий.
* Необходимые метрики соискатель должен выбрать сам, обосновать их выбор и предоставить значения метрик для обученного классификатора.
* Выбор фреймворка и модели остается за соискателем.
* Будет плюсом если соискатель проверит разные модели, приведет значения метрик и обоснует выбор одной из моделей для потенциального запуска в продакшн.
____________________
## 1. Настройка среды окружения и установка зависимостей.
 В качестве менеджера зависимостей в проекте используется пакет [poetry](https://python-poetry.org/).
 Для установки зависимостей и активации виртуального окружения введите команды
 ```
 pip install poetry
 poetry install
 poetry shell
 ```
## 2. Управление данными.
 Исходные данные размещены на S3 хранилище в Яндекс облаке.  В качестве инструмента по управлению данными используется dvc (для работы dvc должен быть предварительно активирован git). 
 Для доступа к s3 хранилищу необходимо заполнить [config_example](https://github.com/dmitrykhrabroff/test/blob/main/.dvc/config_example) (криды предоставляются по запросу) и выполнить следующие команды.
 ```
 git init
 dvc init
 dvc pull
 ```
 ## 3. Предобработка данных.
 Реализована возможность очистки данных с помощью инструментов командной строки.
 Функция prepare_data удаляет дубликаты в наших данных и приводит целевые признаки к OneHotEncoder вектору,
 а так же выполняет DownSampling данных
  ```
  python src/data/prepare_data.py data/raw data/interim/interim_df.csv
 ```
 Функция make_dataset выполняет финальную фильтрацию данных, обрезает текст до заданного кол-ва токенов в зависимости от модели.  
  ```
 python src/data/make_dataset.py data/interim/interim_df.csv data/processed/processed_df.csv
 ```
 Более подробно процедура обработки данныз показана в [ноутбуке](https://github.com/dmitrykhrabroff/test/blob/main/notebooks/EDA.ipynb)
 
## 4. Обучение модели.
 Архитектура проекта предусматривает возможность дообучения различных моделей из библиотеки [transformers](https://huggingface.co/docs/transformers/index).
 Для инициализации моедлей используйте
```
from transformers import AutoModel # For BERTs
from transformers import AutoModeForSequenceClassification # For models fine-tuned on MNLI
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModel.from_pretrained(model_name) 
```
Обучение модели доступно из командной строки  
  ```
python src/models/train_model.py data/processed/processed_df.csv
 ```
 
## 5. Оценка результатов модели
В ходе исследования были протестированы следующие модели:
* distil-bert-uncased
* "smallbenchnlp/bert-small"
* "prajjwal1/bert-mini"
* "prajjwal1/bert-small"

Результаты оценки моделей представлены в [ноутбуке](https://github.com/dmitrykhrabroff/test/blob/main/notebooks/evaluate_models.ipynb)

## 6. Структура проекта
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
