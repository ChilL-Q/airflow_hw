import os
import pandas as pd
import dill
from typing import Tuple
import sys

# Добавляем путь к папке, где находится модуль 'modules'
sys.path.append(os.getcwd())  # Замените на актуальный путь


# Функция для загрузки модели
def load_model(model_path: str) -> Tuple:
    with open(model_path, 'rb') as file:
        model = dill.load(file)
    return model


# Функция для получения пути к последней обученной модели
def get_latest_model_path(models_dir: str) -> str:
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    latest_model_file = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
    return os.path.join(models_dir, latest_model_file)


# Функция для выполнения предсказаний
def make_predictions(model, test_data_path: str) -> pd.DataFrame:
    test_data = pd.read_csv(test_data_path)
    predictions = model.predict(test_data)
    return pd.DataFrame(predictions, columns=['predictions'])


# Функция для сохранения предсказаний
def save_predictions(predictions: pd.DataFrame, predictions_path: str):
    predictions.to_csv(predictions_path, index=False)


# Основная функция
def predict():
    # Пути к файлам и директориям
    project_path = os.environ.get('PROJECT_PATH', os.getcwd())
    model_dir = os.path.join(project_path, 'data/models')

    # Путь к файлу с тестовыми данными должен быть изменен на 'data/train/homework.csv'
    test_data_path = os.path.join(project_path, 'data/train/homework.csv')

    predictions_dir = os.path.join(project_path, 'data/predictions')
    predictions_path = os.path.join(predictions_dir, 'predictions.csv')

    # Создание директории для предсказаний, если она не существует
    os.makedirs(predictions_dir, exist_ok=True)

    # Загрузка модели
    latest_model_path = get_latest_model_path(model_dir)
    model = load_model(latest_model_path)

    # Предсказание
    predictions_df = make_predictions(model, test_data_path)

    # Сохранение предсказаний
    save_predictions(predictions_df, predictions_path)

    print(f"Predictions are saved to {predictions_path}")


if __name__ == '__main__':
    predict()
