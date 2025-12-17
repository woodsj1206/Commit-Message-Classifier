# Original Author: Sebastian Raschka (https://github.com/rasbt/LLMs-from-scratch)
# Modified By: woodsj1206 (https://github.com/woodsj1206)
# Last Modified: 12/16/2025
import requests
import shutil
import os


def create_directory(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=False)
        return dir_path
    except FileExistsError:
        print(
            f"Error: Output directory '{dir_path}' already exists. Please change the name of the csv file or remove the existing directory.")
    except OSError as e:
        print(
            f"Error: An error occurred when creating output directory. {e}")
    return None


def download_gpt2_model(file_name, dir_path):
    try:
        response = requests.get(
            f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}", timeout=60, stream=True)
        response.raise_for_status()

        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
    except Exception as e:
        print(f"Error: {e}")


def split_data(data, train_frac, validation_frac, seed=None):
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    if train_frac + validation_frac >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1")

    train_index = int(len(data) * train_frac)
    validation_index = train_index + int(len(data) * validation_frac)

    train_data = data[:train_index]
    validation_data = data[train_index:validation_index]
    test_data = data[validation_index:]

    return train_data, validation_data, test_data
