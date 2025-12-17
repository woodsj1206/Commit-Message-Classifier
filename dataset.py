# Original Author: Sebastian Raschka (https://github.com/rasbt/LLMs-from-scratch)
# Modified By: woodsj1206 (https://github.com/woodsj1206)
# Last Modified: 12/16/2025
import pandas
import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, csv_file_path, tokenizer, csv_text="Text", csv_label="Label", max_length=None, pad_token_id=50256):
        self.csv_text = csv_text
        self.csv_label = csv_label

        self.data = pandas.read_csv(csv_file_path)

        self.encoded_texts = [tokenizer.encode(
            text) for text in self.data[self.csv_text]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:self.max_length]
                                  for encoded_text in self.encoded_texts]

        self.encoded_texts = [encoded_text + [pad_token_id] * (
            self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index][self.csv_label]
        return [torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)]

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)
