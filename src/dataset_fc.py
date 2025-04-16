import os
import numpy as np
import torch
from torch.utils.data import Dataset


class FCDataset(Dataset):
    def __init__(self, data_dir, df_labels, label_column, task, transform=None):
        assert task in ['classification', 'regression']

        self.data_dir = data_dir
        self.df_labels = df_labels.reset_index(drop=True)
        self.label_column = label_column
        self.task = task
        self.transform = transform

        if self.task == 'classification':
            unique_labels = sorted(self.df_labels[self.label_column].unique())
            self.label_mapping = {label: i for i, label in enumerate(unique_labels)}

        self.samples = []
        for _, row in self.df_labels.iterrows():
            subj_id = row['ID']
            label = self.label_mapping[row[self.label_column]] if self.task == 'classification' else float(
                row[self.label_column])
            file_path = os.path.join(data_dir, f"{subj_id}.npy")
            if os.path.exists(file_path):
                self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        volume = np.load(file_path)
        volume = np.expand_dims(volume, axis=0)
        x = torch.tensor(volume, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long if self.task == 'classification' else torch.float32)
        if self.transform:
            x = self.transform(x)
        return x, y
