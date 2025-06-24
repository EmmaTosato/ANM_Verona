# datasets.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class FCDataset(Dataset):
    def __init__(self, data_dir, df_labels, label_column, task, transform=None):
        self.data_dir = data_dir
        self.df_labels = df_labels.reset_index(drop=True)
        self.label_column = label_column
        self.task = task
        self.transform = transform

        # Dictionary for mapping strings to indices if labels are not numbers
        if not pd.api.types.is_numeric_dtype(self.df_labels[self.label_column]):
            unique_labels = sorted(self.df_labels[self.label_column].unique())
            self.label_mapping = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_mapping = None

        self.samples = []

        # Loop over each row of the dataframe
        for _, row in self.df_labels.iterrows():
            subj_id = row['ID']
            label = self.label_mapping[row[self.label_column]] if self.task == 'classification' else float(row[self.label_column])
            file_path = os.path.join(data_dir, f"{subj_id}.processed.npy")
            if os.path.exists(file_path):
                self.samples.append((file_path, label))
            else:
                print(f"Missing file: {file_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        subj_id = os.path.basename(file_path).replace('.processed.npy', '')

        # Load and reshape the volume: (1, 91, 109, 91)
        volume = np.load(file_path)
        volume = np.expand_dims(volume, axis=0)

        # Covert volume into a tensor
        x = torch.tensor(volume, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long if self.task == 'classification' else torch.float32)
        if self.transform:
            x = self.transform(x)

        return {'X': x, 'y': y, 'id': subj_id}


class AugmentedFCDataset(Dataset):
    def __init__(self, data_dir, df_labels, label_column, task, transform=None):

        self.data_dir = data_dir
        self.df_labels = df_labels.reset_index(drop=True)
        self.label_column = label_column
        self.task = task
        self.transform = transform

        # Mapping
        if not pd.api.types.is_numeric_dtype(self.df_labels[self.label_column]):
            unique_labels = sorted(self.df_labels[self.label_column].unique())
            self.label_mapping = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_mapping = None

        self.samples = []

        for _, row in self.df_labels.iterrows():
            subj_id = row['ID']
            label = self.label_mapping[row[self.label_column]] if self.task == 'classification' else float(row[self.label_column])
            subject_folder = os.path.join(data_dir, subj_id)
            if os.path.isdir(subject_folder):
                for file in os.listdir(subject_folder):
                    if file.endswith('.npy'):
                        file_path = os.path.join(subject_folder, file)
                        self.samples.append((file_path, label))
            else:
                print(f"Warning: missing augmented folder for subject {subj_id}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        subj_id = os.path.basename(file_path).replace('.npy', '').split('_')[0]  # o regola secondo tuo naming

        # Load and reshape the volume: (1, 91, 109, 91)
        volume = np.load(file_path)
        volume = np.expand_dims(volume, axis=0)

        x = torch.tensor(volume, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long if self.task == 'classification' else torch.float32)
        if self.transform:
            x = self.transform(x)

        return {'X': x, 'y': y, 'id': subj_id}


