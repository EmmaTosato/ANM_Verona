import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class FCDataset(Dataset):
    def __init__(self, data_dir, labels_csv, label_column='Group'):
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.label_column = label_column

        # List of (file_path, label)
        self.samples = []
        for _, row in self.labels_df.iterrows():
            subj_id = row['ID']
            label = row[self.label_column]
            file_path = os.path.join(data_dir, f"{subj_id}.npy")
            if os.path.exists(file_path):
                self.samples.append((file_path, label))
            else:
                print(f"⚠️ Missing file: {file_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        # Load FC map (shape: D, H, W)
        volume = np.load(file_path)  # (91, 109, 91)

        # Add channel dimension (C, D, H, W)
        volume = np.expand_dims(volume, axis=0)  # (1, 91, 109, 91)

        # Convert to tensors
        x = torch.tensor(volume, dtype=torch.float32)
        y = torch.tensor(label)

        return x, y
