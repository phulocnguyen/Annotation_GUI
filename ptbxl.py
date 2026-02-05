import os
import ast
import wfdb
import torch
from torch.utils.data import Dataset
import pandas as pd

class PTBXLDataset(Dataset):
    def __init__(self, root_dir, split="train", sampling_rate=500, max_length=5000, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length

        df = pd.read_csv(os.path.join(root_dir, "ptbxl_database.csv"))

        # Parse scp_codes
        df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

                # Load diagnostic mapping
        scp_df = pd.read_csv(os.path.join(root_dir, "scp_statements.csv"))

        # Map: description/code → diagnostic_class
        diag_map = {}
        for _, row in scp_df.iterrows():
            if row['diagnostic'] == 1:
                # dùng diagnostic_class làm label
                diag_map[row['diagnostic_subclass']] = row['diagnostic_class']

        print("Example diag_map:", list(diag_map.items())[:5])


        df['diagnostic_superclass'] = df['scp_codes'].apply(
            lambda x: list({diag_map[k] for k in x.keys() if k in diag_map})
        )

        print("Total records:", len(df))
        print("Records with diagnostic_superclass:", sum(df['diagnostic_superclass'].map(len) > 0))

        # Filter rows with valid superclass
        df = df[df['diagnostic_superclass'].map(len) > 0]

        # Check strat_fold column
        if 'strat_fold' in df.columns:
            folds = {"train": [1,2,3,4,5,6,7,8], "val": [9], "test": [10]}
            df = df[df['strat_fold'].isin(folds.get(split, []))]
            print("Records after split filter:", len(df))
        else:
            print("Warning: strat_fold column not found, using all data for split")

        # Set filenames
        filename_col = 'filename_hr' if sampling_rate==500 else 'filename_lr'
        if filename_col not in df.columns:
            print(f"Warning: {filename_col} column not found, using first available column")
            filename_col = df.columns[0]

        self.records = df[filename_col].values
        self.labels = df['diagnostic_superclass'].values

        # Label mapping
        self.classes = sorted(list(set(sum(self.labels, []))))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        print("Final dataset size:", len(self.records))
        print("Classes:", self.classes)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path = os.path.join(self.root_dir, self.records[idx])

        # Load ECG
        signal, _ = wfdb.rdsamp(record_path)
        signal = torch.tensor(signal, dtype=torch.float32)

        # Pad/crop
        if signal.shape[0] > self.max_length:
            signal = signal[:self.max_length]
        else:
            pad = self.max_length - signal.shape[0]
            signal = torch.nn.functional.pad(signal, (0,0,0,pad))

        signal = signal.transpose(0,1)  # (C, T)

        # Multi-hot labels
        label = torch.zeros(len(self.classes))
        for c in self.labels[idx]:
            label[self.class_to_idx[c]] = 1.0

        if self.transform:
            signal = self.transform(signal)

        return signal, label