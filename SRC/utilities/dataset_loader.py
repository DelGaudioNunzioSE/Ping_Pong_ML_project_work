import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define custom dataset
class ArmDataset(Dataset):
    def __init__(self, csv_file, batch_size=124, seed=3, train_size=0.5, val_size=0.5):
        data_set = pd.read_csv(csv_file)

        target = data_set.iloc[:, :-2]
        input = data_set.iloc[:, 4:6]

        input_train, input_sub_test, target_train, target_sub_test = train_test_split(input, target, test_size=train_size, random_state=seed)

        input_test, input_val, target_test, target_val = train_test_split(input_sub_test, target_sub_test, test_size=val_size, random_state=seed)

        input_train_np = input_train.to_numpy()
        input_test_np = input_test.to_numpy()
        input_val_np = input_val.to_numpy()

        target_train_np = target_train.to_numpy()
        target_test_np = target_test.to_numpy()
        target_val_np = target_val.to_numpy()

        input_train = torch.tensor(input_train_np, dtype=torch.float32).to(device)
        input_test = torch.tensor(input_test_np, dtype=torch.float32).to(device)
        input_val = torch.tensor(input_val_np, dtype=torch.float32).to(device)

        target_train = torch.tensor(target_train_np, dtype=torch.float32).to(device)
        target_test = torch.tensor(target_test_np, dtype=torch.float32).to(device)
        target_val = torch.tensor(target_val_np, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(input_train, target_train)
        test_dataset = TensorDataset(input_test, target_test)
        validation_dataset = TensorDataset(input_val, target_val)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
