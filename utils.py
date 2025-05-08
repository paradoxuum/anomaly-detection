import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import numpy as np
from typing import Tuple


class ReconstructionDataModule(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, device: str) -> None:
        """
        PyTorch Dataset for time series data.

        Each sample is a tuple:
            (input_sequence, target_sequence)
        where:
            - input_sequence: shape (num_features, window_size)
            - target_sequence: shape (num_features, window_size)

        Args:
            data (np.ndarray): Array of time series data,
                               shape (num_samples, window_size, num_features).
            device (str): Compute device to use ('cuda' or 'cpu').
        """
        self.data = data
        self.device = device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        val = torch.tensor(
            self.data[idx], device=self.device, dtype=torch.float32
        ).unsqueeze(1)
        return val


class ForecastingDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y, device):
        self.data_x = data_x
        self.data_y = data_y
        self.device = device

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return (
            torch.tensor(
                self.data_x[index], dtype=torch.float32, device=self.device
            ).transpose(0, 1),
            torch.tensor(
                self.data_y[index], dtype=torch.float32, device=self.device
            ).squeeze(0),
        )


def load_ae_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(
        data_path,
        header=None,
    )
    return (data.iloc[:, 1:], data.iloc[:, 0])


def load_forecasting_data(data_path: Path):
    data = pd.read_csv(
        data_path,
        index_col="timestamp",
        parse_dates=["timestamp"],
    )
    sc = MinMaxScaler()
    data_scaled = sc.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=data.columns, index=data.index)


def get_split_slices(
    data_len: int, val_split: float, test_split: float
) -> Tuple[slice, slice, slice]:
    train_slice = slice(None, int((1 - val_split - test_split) * data_len))
    val_slice = slice(
        int((1 - val_split - test_split) * data_len),
        int((1 - test_split) * data_len),
    )
    test_slice = slice(int((1 - test_split) * data_len), None)
    return train_slice, val_slice, test_slice


def split_data(data: np.ndarray, val_split: float, test_split: float):
    """
    Calculate the train, validation, and test splits.

    Args:
        val_split (float): Proportion of data to use for validation.
        test_split (float): Proportion of data to use for testing.

    Returns:
        Tuple[float, float, float]: Proportions for train, validation, and test sets.
    """
    data_len = len(data)
    train_slice = slice(None, int((1 - val_split - test_split) * data_len))
    val_slice = slice(
        int((1 - val_split - test_split) * data_len),
        int((1 - test_split) * data_len),
    )
    test_slice = slice(int((1 - test_split) * data_len), None)
    return data[train_slice], data[val_slice], data[test_slice]


def get_latest_log(folder: Path, model_name: str):
    folder = folder / model_name
    latest_version = None

    for version in folder.glob("version_*"):
        if not version.is_dir():
            continue

        version_num = version.name.split("_")[-1]
        if not version_num.isdigit():
            continue

        if latest_version is None or int(version_num) > int(latest_version):
            latest_version = version
    return version / "metrics.csv" if latest_version else None


def get_latest_checkpoint(folder: Path, model_name: str):
    for checkpoint in folder.glob("*.ckpt"):
        if model_name in checkpoint.stem:
            return checkpoint


def create_windows(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    data_x, data_y = [], []
    for i in range(window_size, data.shape[0]):
        if (i + 1) >= data.shape[0]:
            break
        window = data[i - window_size : i]
        target = data[i : i + 1]
        data_x.append(window)
        data_y.append(target)
    return np.array(data_x), np.array(data_y)
