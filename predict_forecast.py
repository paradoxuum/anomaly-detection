import seaborn as sns
import pandas as pd
import numpy as np
import lightning as pl
from typing import List, Tuple
from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import torch
from models.lstm import LSTMForecaster
from models.gru import GRUForecaster
from models.deepant import AnomalyDetector, DeepAntPredictor
from utils import (
    ForecastingDataset,
    get_latest_checkpoint,
    get_latest_log,
    get_split_slices,
    load_forecasting_data,
    create_windows,
)


def reconstruct_original_sequence(
    val_dataset: ForecastingDataset, test_dataset: ForecastingDataset
) -> Tuple[np.ndarray, int]:
    """
    Reconstruct the original time series from overlapping windows in the dataset (validation + test).

    Returns:
        Tuple[np.ndarray, int]:
            - full_seq: shape (val_length + test_length, feature_dim).
            - boundary_idx: the index separating validation data and test data.
    """
    val_windows = val_dataset.data_x
    val_reconstructed_list = []
    for i in range(len(val_windows)):
        window = val_windows[i]
        if i == 0:
            val_reconstructed_list.append(window)
        else:
            val_reconstructed_list.append(window[-1:])

    val_reconstructed = np.concatenate(val_reconstructed_list, axis=0)

    test_windows = test_dataset.data_x
    test_reconstructed_list = []
    for i in range(len(test_windows)):
        window = test_windows[i]
        test_reconstructed_list.append(window[-1:])

    test_reconstructed = np.concatenate(test_reconstructed_list, axis=0)

    boundary_idx = len(val_reconstructed)
    full_seq = np.concatenate([val_reconstructed, test_reconstructed], axis=0)
    return full_seq, boundary_idx


def calculate_threshold(anomaly_scores: np.ndarray, std_rate: int = 2) -> float:
    feature_scores = anomaly_scores[:, 0]
    return np.mean(feature_scores) + std_rate * np.std(feature_scores)


def identify_anomalies(anomaly_scores: np.ndarray, threshold: float) -> List[int]:
    return [i for i, score in enumerate(anomaly_scores) if score > threshold]


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="./data/TravelTime_451.csv",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_forecasting_data(args.dataset)
    X, y = create_windows(data, 10)
    _, val, test = get_split_slices(len(X), 0.1, 0.1)

    test_dataset = ForecastingDataset(X[test], y[test], device)
    print("Test shape: ", test_dataset.data_x.shape)

    test_loader = DataLoader(
        test_dataset, batch_size=test_dataset.data_x.shape[0], shuffle=False
    )

    checkpoints_folder = Path("checkpoints/forecasting")
    models = {
        "LSTM": LSTMForecaster.load_from_checkpoint(
            get_latest_checkpoint(checkpoints_folder, "lstm")
        ),
        "GRU": GRUForecaster.load_from_checkpoint(
            get_latest_checkpoint(checkpoints_folder, "gru")
        ),
        "DeepAnT": AnomalyDetector.load_from_checkpoint(
            get_latest_checkpoint(checkpoints_folder, "deepant"),
            model=DeepAntPredictor(1, 10),
            lr=1e-3,
        ),
    }

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
    )

    ground_truth = test_loader.dataset.data_y.squeeze()
    if ground_truth.ndim == 1:
        ground_truth = ground_truth.reshape(-1, 1)

    results_dir = Path(f"results/forecasting")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    i = 0
    for model_name, model in models.items():
        metrics_file = get_latest_log(
            Path("lightning_logs") / "forecasting", model_name
        )
        print(
            f"[{model_name}] Generating train and validation loss plots from {metrics_file}"
        )

        metrics = pd.read_csv(
            metrics_file,
            usecols=["epoch", "train_loss", "val_loss"],
        )
        metrics = metrics.groupby("epoch").mean().reset_index()

        ax = axs[i]
        sns.lineplot(
            data=metrics,
            x="epoch",
            y="train_loss",
            label="Train Loss",
            estimator=None,
            ax=ax,
        )
        sns.lineplot(
            data=metrics,
            x="epoch",
            y="val_loss",
            label="Validation Loss",
            estimator=None,
            ax=ax,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model_name} Training and Validation Loss")
        ax.legend()
        i += 1

    plt.tight_layout()
    plt.savefig(results_dir / "train_val_loss.png")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    i = 0
    for model_name, model in models.items():
        print(f"[{model_name}] Running prediction...")
        output = trainer.predict(model, test_loader)

        predictions = output[0].numpy().squeeze()
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        anomaly_scores = np.abs(predictions - ground_truth)
        threshold = calculate_threshold(anomaly_scores)
        print(f"[{model_name}] Threshold: {threshold}")

        anomalies_indices = identify_anomalies(anomaly_scores, threshold)
        print(f"[{model_name}] Anomalies detected: {anomalies_indices}")

        original, boundary_idx = reconstruct_original_sequence(
            ForecastingDataset(X[val], y[val], device), test_dataset
        )
        time_steps = range(original.shape[0])

        ax = axs[i]
        sns.lineplot(
            x=time_steps[:boundary_idx],
            y=original[:boundary_idx, 0],
            label="Validation Data",
            color="blue",
            ax=ax,
        )
        sns.lineplot(
            x=time_steps[boundary_idx:],
            y=ground_truth[:, 0],
            label="Test Data",
            color="orange",
            ax=ax,
        )
        sns.lineplot(
            x=time_steps[boundary_idx:],
            y=predictions[:, 0],
            label="Predicted",
            color="green",
            ax=ax,
        )
        ax.axvline(
            x=boundary_idx,
            color="red",
            linestyle="--",
            label="Boundary Index",
        )
        ax.scatter(
            [time_steps[boundary_idx + i] for i in anomalies_indices],
            ground_truth[anomalies_indices, 0],
            color="red",
            label="Anomalies",
        )
        ax.set_title(f"{model_name} Predictions")
        ax.legend()
        i += 1

    plt.tight_layout()
    plt.savefig(results_dir / "predictions.png")


if __name__ == "__main__":
    main()
