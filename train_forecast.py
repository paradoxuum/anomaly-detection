import torch
import lightning as pl
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from models.lstm import LSTMForecaster
from models.gru import GRUForecaster
from models.deepant import AnomalyDetector, DeepAntPredictor
from utils import (
    ForecastingDataset,
    create_windows,
    get_split_slices,
    load_forecasting_data,
    split_data,
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="lstm", choices=["lstm", "gru", "deepant"]
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="./data/forecasting/TravelTime_451.csv",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_forecasting_data(args.dataset)
    X, y = create_windows(data, 10)
    train, val, _ = get_split_slices(len(X), 0.1, 0.1)

    train_dataset = ForecastingDataset(X[train], y[train], device)
    val_dataset = ForecastingDataset(X[val], y[val], device)

    print(
        f"Train shape: {train_dataset.data_x.shape}, "
        f"Val shape: {val_dataset.data_x.shape}"
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    if args.model == "lstm":
        model = LSTMForecaster(input_dim=1, hidden_dim=150, output_dim=1)
    elif args.model == "gru":
        model = GRUForecaster(input_dim=1, hidden_dim=150, output_dim=1)
    elif args.model == "deepant":
        predictor = DeepAntPredictor(
            1,
            10,
        )
        model = AnomalyDetector(predictor, 1e-3)

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[
            pl.pytorch.callbacks.ModelCheckpoint(
                monitor="val_loss",
                dirpath="./checkpoints/forecasting",
                filename=args.model + "-{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
                mode="min",
            ),
            pl.pytorch.callbacks.EarlyStopping(
                monitor="val_loss", patience=30, mode="min"
            ),
        ],
        logger=pl.pytorch.loggers.CSVLogger(
            save_dir="./lightning_logs/forecasting",
            name=args.model,
        ),
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    main()
