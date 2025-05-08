import logging
import lightning as pl
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from models.lstm import LSTMAutoencoder, LSTMVAE
from models.gru import GRUAutoencoder, GRUVAE
from utils import ReconstructionDataModule, load_ae_data, split_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["lstm", "lstm-vae", "gru", "gru-vae"],
    )
    parser.add_argument("--dataset", type=str, default="./data/reconstruction/ecg.csv")
    args = parser.parse_args()

    X, y = load_ae_data(args.dataset)
    X_normal = X[y == 1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, test = split_data(X_normal, 0.1, 0.7)
    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {val.shape}")
    print(f"Test shape: {test.shape}")
    train_dataloader = DataLoader(ReconstructionDataModule(train.values, device))
    val_dataloader = DataLoader(ReconstructionDataModule(val.values, device))

    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints/autoencoder",
        filename=args.model + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    if args.model == "lstm":
        model = LSTMAutoencoder(train.shape[1], 1, 128)
    elif args.model == "gru":
        model = GRUAutoencoder(train.shape[1], 1, 128)
    elif args.model == "lstm-vae":
        model = LSTMVAE(train.shape[1], 1, hidden_dim=128, latent_dim=64)
    elif args.model == "gru-vae":
        model = GRUVAE(train.shape[1], 1, hidden_dim=128, latent_dim=64)
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback],
        logger=pl.pytorch.loggers.CSVLogger(
            save_dir="./lightning_logs/reconstruction",
            name=args.model,
        ),
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
