import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from models.lstm import LSTMAutoencoder, LSTMVAE
from models.gru import GRUAutoencoder, GRUVAE
from torch.utils.data import DataLoader

from utils import (
    ReconstructionDataModule,
    get_latest_checkpoint,
    get_latest_log,
    load_ae_data,
    split_data,
)


def predict(model, dataset):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction="sum").to(model.device)
    with torch.no_grad():
        model = model.eval()
        for input in dataset:
            input = input.to(model.device)
            pred = model.forward(input)
            if type(pred) == tuple:
                pred = pred[0]
            input = input.view(-1, input.size(0))
            loss = criterion(pred, input)
            predictions.append(pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses


def plot_prediction(data, prediction, loss, ax):
    if data.ndim == 3:
        data = data.flatten()
    df = pd.DataFrame(
        {
            "time": list(range(len(data))),
            "input": data.flatten(),
            "reconstructed": prediction,
        }
    )

    dfl = pd.melt(df, ["time"], var_name="label")
    sns.lineplot(
        data=dfl,
        x="time",
        y="value",
        ax=ax,
        hue="label",
        estimator=None,
    )
    ax.legend(loc="lower right")


def main():
    X, y = load_ae_data("./data/reconstruction/ecg.csv")
    X_normal = X[y == 1]
    X_anomalous = X[y == 0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test = split_data(X_normal, 0.1, 0.7)

    print(f"Normal shape: {X_normal.shape}")
    print(f"Anomalous shape: {X_anomalous.shape}")
    print(f"Test shape: {test.shape}")

    test_dataloader = DataLoader(
        ReconstructionDataModule(test.values, device),
        batch_size=1,
        num_workers=0,
    )
    anomaly_dataloader = DataLoader(
        ReconstructionDataModule(X_anomalous.values, device),
        batch_size=1,
        num_workers=0,
    )

    checkpoints = Path("checkpoints/autoencoder")
    seq_len = X.shape[1]
    hidden_dim = 128
    models = {
        "LSTM": LSTMAutoencoder.load_from_checkpoint(
            get_latest_checkpoint(checkpoints, "lstm"),
            seq_len=seq_len,
            n_features=1,
            embedding_dim=hidden_dim,
            device=device,
        ),
        "GRU": GRUAutoencoder.load_from_checkpoint(
            get_latest_checkpoint(checkpoints, "gru"),
            seq_len=seq_len,
            n_features=1,
            embedding_dim=hidden_dim,
            device=device,
        ),
        "LSTM-VAE": LSTMVAE.load_from_checkpoint(
            get_latest_checkpoint(checkpoints, "lstm-vae"),
            seq_len=seq_len,
            n_features=1,
            hidden_dim=hidden_dim,
            latent_dim=hidden_dim // 2,
            device=device,
        ),
        "GRU-VAE": GRUVAE.load_from_checkpoint(
            get_latest_checkpoint(checkpoints, "gru-vae"),
            seq_len=seq_len,
            n_features=1,
            hidden_dim=hidden_dim,
            latent_dim=hidden_dim // 2,
            device=device,
        ),
    }

    results_dir = Path("results") / "reconstruction"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, (model_name, model) in enumerate(models.items()):
        metrics_file = get_latest_log(
            Path("lightning_logs") / "reconstruction", model_name
        )
        print(
            f"[{model_name}] Generating train and validation loss plots from {metrics_file}"
        )

        metrics = pd.read_csv(
            metrics_file,
            usecols=["epoch", "train_loss", "val_loss"],
        )

        # Ensure only one entry per epoch
        metrics = metrics.groupby("epoch").mean().reset_index()

        ax = axs[i // 2, i % 2]
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
        ax.set_title(f"{model_name} Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "train_val_loss.png")

    predictions = {}
    for i, (model_name, model) in enumerate(models.items()):
        print(f"[{model_name}] Predicting on test data...")
        normal_preds, normal_losses = predict(model, test_dataloader)

        print(f"[{model_name}] Predicting on anomaly data...")
        anomaly_preds, anomaly_losses = predict(model, anomaly_dataloader)

        predictions[model_name] = {
            "normal": (normal_preds, normal_losses),
            "anomalous": (anomaly_preds, anomaly_losses),
        }

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, (model_name, model) in enumerate(models.items()):
        normal_preds, normal_losses = predictions[model_name]["normal"]
        anomaly_preds, anomaly_losses = predictions[model_name]["anomalous"]

        ax = axs[i // 2, i % 2]
        plot_prediction(test.values[0], normal_preds[0], normal_losses[0], ax)
        ax.set_title(
            f"{model_name} Reconstruction on Normal Data (loss: {np.around(normal_losses[0], 2)})"
        )
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(results_dir / "normal.png")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, (model_name, model) in enumerate(models.items()):
        normal_preds, normal_losses = predictions[model_name]["normal"]
        anomaly_preds, anomaly_losses = predictions[model_name]["anomalous"]

        ax = axs[i // 2, i % 2]
        plot_prediction(X_anomalous.values[0], anomaly_preds[0], anomaly_losses[0], ax)
        ax.set_title(
            f"{model_name} Reconstruction on Anomalous Data (loss: {np.around(anomaly_losses[0], 2)})"
        )
    plt.tight_layout()
    plt.savefig(results_dir / "anomalous.png")

    # Plotting loss distribution and calculating threshold
    thresholds = {}

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, (model_name, model) in enumerate(models.items()):
        normal_preds, normal_losses = predictions[model_name]["normal"]
        anomaly_preds, anomaly_losses = predictions[model_name]["anomalous"]

        ax = axs[i // 2, i % 2]
        sns.histplot(
            normal_losses,
            bins=50,
            kde=True,
            label="Normal",
            color="blue",
            ax=ax,
        )
        sns.histplot(
            anomaly_losses,
            bins=50,
            kde=True,
            label="Anomalous",
            color="red",
            ax=ax,
        )
        threshold = np.percentile(normal_losses, 95)
        thresholds[model_name] = threshold

        ax.axvline(threshold, color="green", linestyle="--", label="Threshold")
        ax.annotate(
            f"Threshold: {threshold:.3f}",
            xy=(threshold, ax.get_ylim()[1]),
            xytext=(threshold + 0.1, ax.get_ylim()[1] * 0.85),
            arrowprops=dict(arrowstyle="->", lw=1),
            fontsize=9,
            color="green",
        )
        ax.set_xlabel("Loss")
        ax.set_ylabel("Density")
        ax.set_title(f"{model_name} Loss Distribution")
        ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "loss_distribution.png")

    # Confusion matrix
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, (model_name, model) in enumerate(models.items()):
        normal_preds, normal_losses = predictions[model_name]["normal"]
        anomaly_preds, anomaly_losses = predictions[model_name]["anomalous"]
        threshold = thresholds[model_name]

        y_pred = [
            1 if loss < threshold else 0 for loss in normal_losses + anomaly_losses
        ]
        y_true = [1] * len(normal_losses) + [0] * len(anomaly_losses)

        print(f"[{model_name}] Classification report:")
        print(classification_report(y_true, y_pred, target_names=["Anomaly", "Normal"]))

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
        cm_pct = np.array(
            [f"{count}\n({pct:.2%})" for count, pct in zip(cm.ravel(), cm_norm.ravel())]
        ).reshape(cm.shape)
        cm_labels = ["Anomaly", "Normal"]

        ax = axs[i // 2, i % 2]
        sns.heatmap(
            cm,
            annot=cm_pct,
            fmt="",
            xticklabels=cm_labels,
            yticklabels=cm_labels,
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 30},
            ax=ax,
        )
        ax.set_title(f"{model_name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["Anomaly", "Normal"])
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Anomaly", "Normal"])
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png")

    # Plot ROC curve
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, (model_name, model) in enumerate(models.items()):
        normal_preds, normal_losses = predictions[model_name]["normal"]
        anomaly_preds, anomaly_losses = predictions[model_name]["anomalous"]
        threshold = thresholds[model_name]

        y_true = np.concatenate(
            [np.zeros(len(normal_losses)), np.ones(len(anomaly_losses))]
        )
        y_scores = np.concatenate([normal_losses, anomaly_losses])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)

        ax = axs[i // 2, i % 2]
        ax.plot(fpr, tpr, label=f"AUC: {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title(f"{model_name} ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curve.png")

    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
