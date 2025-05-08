import torch
import torch.nn as nn
import lightning as pl


class AnomalyDetector(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float) -> None:
        super(AnomalyDetector, self).__init__()
        self.model = model
        self.criterion = torch.nn.L1Loss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)

        self.log("train_loss", loss.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)

        self.log("val_loss", loss.item(), prog_bar=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self(x)
        return y_pred

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)


class DeepAntPredictor(nn.Module):
    def __init__(
        self, feature_dim: int, window_size: int, hidden_size: int = 256
    ) -> None:
        super(DeepAntPredictor, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=feature_dim, out_channels=64, kernel_size=3, padding="valid"
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(
                in_features=(window_size - 2) // 4 * 128, out_features=hidden_size
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=hidden_size, out_features=feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
