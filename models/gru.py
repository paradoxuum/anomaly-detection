import lightning as pl
import torch
from torch import optim, nn


class GRUForecaster(pl.LightningModule):
    def __init__(self, input_dim=1, hidden_dim=150, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.L1Loss()

    def forward(self, x):
        x = x.view(x.size(0), -1, 1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class GRUEncoder(nn.Module):
    def __init__(self, seq_len, n_features=1, embedding_dim=20):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        _, hidden_n = self.rnn(x)
        return hidden_n.reshape((self.n_features, self.embedding_dim))


class GRUDecoder(nn.Module):
    def __init__(self, seq_len, n_features=1, input_dim=20):
        super(GRUDecoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.input_dim = input_dim
        self.hidden_dim = 2 * input_dim

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))

        x, _ = self.rnn(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        return self.output_layer(x)


class GRUAutoencoder(pl.LightningModule):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()

        self.encoder = GRUEncoder(seq_len, n_features, embedding_dim)
        self.decoder = GRUDecoder(seq_len, n_features, embedding_dim)

    def forward(self, x):
        x = x.view(-1, x.size(0))
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class GRUVAE(pl.LightningModule):
    def __init__(self, seq_len, n_features, hidden_dim=64, latent_dim=20):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_z2h = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_features)

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        _, x = self.encoder(x)
        x = x[-1]

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparametize(mu, logvar)

        # initialize hidden state
        h0 = torch.tanh(self.fc_z2h(z))  # [B, H]
        # split into (n_layers, B, H)

        h0 = h0.view(1, z.size(0), self.hidden_dim)
        # start tokens: zeros
        inputs = torch.zeros(
            z.size(0), self.seq_len, self.fc_out.out_features, device=z.device
        )
        outputs, _ = self.decoder(inputs, h0)
        #   h = self.fc_z2h(z)

        #   z = z.repeat(1, self.seq_len, 1)
        #   z = z.reshape((1, self.seq_len, self.latent_dim)).to(self.device)

        #   output, (_, _) = self.decoder(z, (h.contiguous(), h.contiguous()))
        return self.fc_out(outputs), mu, logvar

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, logvar = self.forward(x)
        loss, _, _ = self.loss(x, x_hat, mu, logvar)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, logvar = self.forward(x)
        loss, _, _ = self.loss(x, x_hat, mu, logvar)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss(self, x, x_rec, mu, logvar, beta=1.0):
        # Reconstruction loss (MSE)
        recon = nn.functional.mse_loss(x_rec, x, reduction="mean")
        # KL divergence
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kld, recon, kld
