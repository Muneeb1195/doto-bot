import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

_DEVICE = torch.device("cpu")
_N_EPOCHS = 50
_BATCH_SIZE = 128
_LEARNING_RATE = 1e-3
_PATIENCE = 8
_SEQ_LEN = 20


def _to_device(tensor):
    return tensor.to(_DEVICE)


def _build_sequences(X, seq_len):
    n = len(X)
    if n <= seq_len:
        return np.expand_dims(X, 0), np.array([0])
    seqs = np.lib.stride_tricks.sliding_window_view(X, seq_len, axis=0)
    if seqs.shape[-1] == seq_len:
        seqs = seqs.transpose(0, 2, 1)
    idxs = np.arange(seq_len - 1, n)
    return seqs, idxs


class _TemporalFusionTransformer(nn.Module):
    """Minimal TFT for binary classification on tabular time series.

    Architecture:
      - LSTM encoder (2 layers)
      - Self-attention over the LSTM output sequence
      - Gated residual skip connection
      - Linear projection → sigmoid output
    """

    def __init__(self, n_features, hidden_size=64, n_lstm_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.skip = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        g = self.gate(attn_out)
        skip = self.skip(lstm_out)
        gated = g * attn_out + (1 - g) * skip
        gated = self.norm(gated)
        pooled = gated[:, -1, :]
        logits = self.fc_out(pooled)
        return logits


class TFTClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper around the TFT model."""

    def __init__(self, n_features, seq_len=_SEQ_LEN, hidden_size=64, n_lstm_layers=2, dropout=0.2):
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.dropout = dropout
        self.model = None
        self._classes = np.array([0, 1])
        self._fitted = False

    def fit(self, X, y, X_val=None, y_val=None):
        n_features = X.shape[1]
        model = _TemporalFusionTransformer(
            n_features=n_features,
            hidden_size=self.hidden_size,
            n_lstm_layers=self.n_lstm_layers,
            dropout=self.dropout,
        ).to(_DEVICE)

        seqs, idxs = _build_sequences(X, self.seq_len)
        labels = y[idxs]
        seqs_t = _to_device(torch.tensor(seqs, dtype=torch.float32))
        labels_t = _to_device(torch.tensor(labels, dtype=torch.float32).view(-1, 1))

        dataset = TensorDataset(seqs_t, labels_t)
        loader = DataLoader(dataset, batch_size=_BATCH_SIZE, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) > self.seq_len:
            val_seqs, val_idxs = _build_sequences(X_val, self.seq_len)
            val_labels = y_val[val_idxs]
            val_seqs_t = _to_device(torch.tensor(val_seqs, dtype=torch.float32))
            val_labels_t = _to_device(torch.tensor(val_labels, dtype=torch.float32).view(-1, 1))
            val_loader = DataLoader(TensorDataset(val_seqs_t, val_labels_t), batch_size=_BATCH_SIZE, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=_LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()
        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(_N_EPOCHS):
            model.train()
            train_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(loader)

            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        logits = model(bx)
                        val_loss += criterion(logits, by).item()
                val_loss /= len(val_loader)
                if val_loss < best_loss - 1e-4:
                    best_loss = val_loss
                    best_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    pass
                if patience_counter >= _PATIENCE:
                    break
            else:
                if train_loss < best_loss - 1e-4:
                    best_loss = train_loss
                    best_state = model.state_dict()

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        self._fitted = True
        return self

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("TFT model not fitted yet")
        self.model.eval()
        seqs, _ = _build_sequences(np.asarray(X), self.seq_len)
        if len(seqs) == 0:
            return np.full((len(X), 2), 0.5)
        seqs_t = _to_device(torch.tensor(seqs, dtype=torch.float32))
        with torch.no_grad():
            logits = self.model(seqs_t)
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
        full = np.full((len(X), 2), 0.5)
        last_idx = min(len(proba), len(X))
        full[-last_idx:, 1] = proba[-last_idx:]
        full[-last_idx:, 0] = 1.0 - proba[-last_idx:]
        return full

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    @property
    def classes_(self):
        return self._classes

    def get_params(self, deep=True):  # noqa: ARG002
        return {
            "n_features": self.n_features,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "n_lstm_layers": self.n_lstm_layers,
            "dropout": self.dropout,
        }


def train_tft(X, y, X_val=None, y_val=None, n_features=None):  # noqa: ARG001
    if n_features is None:
        n_features = X.shape[1]
    model = TFTClassifier(n_features=n_features)
    model.fit(X, y, X_val=X_val, y_val=y_val)
    return model
