"""
src/models/cnn_lstm.py

CNN-LSTM hybrid model for temporal ECG risk prediction.
- CNN layers extract local signal features (QRS complexes, ST segments)
- LSTM layers capture temporal dependencies across the 5-min window
"""

import torch
import torch.nn as nn
import yaml


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


class CnnLstmECG(nn.Module):
    """
    Hybrid CNN-LSTM for single-lead ECG cardiac risk classification.

    Architecture:
        Input  → [Conv1D → BN → ReLU → MaxPool] x3
               → LSTM (2 layers, bidirectional optional)
               → Dropout → FC → Sigmoid
    """

    def __init__(self, input_length: int, cnn_filters: list,
                 kernel_size: int, lstm_hidden: int, lstm_layers: int,
                 dropout: float = 0.3, num_classes: int = 1):
        super().__init__()

        # ── CNN Feature Extractor ───────────────────────────────
        cnn_blocks = []
        in_channels = 1
        for out_channels in cnn_filters:
            cnn_blocks += [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_blocks)

        # Compute flattened time dimension after pooling
        cnn_output_len = input_length
        for _ in cnn_filters:
            cnn_output_len = cnn_output_len // 2

        # ── LSTM Temporal Encoder ───────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # ── Classification Head ─────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, signal_length)
        Returns:
            Risk probability in [0, 1], shape (batch, 1)
        """
        # CNN: (batch, channels, length) → (batch, filters[-1], reduced_len)
        cnn_out = self.cnn(x)

        # Reshape for LSTM: (batch, seq_len, features)
        cnn_out = cnn_out.permute(0, 2, 1)

        # LSTM: use last hidden state
        _, (hidden, _) = self.lstm(cnn_out)
        last_hidden = hidden[-1]  # (batch, lstm_hidden)

        return self.classifier(last_hidden)


def build_model_from_config(config_path="configs/config.yaml",
                             input_length: int = 150_000) -> CnnLstmECG:
    """
    Instantiate CnnLstmECG using project config.

    Args:
        config_path:  Path to config.yaml
        input_length: Samples per window (fs * window_min * 60)
                      Default: 500 Hz × 5 min × 60 s = 150,000

    Returns:
        Initialized CnnLstmECG model.
    """
    cfg = load_config(config_path)["models"]["cnn_lstm"]
    return CnnLstmECG(
        input_length=input_length,
        cnn_filters=cfg["cnn_filters"],
        kernel_size=cfg["cnn_kernel_size"],
        lstm_hidden=cfg["lstm_hidden_size"],
        lstm_layers=cfg["lstm_layers"],
        dropout=cfg["dropout"],
    )


if __name__ == "__main__":
    model = build_model_from_config()
    dummy_input = torch.randn(8, 1, 150_000)  # batch=8, 5-min window
    output = model(dummy_input)
    print(f"✓ CNN-LSTM OK | Output shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")
