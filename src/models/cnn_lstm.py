import torch
import torch.nn as nn
import yaml

def load_config(path='configs/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

class CnnLstmECG(nn.Module):
    """
    CNN-LSTM for single-lead ECG cardiac risk classification.
    CNN extracts local features (QRS shape, ST segment).
    LSTM captures temporal patterns across the 5-min window.
    """
    def __init__(self, input_length, cnn_filters,
                 kernel_size, lstm_hidden, lstm_layers,
                 dropout=0.3, num_classes=1):
        super().__init__()
        
        # CNN feature extractor
        cnn_blocks = []
        in_ch = 1
        for out_ch in cnn_filters:
            cnn_blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)
        
        # LSTM temporal encoder
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            #nn.Sigmoid(),
        )
    
    def forward(self, x):
        # x: (batch, 1, signal_length)
        cnn_out = self.cnn(x)            # (batch, filters[-1], reduced_len)
        cnn_out = cnn_out.permute(0,2,1) # (batch, seq_len, features)
        _, (hidden, _) = self.lstm(cnn_out)
        last_hidden = hidden[-1]         # (batch, lstm_hidden)
        return self.classifier(last_hidden)

def build_model(config_path='configs/config.yaml', input_length=150_000):
    cfg = load_config(config_path)['models']['cnn_lstm']
    return CnnLstmECG(
        input_length=input_length,
        cnn_filters=cfg['cnn_filters'],
        kernel_size=cfg['cnn_kernel_size'],
        lstm_hidden=cfg['lstm_hidden_size'],
        lstm_layers=cfg['lstm_layers'],
        dropout=cfg['dropout'],
    )

if __name__ == '__main__':
    model = build_model()
    dummy = torch.randn(4, 1, 150_000)  # batch=4, 5-min windows
    out = model(dummy)
    print(f'Output shape: {out.shape}')      # Expect: torch.Size([4, 1])
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {params:,}')
    print('Model OK' if out.shape == torch.Size([4, 1]) else 'SHAPE MISMATCH')