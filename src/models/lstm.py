import torch
import torch.nn as nn
from .dlinear import TradingSignalModel

class LSTMSignalModel(TradingSignalModel):
    def __init__(self, lookback, num_features, num_assets, hidden_dim=64, num_layers=2, dropout=0.1):
        super(LSTMSignalModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_dim, num_assets)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [batch, lookback, num_features]
        """
        out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_out = out[:, -1, :] # [batch, hidden_dim]
        last_out = self.dropout(last_out)
        
        logits = self.fc(last_out) # [batch, num_assets]
        return torch.tanh(logits)
