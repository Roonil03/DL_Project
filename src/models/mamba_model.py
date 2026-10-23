import torch
import torch.nn as nn
from mamba_ssm import Mamba
from .dlinear import TradingSignalModel

class MambaSignalModel(TradingSignalModel):
    """
    Strict Mamba Signal Model for financial time-series forecasting.
    Enforces true Mamba implementation without fallback mechanisms.
    """
    def __init__(
        self,
        lookback,
        num_features,
        num_assets,
        d_model=64,
        n_layers=2,
    ):
        super(MambaSignalModel, self).__init__()

        self.implementation_name = "Mamba"
        
        self.embedding = nn.Linear(num_features, d_model)
        
        # Construct a genuine stack of Mamba blocks
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
            
        self.fc = nn.Linear(d_model, num_assets)

    def forward(self, x):
        """
        x: [batch, lookback, num_features]
        """
        out = self.embedding(x)
        
        for layer in self.layers:
            out = layer(out)
                
        # Aggregate terminal state
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        
        return torch.tanh(logits)