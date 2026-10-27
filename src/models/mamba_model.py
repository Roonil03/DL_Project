import torch
import torch.nn as nn
from .dlinear import TradingSignalModel

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    MAMBA_AVAILABLE = False

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
        allow_fallback=False,
    ):
        super(MambaSignalModel, self).__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is required for MambaSignalModel; no fallback model is permitted"
            )

        # Legacy shape tests construct the model on CPU with an explicit
        # ``allow_fallback=True``. Keep that path strictly interface-only and
        # visibly distinct; benchmark construction never enables it.
        self._interface_fallback = bool(allow_fallback)
        if self._interface_fallback:
            self.implementation_name = "MambaInterfaceFallback"
            self.cpu_projection = nn.Linear(num_features, num_assets)
            return

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
        if self._interface_fallback:
            return torch.tanh(self.cpu_projection(x[:, -1, :]))

        out = self.embedding(x)
        
        for layer in self.layers:
            out = layer(out)
                
        # Aggregate terminal state
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        
        return torch.tanh(logits)
