import torch
import torch.nn as nn
import warnings
from .dlinear import TradingSignalModel

# Lazy import for mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    warnings.warn(
        "mamba_ssm not found. The Mamba benchmark will raise unless an LSTM "
        "fallback is explicitly enabled for debugging."
    )

class MambaSignalModel(TradingSignalModel):
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

        if not MAMBA_AVAILABLE and not allow_fallback:
            raise ImportError(
                "mamba_ssm is required for the Mamba benchmark. Install the "
                "optional dependency or pass allow_fallback=True for shape/debug "
                "checks only. Fallback results must not be reported as Mamba."
            )

        self.uses_mamba = MAMBA_AVAILABLE
        self.implementation_name = "Mamba" if self.uses_mamba else "LSTM fallback (debug only)"
        
        self.embedding = nn.Linear(num_features, d_model)
        
        if self.uses_mamba:
            # Construct a stack of Mamba blocks
            self.layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                for _ in range(n_layers)
            ])
        else:
            # Fallback approximation for structural coherence if package is missing
            self.layers = nn.ModuleList([
                nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)
                for _ in range(n_layers)
            ])
            
        self.fc = nn.Linear(d_model, num_assets)

    def forward(self, x):
        """
        x: [batch, lookback, num_features]
        """
        out = self.embedding(x)
        
        for layer in self.layers:
            if self.uses_mamba:
                out = layer(out)
            else:
                out, _ = layer(out)
                
        # Aggregate terminal state
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        
        return torch.tanh(logits)
