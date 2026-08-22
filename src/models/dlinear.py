import torch
import torch.nn as nn

class TradingSignalModel(nn.Module):
    """
    Common interface for all models.
    Output: [batch, num_assets] in range [-1, 1] (via tanh)
    """
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class DLinear(TradingSignalModel):
    """
    DLinear Baseline Model for multivariate financial time series.
    Decomposes sequence into Trend and Seasonal/Remainder components via moving average,
    applies temporal linear projections, and projects to asset signals.
    """
    def __init__(self, lookback: int, num_features: int = 10, num_assets: int = None, trend_window: int = 5):
        super(DLinear, self).__init__()
        self.lookback = lookback
        if num_assets is None:
            self.num_features = num_features
            self.num_assets = num_features
        else:
            self.num_features = num_features
            self.num_assets = num_assets
            
        self.trend_window = max(1, min(trend_window, lookback))
        
        # Temporal Linear layers mapping lookback dimension to 1
        self.Linear_Trend = nn.Linear(self.lookback, 1)
        self.Linear_Remainder = nn.Linear(self.lookback, 1)
        
        # Cross-feature projection to asset signals
        self.proj = nn.Linear(self.num_features, self.num_assets)

    def forward(self, x):
        """
        x: [batch, lookback, num_features]
        """
        batch_size, seq_len, n_feat = x.shape
        
        # Transpose to [batch, num_features, lookback] for 1D pooling
        x_trans = x.transpose(1, 2)
        
        # Moving average trend decomposition with padding
        padding = (self.trend_window - 1) // 2
        trend = nn.functional.avg_pool1d(
            x_trans, 
            kernel_size=self.trend_window, 
            stride=1, 
            padding=padding, 
            count_include_pad=False
        )
        # Ensure length matches lookback
        if trend.shape[-1] < self.lookback:
            trend = nn.functional.pad(trend, (0, self.lookback - trend.shape[-1]), mode='replicate')
        elif trend.shape[-1] > self.lookback:
            trend = trend[:, :, :self.lookback]
            
        remainder = x_trans - trend
        
        # Temporal mappings: [batch, num_features, lookback] -> [batch, num_features]
        trend_out = self.Linear_Trend(trend).squeeze(-1)
        rem_out = self.Linear_Remainder(remainder).squeeze(-1)
        
        combined = trend_out + rem_out
        
        # Project to target assets
        out = self.proj(combined)
        
        return torch.tanh(out)

