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
    DLinear Baseline Model for multivariate time series.
    Uses moving average to decompose the sequence into trend and remainder components.
    """
    def __init__(self, lookback, num_assets, trend_window=20):
        super(DLinear, self).__init__()
        self.lookback = lookback
        self.num_assets = num_assets
        self.trend_window = trend_window
        
        # Linear layer mapping the temporal dimension (lookback) down to 1 
        # for both trend and remainder
        self.Linear_Trend = nn.Linear(self.lookback, 1)
        self.Linear_Remainder = nn.Linear(self.lookback, 1)
        
        # Projection to signal per asset
        self.proj = nn.Linear(num_assets, num_assets)

    def forward(self, x):
        """
        x: [batch, lookback, num_assets] (simplified feature space for DLinear)
        """
        # Trend Decomposition
        # Using a simple average pooling for moving average
        kernel_size = self.trend_window
        padding = kernel_size // 2
        
        # Handle padding for valid convolution/pooling
        # A proper implementation handles edge cases carefully.
        # For simplicity in this benchmark, we approximate:
        trend = nn.functional.avg_pool1d(x.transpose(1, 2), kernel_size=kernel_size, stride=1, padding=padding)
        trend = trend[:, :, :self.lookback].transpose(1, 2)
        
        remainder = x - trend
        
        # Temporal Linear mapping
        # trend: [batch, lookback, num_assets] -> [batch, num_assets, lookback]
        trend_out = self.Linear_Trend(trend.transpose(1, 2)).squeeze(-1) # [batch, num_assets]
        rem_out = self.Linear_Remainder(remainder.transpose(1, 2)).squeeze(-1) # [batch, num_assets]
        
        out = trend_out + rem_out
        
        # Optional cross-asset projection
        out = self.proj(out)
        
        # Bounded signal
        return torch.tanh(out)
