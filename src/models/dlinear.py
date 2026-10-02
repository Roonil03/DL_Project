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
    Enhanced DLinear Baseline Model with independent seasonal and trend projections.
    """
    def __init__(self, lookback: int, num_features: int = 10, num_assets: int = None, trend_window: int = 25):
        super(DLinear, self).__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.num_assets = num_assets if num_assets is not None else num_features
        
        self.trend_window = max(3, min(trend_window, lookback))
        
        # Enhanced linear projections with an intermediate bottleneck for better feature extraction
        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.lookback, self.lookback // 2),
            nn.GELU(),
            nn.Linear(self.lookback // 2, 1)
        )
        self.linear_trend = nn.Sequential(
            nn.Linear(self.lookback, self.lookback // 2),
            nn.GELU(),
            nn.Linear(self.lookback // 2, 1)
        )
        
        # Cross-feature projection to target asset signals
        self.proj = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.LayerNorm(self.num_features),
            nn.Tanh(),
            nn.Linear(self.num_features, self.num_assets)
        )

    def forward(self, x):
        """
        x: [batch, lookback, num_features]
        """
        batch_size, seq_len, n_feat = x.shape
        x_trans = x.transpose(1, 2)  # [batch, num_features, lookback]
        
        # Moving average decomposition
        padding = (self.trend_window - 1) // 2
        trend = nn.functional.avg_pool1d(
            x_trans, 
            kernel_size=self.trend_window, 
            stride=1, 
            padding=padding, 
            count_include_pad=False
        )
        
        if trend.shape[-1] < self.lookback:
            trend = nn.functional.pad(trend, (0, self.lookback - trend.shape[-1]), mode='replicate')
        elif trend.shape[-1] > self.lookback:
            trend = trend[:, :, :self.lookback]
            
        seasonal = x_trans - trend
        
        # Project temporal dimensions
        trend_out = self.linear_trend(trend).squeeze(-1)       # [batch, num_features]
        seasonal_out = self.linear_seasonal(seasonal).squeeze(-1) # [batch, num_features]
        
        combined = trend_out + seasonal_out
        out = self.proj(combined)
        
        return torch.tanh(out)