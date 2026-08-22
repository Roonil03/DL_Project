import torch
import torch.nn as nn

class NegativeSharpeLoss(nn.Module):
    """
    Differentiable Negative Annualized Sharpe Ratio Loss.
    Accepts either (signals, next_returns) or pre-computed portfolio returns.
    """
    def __init__(self, eps: float = 1e-6, ann_factor: float = 252.0, alpha: float = 1.0):
        super(NegativeSharpeLoss, self).__init__()
        self.eps = eps
        self.ann_factor = ann_factor
        self.alpha = alpha

    def forward(self, signals_or_returns: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """
        signals_or_returns: [batch_size, num_assets]
        targets: [batch_size, num_assets] next-period returns
        """
        if targets is not None:
            # signals_or_returns represents model output positions s_t
            # targets represents next period asset returns r_{t+1}
            # Element-wise product = asset return contribution
            port_returns = torch.mean(signals_or_returns * targets, dim=-1)
        else:
            if signals_or_returns.dim() == 2:
                port_returns = torch.mean(signals_or_returns, dim=-1)
            else:
                port_returns = signals_or_returns
                
        mean_ret = torch.mean(port_returns)
        # Sample variance with degrees of freedom correction
        var_ret = torch.var(port_returns, unbiased=True) if port_returns.numel() > 1 else torch.tensor(1.0, device=port_returns.device)
        
        # Annualized Sharpe
        ann_mult = torch.sqrt(torch.tensor(self.ann_factor, dtype=port_returns.dtype, device=port_returns.device))
        sharpe = (mean_ret / (torch.sqrt(var_ret + self.eps))) * ann_mult
        
        loss = -sharpe
        
        # Optional blend with directional MSE if alpha < 1.0
        if targets is not None and self.alpha < 1.0:
            # Scaled directional target: sign(targets)
            dir_target = torch.tanh(targets * 10.0)
            mse_loss = torch.mean((signals_or_returns - dir_target) ** 2)
            loss = self.alpha * loss + (1.0 - self.alpha) * mse_loss
            
        return loss

