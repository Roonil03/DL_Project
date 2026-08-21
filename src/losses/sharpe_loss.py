import torch
import torch.nn as nn

class NegativeSharpeLoss(nn.Module):
    def __init__(self, eps=1e-6, ann_factor=252):
        super(NegativeSharpeLoss, self).__init__()
        self.eps = eps
        self.ann_factor = ann_factor

    def forward(self, returns: torch.Tensor):
        """
        Calculates the differentiable negative annualized Sharpe Ratio.
        returns: shape (batch_size, num_assets) or (batch_size) if portfolio returns are pre-aggregated.
        """
        # If returns are per-asset, we first compute the equal-weight portfolio return per timestep.
        if returns.dim() == 2:
            port_returns = returns.mean(dim=1)
        else:
            port_returns = returns
            
        mean_ret = torch.mean(port_returns)
        var_ret = torch.var(port_returns, unbiased=True)
        
        # Annualize
        sharpe = (mean_ret / (torch.sqrt(var_ret + self.eps))) * torch.sqrt(torch.tensor(self.ann_factor, dtype=returns.dtype, device=returns.device))
        
        return -sharpe
