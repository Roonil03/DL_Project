import torch
import torch.nn as nn
from .dlinear import TradingSignalModel

class PatchTSTSignalModel(TradingSignalModel):
    """
    Simplified implementation of PatchTST for benchmark purposes.
    Segments the time series into patches before applying Transformer Encoder.
    """
    def __init__(self, lookback, num_features, num_assets, patch_length=5, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super(PatchTSTSignalModel, self).__init__()
        
        self.patch_length = patch_length
        self.num_patches = lookback // patch_length
        
        # Linear projection for each patch
        self.patch_embedding = nn.Linear(patch_length * num_features, d_model)
        
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.num_patches * d_model, num_assets)
        
    def forward(self, x):
        """
        x: [batch, lookback, num_features]
        """
        batch_size = x.shape[0]
        
        # For simplicity, truncate sequence if not perfectly divisible
        valid_len = self.num_patches * self.patch_length
        x = x[:, -valid_len:, :]
        
        # Reshape to patches: [batch, num_patches, patch_length * num_features]
        patches = x.reshape(batch_size, self.num_patches, self.patch_length * x.shape[-1])
        
        # Embed
        emb = self.patch_embedding(patches) + self.position_embedding
        
        # Transform
        out = self.transformer_encoder(emb)
        
        # Flatten and project
        out = self.flatten(out)
        logits = self.fc(out)
        
        return torch.tanh(logits)
