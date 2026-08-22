import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from src.models.dlinear import DLinear
from src.models.lstm import LSTMSignalModel
from src.models.patchtst import PatchTSTSignalModel
from src.models.mamba_model import MambaSignalModel

def test_model_shapes():
    batch_size = 32
    lookback = 20
    num_assets = 10
    num_features = num_assets # Simplified for this test
    
    x = torch.randn(batch_size, lookback, num_features)
    
    # 1. DLinear
    model_dlinear = DLinear(lookback=lookback, num_assets=num_assets)
    out_dlinear = model_dlinear(x)
    assert out_dlinear.shape == (batch_size, num_assets), f"DLinear shape error: {out_dlinear.shape}"
    
    # 2. LSTM
    model_lstm = LSTMSignalModel(lookback=lookback, num_features=num_features, num_assets=num_assets)
    out_lstm = model_lstm(x)
    assert out_lstm.shape == (batch_size, num_assets), f"LSTM shape error: {out_lstm.shape}"
    
    # 3. PatchTST
    # Note: PatchTST expects lookback divisible by patch_length
    model_patch = PatchTSTSignalModel(lookback=lookback, num_features=num_features, num_assets=num_assets, patch_length=5)
    out_patch = model_patch(x)
    assert out_patch.shape == (batch_size, num_assets), f"PatchTST shape error: {out_patch.shape}"
    
    # 4. Mamba (or fallback)
    model_mamba = MambaSignalModel(lookback=lookback, num_features=num_features, num_assets=num_assets)
    out_mamba = model_mamba(x)
    assert out_mamba.shape == (batch_size, num_assets), f"Mamba shape error: {out_mamba.shape}"
    
    print("All model shapes validated successfully.")

if __name__ == "__main__":
    test_model_shapes()
