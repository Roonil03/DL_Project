# Paired Bootstrap Hypothesis Tests (Sharpe Ratio Differences)

| Comparison          |   Sharpe (A) |   Sharpe (B) |   Difference (A - B) |   Bootstrap p-value | Significant at 5%   |
|:--------------------|-------------:|-------------:|---------------------:|--------------------:|:--------------------|
| DLinear vs LSTM     |        1.127 |        2.215 |               -1.089 |               0.941 | No                  |
| DLinear vs Mamba    |        1.127 |        0.655 |                0.472 |               0.189 | No                  |
| DLinear vs PatchTST |        1.127 |       -0.287 |                1.413 |               0.011 | Yes                 |
| LSTM vs Mamba       |        2.215 |        0.655 |                1.56  |               0.012 | Yes                 |
| LSTM vs PatchTST    |        2.215 |       -0.287 |                2.502 |               0.001 | Yes                 |
| Mamba vs PatchTST   |        0.655 |       -0.287 |                0.942 |               0.059 | No                  |
