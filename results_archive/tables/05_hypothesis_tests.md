# Paired Bootstrap Hypothesis Tests (Sharpe Ratio Differences)

| Comparison          |   Sharpe (A) |   Sharpe (B) |   Difference (A - B) |   Bootstrap p-value | Significant at 5%   |
|:--------------------|-------------:|-------------:|---------------------:|--------------------:|:--------------------|
| DLinear vs LSTM     |        0.618 |        1.916 |               -1.298 |               0.967 | Yes                 |
| DLinear vs Mamba    |        0.618 |        1.858 |               -1.241 |               0.983 | Yes                 |
| DLinear vs PatchTST |        0.618 |        0.551 |                0.066 |               0.488 | No                  |
| LSTM vs Mamba       |        1.916 |        1.858 |                0.057 |               0.462 | No                  |
| LSTM vs PatchTST    |        1.916 |        0.551 |                1.364 |               0.034 | Yes                 |
| Mamba vs PatchTST   |        1.858 |        0.551 |                1.307 |               0.013 | Yes                 |
