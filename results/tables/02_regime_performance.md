# Regime-Stratified Sharpe Ratios

|          |   HighVol_Bear |   HighVol_Bull |   LowVol_Bear |   LowVol_Bull |
|:---------|---------------:|---------------:|--------------:|--------------:|
| DLinear  |        5.21315 |       -1.19144 |     -0.307201 |      0.956848 |
| LSTM     |        7.9732  |        2.18874 |      0.243866 |      1.89708  |
| Mamba    |        4.86025 |        2.31533 |      2.48947  |      1.74433  |
| PatchTST |        3.78476 |        1.32212 |      0.236087 |      0.391479 |

# Regime Robustness & Stress Degradation

| Model    |   robustness_score |   stress_degradation |   min_regime_sharpe |   max_regime_sharpe |
|:---------|-------------------:|---------------------:|--------------------:|--------------------:|
| DLinear  |         -0.228545  |             -4.25631 |           -1.19144  |             5.21315 |
| LSTM     |          0.0305857 |             -6.07612 |            0.243866 |             7.9732  |
| Mamba    |          0.358898  |             -3.11592 |            1.74433  |             4.86025 |
| PatchTST |          0.0623784 |             -3.39328 |            0.236087 |             3.78476 |
