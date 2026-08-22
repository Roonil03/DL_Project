# Regime-Stratified Sharpe Ratios

|          |   HighVol_Bear |   HighVol_Bull |   LowVol_Bear |   LowVol_Bull |
|:---------|---------------:|---------------:|--------------:|--------------:|
| DLinear  |      0.334849  |      -0.148322 |      0.285238 |      1.44244  |
| LSTM     |      8.26303   |       2.14037  |     -1.39633  |      2.45066  |
| Mamba    |     -0.0731989 |       1.44003  |      1.60652  |      0.503173 |
| PatchTST |    -13.7466    |       1.76957  |     -0.76256  |     -0.482361 |

# Regime Robustness & Stress Degradation

| Model    |   robustness_score |   stress_degradation |   min_regime_sharpe |   max_regime_sharpe |
|:---------|-------------------:|---------------------:|--------------------:|--------------------:|
| DLinear  |         -0.102827  |             1.10759  |          -0.148322  |             1.44244 |
| LSTM     |         -0.168985  |            -5.81237  |          -1.39633   |             8.26303 |
| Mamba    |         -0.0455635 |             0.576372 |          -0.0731989 |             1.60652 |
| PatchTST |         -7.76835   |            13.2643   |         -13.7466    |             1.76957 |
