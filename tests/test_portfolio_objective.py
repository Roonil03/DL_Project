import numpy as np
import torch
from src.backtest.backtester import Backtester
from src.losses.sharpe_loss import NegativeSharpeLoss


def test_sharpe_loss_matches_gross_backtest_with_sizing_and_cap():
    signals = torch.tensor([[.5, -.8], [.9, .4], [-.2, .6]], requires_grad=True)
    returns = torch.tensor([[.01, -.02], [-.01, .03], [.02, .01]])
    vol = torch.tensor([[.05, .2], [.01, .3], [.1, .05]])
    loss = NegativeSharpeLoss(target_vol=.12, max_leverage=2.0)(signals, returns, vol)
    actual = Backtester(target_vol=.12, max_leverage=2.0, tc_bps=0).run(
        signals.detach().numpy(), vol.numpy(), returns.numpy())['gross_returns'].to_numpy()
    expected = -np.sqrt(252) * np.mean(actual) / np.sqrt(np.var(actual, ddof=1) + 1e-6)
    assert np.isclose(loss.item(), expected, rtol=1e-5)
    loss.backward()
    assert torch.isfinite(signals.grad).all()


def test_bounded_candidates_and_validation_only_selection():
    import copy
    from src.training.tune import bounded_model_candidates, select_validation_candidate
    models = {'dlinear': {'trend_window': 20}, 'lstm': {'hidden_dim': 64, 'layers': 2, 'dropout': .1},
              'mamba': {'d_model': 64, 'layers': 2}, 'patchtst': {'d_model': 64, 'heads': 4, 'layers': 2, 'patch_length': 5},
              'ahat_fin': {'d_model': 64, 'heads': 4, 'layers': 2}}
    before = copy.deepcopy(models)
    for name in models:
        candidates = bounded_model_candidates(name, models, [.001, .0003], 2)
        assert len(candidates) == 2
        base, variant = (candidate['models'][name] for candidate in candidates)
        assert base != variant
        fields = {'dlinear': ('trend_window',), 'lstm': ('hidden_dim', 'layers', 'dropout'),
                  'mamba': ('d_model', 'layers'), 'patchtst': ('d_model', 'heads', 'layers', 'patch_length'),
                  'ahat_fin': ('d_model', 'heads', 'layers')}
        assert all(base[field] != variant[field] for field in fields[name])
        if 'heads' in variant: assert variant['d_model'] % variant['heads'] == 0
    assert models == before
    trials = [{'negative_validation_sharpe': -.2, 'test_score': 100},
              {'negative_validation_sharpe': -.5, 'test_score': -100}]
    assert select_validation_candidate(trials) is trials[1]


def test_dlinear_uses_only_linear_projections_and_bounded_output():
    from src.models.dlinear import DLinear
    model = DLinear(lookback=20, num_features=4, num_assets=2)
    assert all(isinstance(layer, torch.nn.Linear) for layer in
               (model.linear_seasonal, model.linear_trend, model.proj))
    output = model(torch.randn(3, 20, 4))
    assert output.shape == (3, 2)
    assert (output.abs() <= 1).all()


def test_bootstrap_rejects_unpaired_data_and_marks_short_interval_unavailable():
    import pandas as pd
    import pytest
    from src.evaluation.statistical_tests import block_bootstrap_sharpe, paired_sharpe_difference_test
    with pytest.raises(ValueError, match='equal lengths'):
        paired_sharpe_difference_test([.01, .02], [.01])
    with pytest.raises(ValueError, match='indices'):
        paired_sharpe_difference_test(pd.Series([.01, .02], index=[1, 2]), pd.Series([.01, .02], index=[2, 3]))
    assert all(np.isnan(x) for x in block_bootstrap_sharpe(np.ones(5), block_size=4))


def test_notebook_checkpoint_fallback_is_isolated_by_fold(tmp_path):
    import ast
    import json
    import os
    import time
    from pathlib import Path
    notebook = json.loads(Path('notebooks/00_core.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if 'class NotebookUnitRun:' in ''.join(cell.get('source', [])))
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NotebookUnitRun')
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {'Path': Path, 'time': time, 'os': os, 'json': json}
    exec(compile(module, '<notebook>', 'exec'), namespace)
    run_class = namespace['NotebookUnitRun']
    first = run_class(tmp_path, 'run', 'lstm', 42, 0)
    first.save_checkpoint({'epoch': 1})
    first.save_checkpoint({'epoch': 2})
    (first.checkpoints / 'latest.pt').write_bytes(b'corrupt')
    resumed = run_class(tmp_path, 'run', 'lstm', 42, 0)
    assert resumed.load_checkpoint()['epoch'] == 1
    other = run_class(tmp_path, 'run', 'lstm', 42, 1)
    assert other.load_checkpoint() is None
    assert first.checkpoints != other.checkpoints


def test_market_inputs_reject_missing_and_unverified_mixed_market():
    import pandas as pd
    import pytest
    from src.data.adapters import validate_global_market_inputs
    validate_global_market_inputs(pd.DataFrame({'asset': [1., 2.]}))
    with pytest.raises(ValueError, match='missing prices'):
        validate_global_market_inputs(pd.DataFrame({'asset': [1., np.nan]}))
    with pytest.raises(ValueError, match='Mixed-market'):
        validate_global_market_inputs(pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]}))


def test_regime_threshold_is_fixed_to_training():
    import pandas as pd
    from src.regimes.rules import detect_regimes
    rng = np.random.default_rng(8)
    changes = rng.normal(.0003, .01, 180)
    altered = changes.copy()
    altered[100:140] *= 15
    dates = pd.date_range('2020-01-01', periods=180)
    prices_a = pd.Series(100 * np.cumprod(1 + changes), index=dates)
    prices_b = pd.Series(100 * np.cumprod(1 + altered), index=dates)
    labels_a = detect_regimes(prices_a, vol_window=10, trend_window=15, train_end=dates[79])
    labels_b = detect_regimes(prices_b, vol_window=10, trend_window=15, train_end=dates[79])
    assert labels_a.iloc[-20:].equals(labels_b.iloc[-20:])


def test_baselines_have_distinct_causal_positions_and_returns():
    from src.portfolio.baselines import (get_zero_signal_baseline,
        get_long_only_baseline, get_long_only_trend_baseline, get_momentum_baseline)
    history = np.array([[.01, -.02], [.02, -.01], [-.04, .03],
                        [-.02, .01], [.03, -.02], [.02, -.01]])
    lookback = 2
    selection = slice(lookback, None)
    vol = np.array([[.05, .2], [.2, .05], [.08, .25], [.25, .08]])
    targets = np.array([[.01, .04], [-.02, .01], [.03, -.02], [-.01, .05]])
    signals = [vol / .1, get_zero_signal_baseline(targets),
               get_long_only_trend_baseline(history, lookback)[selection],
               get_momentum_baseline(history, lookback)[selection],
               get_long_only_baseline(targets)]
    bt = Backtester(target_vol=.1, max_leverage=3., tc_bps=0.)
    results = [bt.run(signal, vol, targets) for signal in signals]
    for i, result in enumerate(results):
        for other in results[i + 1:]:
            assert not np.allclose(result['positions'], other['positions'])
            assert not np.allclose(result['gross_returns'], other['gross_returns'])
    assert np.allclose(results[0]['positions'], 1.)
    assert np.all(results[2]['positions'].to_numpy() >= 0)
    assert np.any(results[3]['positions'].to_numpy() < 0)
    changed_future = history.copy()
    changed_future[4:] *= -100
    np.testing.assert_array_equal(
        get_long_only_trend_baseline(history, lookback)[:5],
        get_long_only_trend_baseline(changed_future, lookback)[:5])
