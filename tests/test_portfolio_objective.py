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
    models = {'dlinear': {'trend_window': 20}, 'lstm': {'hidden_dim': 64},
              'mamba': {'d_model': 64}, 'patchtst': {'d_model': 64, 'heads': 4},
              'ahat_fin': {'d_model': 64, 'heads': 4}}
    before = copy.deepcopy(models)
    for name in models:
        candidates = bounded_model_candidates(name, models, [.001, .0003], 2)
        assert len(candidates) == 2
        assert candidates[0]['models'][name] != candidates[1]['models'][name]
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
