import json

import numpy as np
import pandas as pd

from src.data.walk_forward import walk_forward_splits
from src.evaluation.analysis import architecture_rank_stability, predictive_metrics
from src.evaluation.statistical_tests import paired_block_bootstrap_indices, paired_sharpe_difference_test
from src.runtime.experiment import ExperimentRun, atomic_json_dump, estimate_runtime


def test_atomic_json_and_run_layout(tmp_path):
    output = atomic_json_dump({'value': np.int64(2)}, tmp_path / 'nested' / 'state.json')
    assert json.loads(output.read_text()) == {'value': 2}
    run = ExperimentRun(root=tmp_path, name='unit', run_id='fixed')
    run.save_config({'epochs': 1})
    run.event('started', seed=42)
    assert (run.path / 'system.json').exists()
    assert (run.path / 'config.json').exists()
    assert 'started' in (run.path / 'events.jsonl').read_text()


def test_runtime_estimate_and_walk_forward_are_chronological():
    assert estimate_runtime(epochs=2, batches_per_epoch=3, seconds_per_batch=2)['seconds'] == 12
    folds = list(walk_forward_splits(20, initial_train_size=8, test_size=3, step_size=3, gap=1))
    assert len(folds) == 3
    assert all(train.max() < test.min() for train, test in folds)
    assert folds[1][0].min() == 0


def test_corrupt_latest_checkpoint_falls_back_to_previous(tmp_path):
    run = ExperimentRun(root=tmp_path, name='unit', run_id='checkpoint')
    run.save_checkpoint({'epoch': 1})
    run.save_checkpoint({'epoch': 2})
    run.checkpoint_path.write_bytes(b'not a torch checkpoint')
    assert run.load_checkpoint()['epoch'] == 1


def test_predictive_rank_and_paired_blocks():
    metrics = predictive_metrics([1, -1, 2], [1, -2, 0])
    assert metrics['observations'] == 3 and metrics['directional_accuracy'] == 2 / 3
    stability = architecture_rank_stability(pd.DataFrame({'a': [2., 1.], 'b': [1., 2.]}))
    assert stability['mean_rank']['a'] == stability['mean_rank']['b'] == 1.5
    index = paired_block_bootstrap_indices(10, 3, np.random.default_rng(2))
    assert len(index) == 10 and np.all((0 <= index) & (index < 10))
    result = paired_sharpe_difference_test(np.linspace(.001, .01, 30), np.linspace(0, .005, 30), num_bootstraps=20, block_size=4)
    assert result['observations'] == 30 and 0 <= result['p_value'] <= 1
