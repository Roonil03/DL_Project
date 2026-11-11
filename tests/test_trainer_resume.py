import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.runtime.experiment import ExperimentRun
from src.training.trainer import BaseTrainer


def _loader(x, y, batch_size=2):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _trainer(run=None, epochs=1):
    torch.manual_seed(7)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    return BaseTrainer(
        model,
        nn.MSELoss(),
        optimizer,
        {'epochs': epochs, 'early_stopping_patience': 10, 'device': 'cpu'},
        run=run,
    )


def test_resume_continues_epoch_and_restores_best_model(tmp_path):
    x = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    y = 2 * x
    run = ExperimentRun(root=tmp_path, run_id='resume', checkpoint_metadata={'fold': 2})
    first = _trainer(run=run, epochs=1)
    first_history, _ = first.train(_loader(x, y), _loader(x, y))
    checkpoint = run.load_checkpoint()
    assert len(first_history['train_loss']) == 1
    assert checkpoint['best_model_state'] is not None

    resumed = _trainer(run=run, epochs=2)
    history, _ = resumed.train(_loader(x, y), _loader(x, y), resume=True)
    assert len(history['train_loss']) == 2
    assert run.load_checkpoint()['epoch'] == 1
    for name, value in resumed.model.state_dict().items():
        assert torch.equal(value, run.load_checkpoint()['best_model_state'][name])


def test_nonfinite_batches_and_gradients_are_skipped_without_updates():
    class NonFiniteBackward(nn.Module):
        def forward(self, prediction, target):
            class InfiniteGradient(torch.autograd.Function):
                @staticmethod
                def forward(ctx, value):
                    return value.sum() * 0

                @staticmethod
                def backward(ctx, grad_output):
                    return torch.full((1, 1), float('inf'))

            return InfiniteGradient.apply(prediction)

    x = torch.tensor([[float('nan')], [1.0]])
    y = torch.tensor([[0.0], [0.0]])
    trainer = _trainer(epochs=1)
    before = copy.deepcopy(trainer.model.state_dict())

    # The NaN input is rejected before forward; the finite input reaches a
    # custom loss whose backward returns an infinite gradient.
    trainer.criterion = NonFiniteBackward()
    history, _ = trainer.train(_loader(x, y, batch_size=1))
    assert history['skipped_batches'] == [2]
    assert torch.isnan(torch.tensor(history['train_loss'][0]))
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, before[name])


def test_resume_rejects_checkpoint_from_another_fold(tmp_path):
    x = torch.ones((2, 1))
    y = torch.zeros((2, 1))
    run = ExperimentRun(root=tmp_path, run_id='fold-check', checkpoint_metadata={'fold': 0})
    _trainer(run=run).train(_loader(x, y))

    other_fold = ExperimentRun(root=tmp_path, run_id='fold-check', resume=True, checkpoint_metadata={'fold': 1})
    with pytest.raises(RuntimeError, match='corrupt or incompatible'):
        _trainer(run=other_fold, epochs=2).train(_loader(x, y), resume=True)
