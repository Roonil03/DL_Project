import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import numpy as np
from typing import Any, Callable, Dict, Optional

class BaseTrainer:
    """
    Generic PyTorch trainer for financial time-series models under Sharpe-loss optimization.
    """
    def __init__(self, model: nn.Module, criterion, optimizer, config: dict, scheduler=None,
                 run=None, telemetry_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        requested_device = config.get('device', 'auto')
        if requested_device == 'auto':
            requested_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(requested_device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available')
        self.model.to(self.device)
        self.run = run
        self.telemetry_callback = telemetry_callback
        
    def train(self, train_loader, val_loader=None, resume: bool = False):
        epochs = self.config.get('epochs', 50)
        patience = self.config.get('early_stopping_patience', 10)
        grad_clip = self.config.get('grad_clip', 1.0)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        history = {'train_loss': [], 'val_loss': [], 'epoch_seconds': [], 'skipped_batches': []}
        start_epoch = 0
        if resume and self.run is not None:
            try:
                checkpoint = self.run.load_checkpoint(map_location=str(self.device))
            except Exception as exc:
                self.run.event('checkpoint_rejected', error=repr(exc))
                raise RuntimeError('Checkpoint is corrupt or incompatible; refusing a silent restart') from exc
            if checkpoint is not None:
                self.model.load_state_dict(checkpoint['model_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                if self.scheduler is not None and checkpoint.get('scheduler_state') is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state'])
                history.update(checkpoint.get('history', {}))
                best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
                patience_counter = int(checkpoint.get('early_stopping_counter', 0))
                start_epoch = int(checkpoint.get('epoch', -1)) + 1
                if checkpoint.get('cpu_rng_state') is not None:
                    torch.set_rng_state(checkpoint['cpu_rng_state'])
                if torch.cuda.is_available() and checkpoint.get('cuda_rng_state') is not None:
                    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
                if checkpoint.get('numpy_rng_state') is not None:
                    np.random.set_state(checkpoint['numpy_rng_state'])
        
        start_time = time.time()
        
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            self.model.train()
            train_loss_accum = 0.0
            num_train_batches = 0
            skipped_batches = 0
            gradient_norms = []
            samples_seen = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                samples_seen += int(batch_x.shape[0])
                
                self.optimizer.zero_grad()
                signals = self.model(batch_x)
                
                loss = self.criterion(signals, batch_y)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    if grad_clip > 0:
                        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                        gradient_norms.append(float(norm.detach().cpu()))
                    self.optimizer.step()
                    train_loss_accum += loss.item()
                    num_train_batches += 1
                else:
                    skipped_batches += 1
                
            avg_train_loss = train_loss_accum / max(1, num_train_batches)
            history['train_loss'].append(avg_train_loss)
            
            if val_loader is not None:
                self.model.eval()
                val_loss_accum = 0.0
                num_val_batches = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        signals = self.model(batch_x)
                        val_loss = self.criterion(signals, batch_y)
                        if not torch.isnan(val_loss) and not torch.isinf(val_loss):
                            val_loss_accum += val_loss.item()
                            num_val_batches += 1
                            
                avg_val_loss = val_loss_accum / max(1, num_val_batches)
                history['val_loss'].append(avg_val_loss)
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                    
            epoch_seconds = time.time() - epoch_start
            history['epoch_seconds'].append(epoch_seconds)
            history['skipped_batches'].append(skipped_batches)
            telemetry = {
                'epoch': epoch + 1, 'train_loss': avg_train_loss,
                'val_loss': history['val_loss'][-1] if history['val_loss'] else None,
                'epoch_seconds': epoch_seconds, 'skipped_batches': skipped_batches,
                'device': str(self.device),
                'learning_rate': float(self.optimizer.param_groups[0]['lr']),
                'gradient_norm': float(np.mean(gradient_norms)) if gradient_norms else None,
                'samples': samples_seen,
                'throughput_samples_per_second': samples_seen / max(epoch_seconds, 1e-12),
                'gpu_memory_allocated_bytes': int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == 'cuda' else 0,
            }
            if self.device.type == 'cuda':
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index or 0)
                    telemetry['gpu_utilization_percent'] = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    telemetry['gpu_memory_used_bytes'] = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                except Exception:
                    telemetry['gpu_utilization_percent'] = None
                    telemetry['gpu_memory_used_bytes'] = None
            if self.run is not None:
                self.run.event('epoch_complete', **telemetry)
                self.run.save_state({'epoch': epoch, 'history': history, 'best_val_loss': best_val_loss})
                self.run.save_checkpoint({
                    'epoch': epoch, 'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'history': history, 'loss_history': history,
                    'best_val_loss': best_val_loss,
                    'early_stopping_counter': patience_counter,
                    'cpu_rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    'numpy_rng_state': np.random.get_state(),
                    'metadata': getattr(self.run, 'checkpoint_metadata', {}),
                })
            if self.telemetry_callback is not None:
                self.telemetry_callback(telemetry)
            if val_loader is not None and patience_counter >= patience:
                break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        elapsed_time = time.time() - start_time
        return history, elapsed_time

    def predict(self, data_loader) -> torch.Tensor:
        self.model.eval()
        all_signals = []
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    batch_x = batch[0]
                else:
                    batch_x = batch
                batch_x = batch_x.to(self.device)
                signals = self.model(batch_x)
                all_signals.append(signals.cpu())
        if len(all_signals) > 0:
            return torch.cat(all_signals, dim=0).numpy()
        return np.array([])

