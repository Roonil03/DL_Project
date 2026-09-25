import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import numpy as np

class BaseTrainer:
    """
    Generic PyTorch trainer for financial time-series models under Sharpe-loss optimization.
    """
    def __init__(self, model: nn.Module, criterion, optimizer, config: dict, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, train_loader, val_loader=None):
        epochs = self.config.get('epochs', 50)
        patience = self.config.get('early_stopping_patience', 10)
        grad_clip = self.config.get('grad_clip', 1.0)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        history = {'train_loss': [], 'val_loss': []}
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.model.train()
            train_loss_accum = 0.0
            num_train_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                signals = self.model(batch_x)
                
                loss = self.criterion(signals, batch_y)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()
                    train_loss_accum += loss.item()
                    num_train_batches += 1
                
            avg_train_loss = train_loss_accum / max(1, num_train_batches)
            history['train_loss'].append(avg_train_loss)
            
            if val_loader:
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
                    
                if patience_counter >= patience:
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

