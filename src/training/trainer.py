import torch
import torch.nn as nn
import torch.optim as optim
import time

class BaseTrainer:
    """
    Generic PyTorch trainer capable of training any model implementing TradingSignalModel interface.
    """
    def __init__(self, model: nn.Module, criterion, optimizer, config: dict):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, train_loader, val_loader=None):
        epochs = self.config.get('epochs', 10)
        patience = self.config.get('early_stopping_patience', 5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        history = {'train_loss': [], 'val_loss': []}
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.model.train()
            train_loss_accum = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                # Assume criterion is NegativeSharpeLoss or similar
                # and batch_y represents the next period returns
                # We need the portfolio returns at t+1 based on positions at t.
                # For simplicity in this generic trainer, we'll assume the criterion
                # accepts (positions, next_period_returns). 
                # Normally, you'd apply volatility scaling inside the loss or before it.
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
                train_loss_accum += loss.item()
                
            avg_train_loss = train_loss_accum / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            if val_loader:
                self.model.eval()
                val_loss_accum = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_x)
                        val_loss = self.criterion(outputs, batch_y)
                        val_loss_accum += val_loss.item()
                        
                avg_val_loss = val_loss_accum / len(val_loader)
                history['val_loss'].append(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            
        end_time = time.time()
        return history, (end_time - start_time)
