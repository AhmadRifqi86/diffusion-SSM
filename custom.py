import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# --- Lion Optimizer Implementation ---
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable, Tuple

class Lion(Optimizer):
    """
    Lion optimizer (EvoLved Sign Momentum)
    Reference: https://arxiv.org/abs/2302.06675
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                # Weight decay (decoupled)
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['weight_decay'])
                # Update
                update = exp_avg.lerp(grad, 1 - beta1).sign_()
                p.data.add_(update, alpha=-group['lr'])
                # Momentum update
                exp_avg.lerp_(grad, 1 - beta2)
        return loss

#Consider using SequentialLR to chain multiple schedulers
class CosineAnnealingWarmRestartsWithDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, decay=0.9, freq_mult=0.9, last_epoch=-1, warmup_epoch=None):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay = decay          # Decay factor for max LR
        self.freq_mult = freq_mult  # Multiplier for cycle length (e.g., 0.9 for shorter cycles)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_max_lrs = self.base_lrs.copy()
        self.T_i = T_0
        self.cycle = 0
        self.epoch_since_restart = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Standard cosine annealing formula, but with decaying max LR
        return [
            self.eta_min + (max_lr - self.eta_min) * (1 + torch.cos(torch.tensor(self.epoch_since_restart * 3.1415926535 / self.T_i))) / 2
            for max_lr in self.current_max_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.epoch_since_restart += 1

        if self.epoch_since_restart >= self.T_i:
            # End of cycle: decay max LR and decrease cycle length
            self.cycle += 1
            self.epoch_since_restart = 0
            self.current_max_lrs = [lr * self.decay for lr in self.current_max_lrs]
            self.T_i = max(1, int(self.T_i * self.freq_mult))  # Prevent T_i from going below 1

        # Update the learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    def __init__(self, patience=20, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def restore_best_model(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            return True
        return False

class PhaseScheduler:
    """Learning rate scheduler with different phases"""
    def __init__(self, optimizer, total_epochs, warmup_ratio=0.2, decay_ratio=0.2):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_ratio)
        self.decay_epochs = int(total_epochs * decay_ratio)
        self.main_epochs = total_epochs - self.warmup_epochs - self.decay_epochs
        self.current_epoch = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Phase 1: Warm-up (linear increase)
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        elif self.current_epoch < self.warmup_epochs + self.main_epochs:
            # Phase 2: Main training (constant)
            lr = self.base_lr
        else:
            # Phase 3: Fine-tuning (cosine decay)
            decay_epoch = self.current_epoch - (self.warmup_epochs + self.main_epochs)
            progress = decay_epoch / self.decay_epochs
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.current_epoch += 1
        return lr
    
    def state_dict(self):
        """Save scheduler state"""
        return {
            'current_epoch': self.current_epoch,
            'base_lr': self.base_lr,
            'total_epochs': self.total_epochs,
            'warmup_epochs': self.warmup_epochs,
            'decay_epochs': self.decay_epochs,
            'main_epochs': self.main_epochs
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state"""
        self.current_epoch = state_dict['current_epoch']
        self.base_lr = state_dict['base_lr']
        self.total_epochs = state_dict['total_epochs']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.decay_epochs = state_dict['decay_epochs']
        self.main_epochs = state_dict['main_epochs']

class LinearScheduler:
    """Linear learning rate scheduler with optional warmup"""
    def __init__(self, optimizer, total_epochs, warmup_epochs=0, end_lr_factor=0.1):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.end_lr_factor = end_lr_factor
        self.current_epoch = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase (linear increase)
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Linear decay phase
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * (1 - progress * (1 - self.end_lr_factor))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.current_epoch += 1
        return lr
    
    def state_dict(self):
        """Save scheduler state"""
        return {
            'current_epoch': self.current_epoch,
            'base_lr': self.base_lr,
            'total_epochs': self.total_epochs,
            'warmup_epochs': self.warmup_epochs,
            'end_lr_factor': self.end_lr_factor
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state"""
        self.current_epoch = state_dict['current_epoch']
        self.base_lr = state_dict['base_lr']
        self.total_epochs = state_dict['total_epochs']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.end_lr_factor = state_dict['end_lr_factor']

# Helper function to create Lion optimizer


# Shifted step-exponential