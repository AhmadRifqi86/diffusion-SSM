import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable
import numpy as np
import math


class MinSNRVLoss(nn.Module):
    """
    🔥 Min-SNR Loss Weighting for V-parameterization
    Optimized for velocity prediction instead of noise prediction
    """
    def __init__(self, gamma=5.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, v_pred, v_target, timesteps, snr):
        """
        Apply Min-SNR weighting to v-parameterization loss
        
        Args:
            v_pred: Predicted velocity from model
            v_target: Target velocity from scheduler
            timesteps: Current timesteps
            snr: Signal-to-noise ratio from scheduler
        """
        # Basic MSE loss between predicted and target velocity
        loss = F.mse_loss(v_pred, v_target, reduction='none')
        loss = loss.mean(dim=[1, 2, 3])  # Average over spatial dimensions
        
        # Min-SNR weighting - prevents over-optimization on high-noise timesteps
        snr_weights = (torch.minimum(snr, torch.full_like(snr, self.gamma)) / snr).detach()
        weighted_loss = loss * snr_weights
        
        return weighted_loss.mean()

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

class CosineAnnealingWarmRestartsWithDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1.0, freq_mult=1.0, eta_min=0, decay=0.9, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay = decay          # Decay factor for max LR
        self.freq_mult = freq_mult  # Multiplier for cycle length (e.g., 0.9 for shorter cycles)
        self.base_lrs = None #[5e-5]#None  # lazy init
        self.current_max_lrs = None #[5e-5]#None
        self.T_i = T_0
        self.cycle = 0
        self.epoch_since_restart = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.base_lrs is None or self.current_max_lrs is None:
            self.base_lrs = [group['initial_lr'] if 'initial_lr' in group else group['lr']
                             for group in self.optimizer.param_groups]
            self.current_max_lrs = self.base_lrs.copy()
            #print("Initialized base_lrs:", self.base_lrs)
        # Standard cosine annealing formula, but with decaying max LR
        return [
            self.eta_min + (max_lr - self.eta_min) * (1 + torch.cos(torch.tensor(self.epoch_since_restart * math.pi/ self.T_i))) / 2
            for max_lr in self.current_max_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.epoch_since_restart += 1
        if self.epoch_since_restart >= self.T_i:
            self.cycle += 1
            self.epoch_since_restart = 0
            self.T_i = int(self.T_i * self.freq_mult) #max(1.0, self.T_i * self.freq_mult) #dipaksa turun berarti cycle nya
            self.current_max_lrs = [
                base_lr * (self.decay ** self.cycle)
                for base_lr in self.base_lrs
            ]
            # print("self.T_i:", self.T_i)
            # print("self.current_max_lrs:", self.current_max_lrs)

        # Apply the new learning rates to param groups
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

        # ✅ Required for PyTorch's SequentialLR compatibility
        self._last_lr = lrs


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


class GradientClipperWithNormTracking:
    """
    Advanced gradient clipping with gradient norm tracking
    """
    def __init__(self, max_norm=1.0, norm_type=2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.grad_norms = []
        
    def clip_gradients(self, model):
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_norm, self.norm_type
        )
        self.grad_norms.append(total_norm.item())
        return total_norm
    
    def get_stats(self):
        if not self.grad_norms:
            return {}
        return {
            'grad_norm_mean': np.mean(self.grad_norms[-100:]),
            'grad_norm_std': np.std(self.grad_norms[-100:]),
            'grad_norm_max': max(self.grad_norms[-100:])
        }

class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}

        # Match device of model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach().to(param.device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Match device just in case
                self.shadow[name] = self.shadow[name].to(param.device)
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.device)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]

    def state_dict(self):
        return {k: v.clone().cpu() for k, v in self.shadow.items()}  # Save on CPU for portability

    def load_state_dict(self, state_dict):
        missing = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in state_dict:
                    self.shadow[name] = state_dict[name].clone().to(param.device)
                else:
                    missing.append(name)
        if missing:
            print(f"[EMAModel] Warning: Missing EMA weights for {len(missing)} parameters.")

