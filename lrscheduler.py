import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CosineAnnealingWarmRestartsWithDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, decay=0.9, freq_mult=0.9, last_epoch=-1):
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

# Shifted step-exponential