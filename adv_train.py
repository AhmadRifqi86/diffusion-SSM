import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import math

class MultiComponentTrainingStrategy:
    """
    Optimal training strategy for UShapeMamba diffusion model with different
    learning rates and schedules for different components.
    """
    
    def __init__(self, model, base_lr=1e-4, warmup_steps=1000, total_steps=100000):
        self.model = model
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        # Separate parameter groups with different learning rates
        self.param_groups = self._create_parameter_groups()
        
        # Create optimizer with parameter groups
        self.optimizer = AdamW(
            self.param_groups,
            weight_decay=0.01,  # L2 regularization
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Create learning rate schedulers for each group
        self.schedulers = self._create_schedulers()
    
    def _create_parameter_groups(self):
        """Create parameter groups with component-specific learning rates"""
        
        # 1. UNet backbone (most parameters) - standard learning rate
        unet_params = []
        for name, param in self.model.unet.named_parameters():
            if param.requires_grad:
                unet_params.append(param)
        
        # 2. Cross-attention layers - higher learning rate (they're doing the heavy lifting)
        cross_attn_params = []
        for name, param in self.model.unet.named_parameters():
            if param.requires_grad and 'cross_attn' in name:
                cross_attn_params.append(param)
        
        # 3. Mamba blocks - moderate learning rate
        mamba_params = []
        for name, param in self.model.unet.named_parameters():
            if param.requires_grad and 'mamba_block' in name:
                mamba_params.append(param)
        
        # 4. Time embedding and conditioning - higher learning rate
        conditioning_params = []
        for name, param in self.model.unet.named_parameters():
            if param.requires_grad and any(x in name for x in ['time_embed', 'scale_shift', 'context_proj']):
                conditioning_params.append(param)
        
        # 5. Input/Output projections - lower learning rate (closer to pre-trained components)
        projection_params = []
        for name, param in self.model.unet.named_parameters():
            if param.requires_grad and any(x in name for x in ['input_proj', 'output_proj']):
                projection_params.append(param)
        
        # Remove duplicates by creating sets of parameter ids
        cross_attn_ids = {id(p) for p in cross_attn_params}
        mamba_ids = {id(p) for p in mamba_params}
        conditioning_ids = {id(p) for p in conditioning_params}
        projection_ids = {id(p) for p in projection_params}
        
        # Filter main unet params to exclude specialized groups
        specialized_ids = cross_attn_ids | mamba_ids | conditioning_ids | projection_ids
        unet_params = [p for p in unet_params if id(p) not in specialized_ids]
        
        param_groups = [
            {
                'params': unet_params,
                'lr': self.base_lr,
                'name': 'unet_backbone'
            },
            {
                'params': cross_attn_params,
                'lr': self.base_lr * 1.5,  # Higher for cross-attention
                'name': 'cross_attention'
            },
            {
                'params': mamba_params,
                'lr': self.base_lr * 1.2,  # Moderate for Mamba
                'name': 'mamba_blocks'
            },
            {
                'params': conditioning_params,
                'lr': self.base_lr * 2.0,  # Higher for conditioning
                'name': 'conditioning'
            },
            {
                'params': projection_params,
                'lr': self.base_lr * 0.5,  # Lower for projections
                'name': 'projections'
            }
        ]
        
        return param_groups
    
    def _create_schedulers(self):
        """Create different schedulers for different parameter groups"""
        schedulers = []
        
        for i, group in enumerate(self.param_groups):
            group_name = group['name']
            initial_lr = group['lr']
            
            if group_name == 'conditioning':
                # Aggressive schedule for conditioning - needs to learn fast
                warmup = LinearLR(
                    self.optimizer, 
                    start_factor=0.01, 
                    end_factor=1.0, 
                    total_iters=self.warmup_steps
                )
                cosine = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.total_steps - self.warmup_steps,
                    eta_min=initial_lr * 0.01
                )
                scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[self.warmup_steps]
                )
            
            elif group_name == 'cross_attention':
                # Moderate warmup with cosine decay
                warmup = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=self.warmup_steps
                )
                cosine = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.total_steps - self.warmup_steps,
                    eta_min=initial_lr * 0.05
                )
                scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[self.warmup_steps]
                )
            
            else:
                # Conservative schedule for backbone, mamba, and projections
                warmup = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=self.warmup_steps * 2  # Longer warmup
                )
                cosine = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.total_steps - self.warmup_steps * 2,
                    eta_min=initial_lr * 0.1  # Higher minimum
                )
                scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[self.warmup_steps * 2]
                )
            
            schedulers.append(scheduler)
        
        return schedulers
    
    def step(self):
        """Step all schedulers"""
        for scheduler in self.schedulers:
            scheduler.step()
    
    def get_current_lrs(self):
        """Get current learning rates for monitoring"""
        return {group['name']: group['lr'] for group in self.optimizer.param_groups}


class ImprovedCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Improved version of your cosine scheduler with better warmup and decay
    """
    def __init__(self, optimizer, warmup_steps=1000, T_0=10000, T_mult=1, 
                 eta_min_ratio=0.01, decay_factor=0.95, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min_ratio = eta_min_ratio
        self.decay_factor = decay_factor
        
        self.T_i = T_0
        self.cycle = 0
        self.step_in_cycle = 0
        self.base_lrs = None
        self.current_max_lrs = None
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.base_lrs is None:
            self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.current_max_lrs = self.base_lrs.copy()
        
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Cosine annealing phase
        adjusted_epoch = self.last_epoch - self.warmup_steps
        
        if adjusted_epoch >= self.T_i:
            # Start new cycle
            self.cycle += 1
            self.T_i = int(self.T_i * self.T_mult)
            self.step_in_cycle = 0
            # Decay maximum learning rate
            self.current_max_lrs = [
                max_lr * self.decay_factor for max_lr in self.current_max_lrs
            ]
        else:
            self.step_in_cycle = adjusted_epoch % self.T_i
        
        # Cosine annealing formula
        cos_factor = 0.5 * (1 + math.cos(math.pi * self.step_in_cycle / self.T_i))
        
        return [
            eta_min + cos_factor * (max_lr - eta_min)
            for max_lr, eta_min in zip(
                self.current_max_lrs,
                [base_lr * self.eta_min_ratio for base_lr in self.base_lrs]
            )
        ]


# Usage example
def setup_training(model):
    """
    Recommended training setup for UShapeMamba diffusion model
    """
    
    # Option 1: Multi-component strategy (RECOMMENDED)
    training_strategy = MultiComponentTrainingStrategy(
        model=model,
        base_lr=1e-4,
        warmup_steps=2000,
        total_steps=200000
    )
    
    return training_strategy.optimizer, training_strategy.schedulers
    
    # Option 2: Single optimizer with improved scheduler
    # optimizer = AdamW(
    #     model.parameters(),
    #     lr=1e-4,
    #     weight_decay=0.01,
    #     betas=(0.9, 0.999)
    # )
    # 
    # scheduler = ImprovedCosineScheduler(
    #     optimizer,
    #     warmup_steps=2000,
    #     T_0=20000,
    #     eta_min_ratio=0.01,
    #     decay_factor=0.9
    # )
    # 
    # return optimizer, scheduler


# Training recommendations summary
TRAINING_RECOMMENDATIONS = {
    "initial_learning_rates": {
        "unet_backbone": 1e-4,
        "cross_attention": 1.5e-4,  # Higher - these learn text-image relationships
        "mamba_blocks": 1.2e-4,     # Moderate - sequential processing
        "conditioning": 2e-4,       # Highest - time/text conditioning
        "projections": 5e-5         # Lower - interface with pretrained components
    },
    
    "optimizer": "AdamW with weight_decay=0.01, betas=(0.9, 0.999)",
    
    "scheduler_strategy": {
        "conditioning": "Fast warmup (1k steps) + aggressive cosine decay",
        "cross_attention": "Medium warmup (1k steps) + moderate cosine decay", 
        "others": "Slow warmup (2k steps) + conservative cosine decay"
    },
    
    "key_principles": [
        "Use different learning rates for different components",
        "Conditioning layers need highest LR (they start from scratch)",
        "Cross-attention needs higher LR (learning alignment)",
        "Projection layers need lower LR (interfacing with frozen components)",
        "Use longer warmup for stable components",
        "Monitor loss per component separately if possible"
    ]
}

print("Training Strategy Recommendations:")
for key, value in TRAINING_RECOMMENDATIONS.items():
    print(f"\n{key.upper()}:")
    if isinstance(value, dict):
        for k, v in value.items():
            print(f"  {k}: {v}")
    elif isinstance(value, list):
        for item in value:
            print(f"  - {item}")
    else:
        print(f"  {value}")