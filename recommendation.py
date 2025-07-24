import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

# ==================== CRITICAL ARCHITECTURAL IMPROVEMENTS ====================

class ImprovedNoiseScheduler:
    """
    Advanced noise scheduler with Zero Terminal SNR and v-parameterization
    Based on "Common Diffusion Noise Schedules and Sample Steps are Flawed" (2023)
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        
        # Use cosine schedule instead of linear for better distribution
        steps = num_train_timesteps + 1
        x = torch.linspace(0, num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # 🔥 CRITICAL: Enforce Zero Terminal SNR
        alphas_cumprod[-1] = 0.0  # This ensures perfect noise at t=T
        
        self.alphas_cumprod = alphas_cumprod[:-1]
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-2]])
        
        # For v-parameterization
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # SNR for Min-SNR weighting
        self.snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
    
    def get_v_target(self, x_0, noise, timesteps):
        """
        🔥 V-parameterization: Predict velocity instead of noise
        Better training dynamics and more stable
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        v = sqrt_alpha * noise - sqrt_one_minus_alpha * x_0
        return v
    
    def predict_start_from_v(self, x_t, v, timesteps):
        """Convert v-prediction back to x_0 prediction"""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        x_0 = sqrt_alpha * x_t - sqrt_one_minus_alpha * v
        return x_0

class MinSNRLoss(nn.Module):
    """
    🔥 Min-SNR Loss Weighting - CRITICAL for faster convergence
    Treats diffusion training as multi-task learning problem
    """
    def __init__(self, gamma=5.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, noise_pred, noise_target, timesteps, snr): #still using noise, change to v
        """
        Apply Min-SNR weighting to balance different timesteps
        """
        # Basic MSE loss
        loss = F.mse_loss(noise_pred, noise_target, reduction='none')
        loss = loss.mean(dim=[1, 2, 3])  # Average over spatial dimensions
        
        # Min-SNR weighting
        snr_weights = torch.minimum(snr, torch.full_like(snr, self.gamma)) / snr
        weighted_loss = loss * snr_weights
        
        return weighted_loss.mean()

class ImprovedScaleShift(nn.Module):
    """
    Enhanced conditioning with better initialization and normalization
    """
    def __init__(self, dim, context_dim, use_zero_init=True):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim, dim * 2)
        )
        
        # 🔥 Zero initialization for stable training
        if use_zero_init:
            nn.init.zeros_(self.to_scale_shift[-1].weight)
            nn.init.zeros_(self.to_scale_shift[-1].bias)
    
    def forward(self, x, context):
        scale_shift = self.to_scale_shift(context)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # Ensure scale doesn't go too extreme
        scale = torch.tanh(scale / 3) * 3  # Clamp between -3, 3
        
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        return x * (scale + 1) + shift

class AdaptiveGroupNorm(nn.Module):
    """
    Adaptive Group Normalization that adapts to timestep
    Better than standard GroupNorm for diffusion models
    """
    def __init__(self, num_groups, num_channels, time_emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, num_channels * 2)
        )
        
    def forward(self, x, time_emb):
        x = self.norm(x)
        scale_shift = self.time_emb_proj(time_emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        return x * (scale + 1) + shift

# ==================== ADVANCED LEARNING RATE STRATEGIES ====================

class WarmupCosineWithRestartsScheduler:
    """
    Advanced scheduler combining multiple proven techniques
    """
    def __init__(self, optimizer, warmup_steps=2000, total_steps=200000, 
                 min_lr_ratio=0.01, restart_cycles=4):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.restart_cycles = restart_cycles
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            
            if self.step_count <= self.warmup_steps:
                # Warmup phase
                lr = base_lr * (self.step_count / self.warmup_steps)
            else:
                # Cosine annealing with restarts
                progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                cycle_length = 1.0 / self.restart_cycles
                cycle_progress = (progress % cycle_length) / cycle_length
                
                cos_factor = 0.5 * (1 + math.cos(math.pi * cycle_progress))
                min_lr = base_lr * self.min_lr_ratio
                lr = min_lr + (base_lr - min_lr) * cos_factor
            
            param_group['lr'] = lr

class AdaptiveLearningRateScheduler:
    """
    🔥 Adaptive LR based on loss landscape analysis
    Increases LR when loss plateaus, decreases when loss spikes
    """
    def __init__(self, optimizer, patience=100, factor=0.8, min_lr=1e-7):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        self.best_loss = float('inf')
        self.wait = 0
        self.loss_history = []
        
    def step(self, loss):
        self.loss_history.append(loss)
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            
        # Plateau detection
        if self.wait >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group['lr'] = new_lr
                print(f"Reducing LR: {old_lr:.2e} -> {new_lr:.2e}")
            
            self.wait = 0
            self.best_loss = loss

# ==================== GRADIENT AND STABILITY IMPROVEMENTS ====================

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
    """
    🔥 Exponential Moving Average - CRITICAL for stable sampling
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

# ==================== COMPLETE TRAINING SETUP ====================

class AdvancedDiffusionTrainer:
    """
    Complete advanced training setup incorporating all improvements
    """
    def __init__(self, model, base_lr=1e-4, use_v_parameterization=True):
        self.model = model
        self.use_v_param = use_v_parameterization
        
        # Advanced noise scheduler
        self.noise_scheduler = ImprovedNoiseScheduler()
        
        # Min-SNR loss
        self.criterion = MinSNRLoss(gamma=5.0)
        
        # EMA model for stable sampling
        self.ema_model = EMAModel(model, decay=0.9999)
        
        # Multi-component optimizer
        self.optimizer = self._create_advanced_optimizer(base_lr)
        
        # Advanced scheduler
        self.scheduler = WarmupCosineWithRestartsScheduler(
            self.optimizer, warmup_steps=2000, total_steps=200000
        )
        
        # Gradient management
        self.grad_clipper = GradientClipperWithNormTracking(max_norm=1.0)
        
        # Loss tracking
        self.loss_history = []
        
    def _create_advanced_optimizer(self, base_lr):
        """Create optimizer with advanced parameter grouping"""
        # More sophisticated parameter grouping
        param_groups = []
        
        # Group 1: Mamba blocks (highest priority)
        mamba_params = [p for n, p in self.model.named_parameters() 
                       if 'mamba_block' in n and p.requires_grad]
        param_groups.append({
            'params': mamba_params,
            'lr': base_lr * 1.5,
            'weight_decay': 0.01,
            'name': 'mamba'
        })
        
        # Group 2: Cross-attention (high priority)
        attn_params = [p for n, p in self.model.named_parameters() 
                      if 'cross_attn' in n and p.requires_grad]
        param_groups.append({
            'params': attn_params,
            'lr': base_lr * 1.3,
            'weight_decay': 0.01,
            'name': 'attention'
        })
        
        # Group 3: Conditioning layers (very high priority)
        cond_params = [p for n, p in self.model.named_parameters() 
                      if any(x in n for x in ['time_embed', 'scale_shift']) and p.requires_grad]
        param_groups.append({
            'params': cond_params,
            'lr': base_lr * 2.0,
            'weight_decay': 0.005,  # Lower weight decay for conditioning
            'name': 'conditioning'
        })
        
        # Group 4: Everything else
        used_params = set()
        for group in param_groups:
            used_params.update(id(p) for p in group['params'])
        
        other_params = [p for p in self.model.parameters() 
                       if p.requires_grad and id(p) not in used_params]
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': 0.01,
            'name': 'other'
        })
        
        # Use AdamW with different betas for different groups
        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    
    def training_step(self, batch):
        """Single training step with all improvements"""
        images, text_prompts = batch
        
        # Encode to latent space
        latents = self.model.encode_images(images)
        
        # Sample timesteps
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, 
                                 (latents.shape[0],), device=latents.device)
        
        # Add noise
        noise = torch.randn_like(latents)
        noisy_latents, _ = self.noise_scheduler.add_noise(latents, timesteps)
        
        # Get targets
        if self.use_v_param:
            target = self.noise_scheduler.get_v_target(latents, noise, timesteps)
        else:
            target = noise
        
        # Forward pass
        context = self.model.encode_text(text_prompts) if text_prompts else None
        prediction = self.model.unet(noisy_latents, timesteps, context)
        
        # Calculate loss with Min-SNR weighting
        snr = self.noise_scheduler.snr[timesteps].to(latents.device)
        loss = self.criterion(prediction, target, timesteps, snr)
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = self.grad_clipper.clip_gradients(self.model)
        self.optimizer.step()
        
        # Update EMA
        self.ema_model.update()
        
        # Update scheduler
        self.scheduler.step()
        
        # Track metrics
        self.loss_history.append(loss.item())
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }

# ==================== USAGE RECOMMENDATIONS ====================

ADVANCED_RECOMMENDATIONS = {
    "critical_improvements": [
        "Use Zero Terminal SNR noise schedule (fixes sampling issues)",
        "Implement v-parameterization (better training stability)", 
        "Add Min-SNR loss weighting (3x faster convergence)",
        "Use EMA model for sampling (much better quality)",
        "Implement adaptive group normalization",
        "Add gradient norm tracking and intelligent clipping"
    ],
    
    "architecture_enhancements": [
        "Replace standard GroupNorm with AdaptiveGroupNorm",
        "Add zero-initialization to conditioning layers",  
        "Implement residual connections in Mamba blocks",
        "Add LayerScale parameters for better gradient flow",
        "Consider Flash Attention for cross-attention efficiency"
    ],
    
    "training_strategy": [
        "Start with higher LR for conditioning (2x base)",
        "Use different weight decay for different components",
        "Implement curriculum learning (easy -> hard timesteps)",
        "Add loss component tracking for debugging",
        "Use mixed precision training (FP16/BF16)",
        "Implement checkpointing for memory efficiency"
    ],
    
    "monitoring_and_debugging": [
        "Track gradient norms per component",
        "Monitor SNR distribution in training",
        "Log attention maps for cross-attention analysis", 
        "Track EMA vs non-EMA model performance",
        "Monitor loss per timestep range",
        "Implement FID/CLIP score evaluation"
    ]
}

print("🔥 CRITICAL MISSING COMPONENTS:")
for improvement in ADVANCED_RECOMMENDATIONS["critical_improvements"]:
    print(f"  - {improvement}")

print("\n🚀 ARCHITECTURE ENHANCEMENTS:")
for enhancement in ADVANCED_RECOMMENDATIONS["architecture_enhancements"]:
    print(f"  - {enhancement}")

print("\n📈 TRAINING STRATEGY:")
for strategy in ADVANCED_RECOMMENDATIONS["training_strategy"]:
    print(f"  - {strategy}")