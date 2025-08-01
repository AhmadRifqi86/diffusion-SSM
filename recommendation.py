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
        noise = torch.randn_like(latents) #udah di forward nya model
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



import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

class DiffusionMambaAnalyzer:
    def __init__(self, model):
        self.model = model
        self.metrics_history = defaultdict(list)
        self.hooks = []
        self.activations = {}
        
    def register_hooks(self):
        """Register hooks for activation analysis"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Hook mamba blocks
        for i, layer in enumerate(self.model.unet.layers):
            if hasattr(layer, 'main_block'):
                handle = layer.main_block.register_forward_hook(
                    get_activation(f'mamba_block_{i}')
                )
                self.hooks.append(handle)
        
        # Hook cross-attention layers
        for i, layer in enumerate(self.model.unet.layers):
            if hasattr(layer, 'cross_attn'):
                handle = layer.cross_attn.register_forward_hook(
                    get_activation(f'cross_attn_{i}')
                )
                self.hooks.append(handle)
    
    def analyze_gradient_flow(self):
        """Analyze gradient flow through different components"""
        gradient_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                
                # Categorize by component
                if 'mamba' in name.lower():
                    component = 'mamba'
                elif 'cross_attn' in name.lower() or 'attention' in name.lower():
                    component = 'cross_attention'
                elif 'time' in name.lower():
                    component = 'time_embed'
                elif 'context' in name.lower():
                    component = 'context_proj'
                else:
                    component = 'other'
                
                if component not in gradient_stats:
                    gradient_stats[component] = {'grad_norms': [], 'param_norms': []}
                
                gradient_stats[component]['grad_norms'].append(grad_norm)
                gradient_stats[component]['param_norms'].append(param_norm)
        
        # Compute statistics
        for component, stats in gradient_stats.items():
            mean_grad = np.mean(stats['grad_norms'])
            std_grad = np.std(stats['grad_norms'])
            mean_param = np.mean(stats['param_norms'])
            
            print(f"{component}: grad_norm={mean_grad:.6f}±{std_grad:.6f}, "
                  f"param_norm={mean_param:.6f}, "
                  f"grad/param_ratio={mean_grad/mean_param:.8f}")
            
            self.metrics_history[f'{component}_grad_norm'].append(mean_grad)
            self.metrics_history[f'{component}_grad_std'].append(std_grad)
    
    def analyze_mamba_states(self):
        """Analyze mamba state utilization and entropy"""
        if not self.activations:
            print("No activations recorded. Call register_hooks() first.")
            return
        
        mamba_stats = {}
        for name, activation in self.activations.items():
            if 'mamba' in name.lower():
                # Compute activation statistics
                mean_act = activation.mean().item()
                std_act = activation.std().item()
                
                # Compute entropy (measure of state utilization)
                act_probs = F.softmax(activation.flatten(), dim=0)
                entropy = -(act_probs * torch.log(act_probs + 1e-8)).sum().item()
                
                mamba_stats[name] = {
                    'mean': mean_act,
                    'std': std_act,
                    'entropy': entropy
                }
                
                print(f"{name}: mean={mean_act:.4f}, std={std_act:.4f}, entropy={entropy:.4f}")
        
        return mamba_stats
    
    def analyze_cross_attention_patterns(self):
        """Analyze cross-attention alignment quality"""
        attention_stats = {}
        
        for name, activation in self.activations.items():
            if 'cross_attn' in name.lower():
                # Simple attention pattern analysis
                attn_mean = activation.mean().item()
                attn_std = activation.std().item()
                
                # Check for attention collapse (all attention on one token)
                attn_max = activation.max().item()
                attn_min = activation.min().item()
                attention_range = attn_max - attn_min
                
                attention_stats[name] = {
                    'mean': attn_mean,
                    'std': attn_std,
                    'range': attention_range
                }
                
                print(f"{name}: mean={attn_mean:.4f}, std={attn_std:.4f}, range={attention_range:.4f}")
        
        return attention_stats
    
    def analyze_timestep_bias(self, timesteps, losses):
        """Analyze if model has bias towards certain timesteps"""
        timestep_loss_map = defaultdict(list)
        
        for t, loss in zip(timesteps, losses):
            timestep_loss_map[t.item()].append(loss.item())
        
        # Compute average loss per timestep
        timestep_avg_loss = {}
        for t, loss_list in timestep_loss_map.items():
            timestep_avg_loss[t] = np.mean(loss_list)
        
        # Plot timestep bias
        timesteps_sorted = sorted(timestep_avg_loss.keys())
        losses_sorted = [timestep_avg_loss[t] for t in timesteps_sorted]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timesteps_sorted, losses_sorted)
        plt.xlabel('Timestep')
        plt.ylabel('Average Loss')
        plt.title('Loss Distribution Across Timesteps')
        plt.show()
        
        # Check for problematic timesteps
        loss_values = list(timestep_avg_loss.values())
        mean_loss = np.mean(loss_values)
        std_loss = np.std(loss_values)
        
        problematic_timesteps = []
        for t, loss in timestep_avg_loss.items():
            if loss > mean_loss + 2 * std_loss:
                problematic_timesteps.append((t, loss))
        
        if problematic_timesteps:
            print("Problematic timesteps (high loss):")
            for t, loss in problematic_timesteps:
                print(f"  Timestep {t}: loss={loss:.4f}")
        
        return timestep_avg_loss
    
    def diagnose_capacity_bottleneck(self):
        """Diagnose if model has capacity bottlenecks"""
        print("\n=== CAPACITY BOTTLENECK DIAGNOSIS ===")
        
        # Check parameter utilization
        total_params = sum(p.numel() for p in self.model.parameters())
        learning_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Learning parameters: {learning_params:,}")
        print(f"Parameter utilization: {learning_params/total_params:.2%}")
        
        # Check gradient magnitudes
        component_grads = defaultdict(list)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if 'mamba' in name.lower():
                    component_grads['mamba'].append(param.grad.norm().item())
                elif 'cross_attn' in name.lower():
                    component_grads['cross_attn'].append(param.grad.norm().item())
        
        for component, grads in component_grads.items():
            mean_grad = np.mean(grads)
            if mean_grad < 1e-6:
                print(f"WARNING: {component} has very small gradients ({mean_grad:.2e})")
                print("  Possible capacity bottleneck or learning rate too low")
    
    def generate_training_report(self):
        """Generate comprehensive training analysis report"""
        print("\n" + "="*50)
        print("DIFFUSION-MAMBA TRAINING ANALYSIS REPORT")
        print("="*50)
        
        self.analyze_gradient_flow()
        print("\n" + "-"*30)
        self.analyze_mamba_states()
        print("\n" + "-"*30)
        self.analyze_cross_attention_patterns()
        print("\n" + "-"*30)
        self.diagnose_capacity_bottleneck()
        print("\n" + "="*50)
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# Usage in your training loop:
def enhanced_training_step(self, batch, analyzer=None):
    """Enhanced training step with analysis"""
    images, text_prompts = batch
    timesteps = torch.randint(0, self.model.noise_scheduler.num_train_timesteps,
                            (images.shape[0],), device=images.device)
    
    with autocast(device_type="cuda"):
        predicted_noise, noise, latents = self.model(images, timesteps, text_prompts)
        target = self.model.noise_scheduler.get_v_target(latents, noise, timesteps) if self.use_v_param else noise
        
        snr = self.model.noise_scheduler.snr[timesteps.to(self.model.noise_scheduler.snr.device)].to(images.device).float()
        loss = self.criterion(predicted_noise, target, timesteps, snr)
        loss = loss / self.gradient_accumulation_steps
    
    self.scaler.scale(loss).backward()
    
    # Analysis every 100 steps
    if analyzer and (self.current_step % 100 == 0):
        analyzer.analyze_gradient_flow()
        analyzer.analyze_timestep_bias(timesteps, [loss.item()] * len(timesteps))
    
    # Rest of your training step...
    return {'loss': loss.item() * self.gradient_accumulation_steps}

# Integration example:
analyzer = DiffusionMambaAnalyzer(model)
analyzer.register_hooks()

# During training
for epoch in range(num_epochs):
    for batch in dataloader:
        metrics = enhanced_training_step(batch, analyzer)
        
        # Generate report every 10 epochs
        if epoch % 10 == 0 and batch_idx == 0:
            analyzer.generate_training_report()


import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class BlockAnalyzer:  #Low level per-block analyzer, track when val_loss is reaching somewhere between 0.2-0.4
    def __init__(self, model):
        self.model = model
        self.gradient_history = defaultdict(list)
        self.weight_importance = defaultdict(list)
        self.learning_effectiveness = defaultdict(list)
        self.activation_stats = defaultdict(list)
        self.hooks = []
        
    def track_gradient_metrics(self):
        """Track comprehensive gradient-based learning metrics"""
        block_metrics = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 1. Gradient Signal Strength
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                
                # 2. Gradient-to-Parameter Ratio (learning effectiveness)
                grad_param_ratio = grad_norm / (param_norm + 1e-8)
                
                # 3. Gradient Variance (stability indicator)
                grad_flat = param.grad.flatten()
                grad_var = torch.var(grad_flat).item()
                grad_mean = torch.mean(grad_flat).item()
                
                # 4. Weight Update Magnitude
                lr = self.get_effective_lr(name)
                update_magnitude = (lr * grad_norm)
                
                # 5. Signal-to-Noise Ratio in gradients
                grad_abs_mean = torch.mean(torch.abs(grad_flat)).item()
                snr = grad_abs_mean / (torch.std(grad_flat).item() + 1e-8)
                
                block_name = self.categorize_parameter(name)
                if block_name not in block_metrics:
                    block_metrics[block_name] = {
                        'grad_norms': [], 'param_norms': [], 'grad_param_ratios': [],
                        'grad_vars': [], 'update_magnitudes': [], 'snrs': []
                    }
                
                block_metrics[block_name]['grad_norms'].append(grad_norm)
                block_metrics[block_name]['param_norms'].append(param_norm)
                block_metrics[block_name]['grad_param_ratios'].append(grad_param_ratio)
                block_metrics[block_name]['grad_vars'].append(grad_var)
                block_metrics[block_name]['update_magnitudes'].append(update_magnitude)
                block_metrics[block_name]['snrs'].append(snr)
        
        # Analyze and store results
        for block_name, metrics in block_metrics.items():
            avg_grad_norm = np.mean(metrics['grad_norms'])
            avg_ratio = np.mean(metrics['grad_param_ratios'])
            avg_update = np.mean(metrics['update_magnitudes'])
            avg_snr = np.mean(metrics['snrs'])
            
            self.gradient_history[f'{block_name}_grad_norm'].append(avg_grad_norm)
            self.learning_effectiveness[f'{block_name}_ratio'].append(avg_ratio)
            
            # Diagnose learning issues
            if avg_grad_norm < 1e-6:
                print(f"⚠️  {block_name}: Very small gradients ({avg_grad_norm:.2e}) - may need higher LR")
            elif avg_grad_norm > 1e-2:
                print(f"⚠️  {block_name}: Large gradients ({avg_grad_norm:.2e}) - may need lower LR or clipping")
            
            if avg_ratio < 1e-5:
                print(f"⚠️  {block_name}: Poor learning effectiveness ({avg_ratio:.2e}) - weights barely changing")
            
            if avg_snr < 1.0:
                print(f"⚠️  {block_name}: Low gradient SNR ({avg_snr:.2f}) - noisy learning signal")
        
        return block_metrics
    
    def track_weight_importance(self):
        """Track weight importance using multiple methods"""
        importance_metrics = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Method 1: Gradient-Weight Product (GWP)
                # Indicates which weights have strong gradient signal
                gwp = torch.abs(param * param.grad).mean().item()
                
                # Method 2: Fisher Information approximation
                # Second moment of gradients
                fisher_diag = (param.grad ** 2).mean().item()
                
                # Method 3: Weight Magnitude Distribution
                weight_std = param.std().item() 
                weight_mean_abs = torch.abs(param).mean().item()
                
                # Method 4: Activation-based importance (if we have activations)
                # This requires forward hooks - simplified version
                activation_sensitivity = self.compute_activation_sensitivity(name, param)
                
                block_name = self.categorize_parameter(name)
                if block_name not in importance_metrics:
                    importance_metrics[block_name] = {
                        'gwp': [], 'fisher': [], 'weight_std': [], 
                        'weight_mean_abs': [], 'activation_sens': []
                    }
                
                importance_metrics[block_name]['gwp'].append(gwp)
                importance_metrics[block_name]['fisher'].append(fisher_diag)
                importance_metrics[block_name]['weight_std'].append(weight_std)
                importance_metrics[block_name]['weight_mean_abs'].append(weight_mean_abs)
                importance_metrics[block_name]['activation_sens'].append(activation_sensitivity)
        
        # Store aggregated importance scores
        for block_name, metrics in importance_metrics.items():
            avg_gwp = np.mean(metrics['gwp'])
            avg_fisher = np.mean(metrics['fisher'])
            
            self.weight_importance[f'{block_name}_gwp'].append(avg_gwp)
            self.weight_importance[f'{block_name}_fisher'].append(avg_fisher)
            
            print(f"{block_name}: GWP={avg_gwp:.6f}, Fisher={avg_fisher:.6f}")
        
        return importance_metrics
    
    def analyze_lr_weight_relationship(self, loss_history):
        """Analyze relationship between learning rate and weight changes"""
        lr_effects = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None and len(self.gradient_history) > 1:
                block_name = self.categorize_parameter(name)
                lr = self.get_effective_lr(name)
                
                # Calculate weight change rate
                current_norm = param.norm().item()
                if hasattr(self, 'prev_param_norms') and block_name in self.prev_param_norms:
                    weight_change = abs(current_norm - self.prev_param_norms[block_name])
                    
                    # LR effectiveness: weight_change per unit learning rate
                    lr_effectiveness = weight_change / (lr + 1e-8)
                    
                    if block_name not in lr_effects:
                        lr_effects[block_name] = []
                    lr_effects[block_name].append(lr_effectiveness)
                
                # Store current norms for next iteration
                if not hasattr(self, 'prev_param_norms'):
                    self.prev_param_norms = {}
                self.prev_param_norms[block_name] = current_norm
        
        return lr_effects
    
    def analyze_loss_weight_correlation(self, current_loss):
        """Analyze how loss changes correlate with weight changes"""
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) < 10:
            return {}
        
        correlations = {}
        recent_loss_trend = np.diff(self.loss_history[-10:])  # Last 10 loss changes
        
        for block_name in ['mamba_block', 'cross_attention', 'time_embed']:
            if f'{block_name}_grad_norm' in self.gradient_history:
                recent_grad_trend = np.diff(self.gradient_history[f'{block_name}_grad_norm'][-10:])
                
                if len(recent_grad_trend) == len(recent_loss_trend):
                    # Correlation between loss change and gradient change
                    correlation = np.corrcoef(recent_loss_trend, recent_grad_trend)[0, 1]
                    correlations[block_name] = correlation
                    
                    # Interpretation
                    if abs(correlation) > 0.7:
                        direction = "positively" if correlation > 0 else "negatively"
                        print(f"{block_name}: Strongly {direction} correlated with loss (r={correlation:.3f})")
                    elif abs(correlation) < 0.3:
                        print(f"{block_name}: Weakly correlated with loss (r={correlation:.3f}) - may need attention")
        
        return correlations
    
    def compute_activation_sensitivity(self, param_name, param):
        """Compute sensitivity based on activation patterns"""
        # Simplified version - in practice, you'd use forward hooks
        # This is a placeholder that returns gradient-based sensitivity
        if param.grad is not None:
            return torch.abs(param.grad).mean().item()
        return 0.0
    
    def categorize_parameter(self, param_name):
        """Categorize parameters by block type"""
        name_lower = param_name.lower()
        if 'mamba' in name_lower:
            return 'mamba_block'
        elif 'cross_attn' in name_lower or 'attention' in name_lower:
            return 'cross_attention'
        elif 'time' in name_lower:
            return 'time_embed'
        elif 'context' in name_lower:
            return 'context_proj'
        elif 'input' in name_lower:
            return 'input_proj'
        elif 'scale' in name_lower or 'shift' in name_lower:
            return 'scale_shift'
        else:
            return 'other'
    
    def get_effective_lr(self, param_name):
        """Get effective learning rate for a parameter"""
        # This should be implemented based on your optimizer setup
        # Placeholder implementation
        base_lr = 1e-5  # Your base learning rate
        
        # Apply param group scaling
        block_name = self.categorize_parameter(param_name)
        lr_scales = {
            'mamba_block': 1.8,
            'cross_attention': 1.5,
            'time_embed': 2.2,
            'scale_shift': 2.2,
            'context_proj': 0.7,
            'input_proj': 0.7
        }
        
        return base_lr * lr_scales.get(block_name, 1.0)
    
    def diagnose_block_health(self):
        """Comprehensive block health diagnosis"""
        print("\n" + "="*60)
        print("BLOCK HEALTH DIAGNOSIS")
        print("="*60)
        
        health_report = {}
        
        for block_name in ['mamba_block', 'cross_attention', 'time_embed', 'context_proj']:
            print(f"\n--- {block_name.upper()} ---")
            
            # Check gradient health
            if f'{block_name}_grad_norm' in self.gradient_history:
                recent_grads = self.gradient_history[f'{block_name}_grad_norm'][-5:]
                grad_trend = "increasing" if recent_grads[-1] > recent_grads[0] else "decreasing"
                grad_stability = np.std(recent_grads) / (np.mean(recent_grads) + 1e-8)
                
                print(f"  Gradient trend: {grad_trend}")
                print(f"  Gradient stability: {grad_stability:.3f} (lower is better)")
                
                # Diagnosis
                if grad_stability > 1.0:
                    print("  ⚠️  High gradient instability - consider lower LR or gradient clipping")
                elif np.mean(recent_grads) < 1e-6:
                    print("  ⚠️  Very small gradients - consider higher LR")
                else:
                    print("  ✅ Gradient health looks good")
            
            # Check learning effectiveness
            if f'{block_name}_ratio' in self.learning_effectiveness:
                recent_ratios = self.learning_effectiveness[f'{block_name}_ratio'][-5:]
                avg_ratio = np.mean(recent_ratios)
                
                print(f"  Learning effectiveness: {avg_ratio:.8f}")
                
                if avg_ratio < 1e-6:
                    print("  ⚠️  Poor learning effectiveness - weights barely changing")
                    health_report[block_name] = 'needs_higher_lr'
                elif avg_ratio > 1e-3:
                    print("  ⚠️  Very high learning rate - may be unstable")
                    health_report[block_name] = 'needs_lower_lr'
                else:
                    print("  ✅ Learning effectiveness looks good")
                    health_report[block_name] = 'healthy'
        
        return health_report
    
    def suggest_lr_adjustments(self):
        """Suggest learning rate adjustments based on analysis"""
        health_report = self.diagnose_block_health()
        suggestions = {}
        
        print(f"\n--- LEARNING RATE ADJUSTMENT SUGGESTIONS ---")
        
        for block_name, health_status in health_report.items():
            current_lr_scale = {
                'mamba_block': 1.8,
                'cross_attention': 1.5, 
                'time_embed': 2.2,
                'context_proj': 0.7
            }.get(block_name, 1.0)
            
            if health_status == 'needs_higher_lr':
                new_scale = current_lr_scale * 1.5
                suggestions[block_name] = new_scale
                print(f"  {block_name}: Increase lr_scale from {current_lr_scale} to {new_scale:.2f}")
            
            elif health_status == 'needs_lower_lr':
                new_scale = current_lr_scale * 0.7
                suggestions[block_name] = new_scale
                print(f"  {block_name}: Decrease lr_scale from {current_lr_scale} to {new_scale:.2f}")
            
            else:
                suggestions[block_name] = current_lr_scale
                print(f"  {block_name}: Keep current lr_scale at {current_lr_scale}")
        
        return suggestions

# Usage in training loop:
def enhanced_training_with_block_analysis(self, batch):
    """Training step with comprehensive block analysis"""
    # Your existing training code...
    images, text_prompts = batch
    # ... forward pass, loss computation ...
    
    # Initialize analyzer if not exists
    if not hasattr(self, 'block_analyzer'):
        self.block_analyzer = BlockAnalyzer(self.model)
    
    # Backward pass
    loss.backward()
    
    # Analyze every 50 steps
    if self.current_step % 50 == 0:
        # Track all metrics
        grad_metrics = self.block_analyzer.track_gradient_metrics()
        importance_metrics = self.block_analyzer.track_weight_importance()
        lr_effects = self.block_analyzer.analyze_lr_weight_relationship(self.loss_history)
        correlations = self.block_analyzer.analyze_loss_weight_correlation(loss.item())
        
        # Generate suggestions every 500 steps
        if self.current_step % 500 == 0:
            suggestions = self.block_analyzer.suggest_lr_adjustments()
            
            # Log to wandb or your preferred logger
            import wandb
            wandb.log({
                "block_analysis/suggestions": suggestions,
                "block_analysis/grad_metrics": grad_metrics,
                "block_analysis/correlations": correlations
            })
    
    # Continue with optimization step...
    return {'loss': loss.item()}

# Example of implementing suggestions:
def update_optimizer_from_suggestions(optimizer, suggestions):
    """Update optimizer learning rates based on analysis"""
    for param_group in optimizer.param_groups:
        group_name = param_group.get('name', 'default')
        if group_name in suggestions:
            old_lr = param_group['lr']
            new_lr_scale = suggestions[group_name]
            param_group['lr'] = param_group['base_lr'] * new_lr_scale
            print(f"Updated {group_name} LR from {old_lr:.2e} to {param_group['lr']:.2e}")



def training_step(self, batch):
    """Clean training step with analyzer monitoring only"""
    images, text_prompts = batch
    timesteps = torch.randint(
        0, self.model.noise_scheduler.num_train_timesteps,
        (images.shape[0],), device=images.device
    )
    images = images.to(next(self.model.parameters()).device)
    timesteps = timesteps.to(next(self.model.parameters()).device)
    
    # Initialize analyzer once (non-intrusive)
    if not hasattr(self, 'analyzer'):
        base_lr = self.optimizer.param_groups[0]['lr']
        self.analyzer = ComprehensiveDiffusionMambaAnalyzer(self.model, base_lr=base_lr)
        self.analyzer.register_hooks()
        print("📊 Training analyzer initialized (monitoring mode)")
    
    # Track accumulation steps
    if not hasattr(self, '_accum_count'):
        self._accum_count = 0
    
    with autocast(device_type="cuda", dtype=self.amp_dtype):
        predicted_noise, noise, latents = self.model(images, timesteps, text_prompts)
        target = self.model.noise_scheduler.get_v_target(latents, noise, timesteps) if self.use_v_param else noise
        predicted_noise = predicted_noise.float()
        target = target.float()
        snr = self.model.noise_scheduler.snr[timesteps.to(self.model.noise_scheduler.snr.device)].to(images.device).float()
        loss = self.criterion(predicted_noise, target, timesteps, snr)
        
        # Normalize loss
        loss = loss / self.gradient_accumulation_steps
    
    # Backward pass
    if self.scaler is not None:
        self.scaler.scale(loss).backward()
    else:
        loss.backward()
    
    # ANALYZER MONITORING - Just analyze and report
    actual_loss = loss.item() * self.gradient_accumulation_steps
    analysis_result = self.analyzer.analyze_step(
        loss=actual_loss,
        timesteps=timesteps
    )
    
    self._accum_count += 1
    
    # Default values
    grad_norm = 0.0
    is_update_step = False
    
    # Optimizer step when accumulation is complete
    if self._accum_count >= self.gradient_accumulation_steps:
        grad_norm = self.grad_clipper.clip_gradients(self.model)
        
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Update EMA and scheduler
        self.ema_model.update()
        self.scheduler.step()
        
        # Reset counter
        self._accum_count = 0
        is_update_step = True
    
    # Logging
    self.loss_history.append(actual_loss)
    if is_update_step:
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
    
    # Return with optional analysis info
    result = {
        'loss': actual_loss,
        'grad_norm': grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
        'lr': self.optimizer.param_groups[0]['lr'],
        'optimizer_step': is_update_step
    }
    
    # Add analysis results when available (every 500 steps)
    if analysis_result:
        result['analysis_report'] = analysis_result
        result['has_analysis'] = True
    
    return result


class TrainingMonitor:
    """
    Separate class to handle analysis results and generate alerts
    This keeps your training loop clean while providing rich monitoring
    """
    
    def __init__(self, alert_thresholds=None):
        self.alert_thresholds = alert_thresholds or {
            'gradient_explosion': 1e-1,
            'gradient_vanishing': 1e-7,
            'dead_neuron_ratio': 0.3,
            'loss_increase_steps': 50,
            'lr_effectiveness_threshold': 1e-6
        }
        
        self.alert_history = []
        self.last_alert_step = 0
        self.alert_cooldown = 100  # Don't spam alerts
    
    def process_training_metrics(self, step, metrics, analyzer=None):
        """Process training metrics and generate informative alerts"""
        alerts = []
        
        # Handle analysis reports
        if metrics.get('has_analysis') and analyzer:
            diagnosis = analyzer.diagnose_training_health()
            suggestions = analyzer.suggest_optimizations()
            
            # Generate smart alerts based on analysis
            alerts.extend(self._generate_health_alerts(diagnosis, step))
            alerts.extend(self._generate_suggestion_alerts(suggestions, step))
            
            # Check for emergency conditions
            emergency_alerts = self._check_emergency_conditions(diagnosis, step)
            if emergency_alerts:
                alerts.extend(emergency_alerts)
        
        # Always check basic metrics
        basic_alerts = self._check_basic_metrics(metrics, step)
        alerts.extend(basic_alerts)
        
        # Display alerts with cooldown
        if alerts and (step - self.last_alert_step) > self.alert_cooldown:
            self._display_alerts(step, alerts)
            self.last_alert_step = step
        
        return alerts
    
    def _generate_health_alerts(self, diagnosis, step):
        """Generate alerts based on health diagnosis"""
        alerts = []
        
        if diagnosis['overall_status'] == 'needs_attention':
            alerts.append({
                'type': 'warning',
                'message': f"Training health needs attention ({len(diagnosis['issues'])} issues detected)",
                'details': diagnosis['issues'][:3],  # Show top 3 issues
                'step': step
            })
        
        # Component-specific alerts
        for component, status in diagnosis.get('gradient_health', {}).items():
            if status in ['vanishing_gradients', 'exploding_gradients']:
                alerts.append({
                    'type': 'critical' if 'exploding' in status else 'warning',
                    'message': f"{component.replace('_', ' ').title()}: {status.replace('_', ' ')}",
                    'component': component,
                    'step': step
                })
        
        return alerts
    
    def _generate_suggestion_alerts(self, suggestions, step):
        """Generate alerts based on optimization suggestions"""
        alerts = []
        
        if suggestions.get('priority') == 'high':
            lr_changes = suggestions.get('learning_rates', {})
            if lr_changes:
                components_needing_change = list(lr_changes.keys())
                alerts.append({
                    'type': 'action_needed',
                    'message': f"High priority: Consider adjusting learning rates for {', '.join(components_needing_change)}",
                    'details': [f"{comp}: {info['current']:.2f} → {info['suggested']:.2f}" 
                              for comp, info in lr_changes.items()],
                    'step': step
                })
        
        # Architecture suggestions
        if suggestions.get('architecture'):
            alerts.append({
                'type': 'info',
                'message': "Architecture optimization suggested",
                'details': suggestions['architecture'][:2],  # Show top 2
                'step': step
            })
        
        return alerts
    
    def _check_emergency_conditions(self, diagnosis, step):
        """Check for conditions that require immediate attention"""
        alerts = []
        
        # Count critical issues
        critical_issues = []
        for category, issues in diagnosis.items():
            if isinstance(issues, dict):
                for component, status in issues.items():
                    if status in ['exploding_gradients', 'dead_neurons', 'excessive_learning']:
                        critical_issues.append(f"{category}.{component}: {status}")
        
        if len(critical_issues) >= 3:
            alerts.append({
                'type': 'emergency',
                'message': f"EMERGENCY: {len(critical_issues)} critical issues detected!",
                'details': critical_issues[:3],
                'action': "Consider stopping training and investigating",
                'step': step
            })
        
        return alerts
    
    def _check_basic_metrics(self, metrics, step):
        """Check basic metrics for issues"""
        alerts = []
        
        # Gradient norm checks
        grad_norm = metrics.get('grad_norm', 0)
        if grad_norm > self.alert_thresholds['gradient_explosion']:
            alerts.append({
                'type': 'critical',
                'message': f"Gradient explosion detected: {grad_norm:.2e}",
                'action': "Consider gradient clipping or lower learning rate",
                'step': step
            })
        elif grad_norm < self.alert_thresholds['gradient_vanishing']:
            alerts.append({
                'type': 'warning',
                'message': f"Very small gradients: {grad_norm:.2e}",
                'action': "Consider higher learning rate",
                'step': step
            })
        
        return alerts
    
    def _display_alerts(self, step, alerts):
        """Display alerts in a clean, informative format"""
        if not alerts:
            return
        
        # Group alerts by type
        emergency_alerts = [a for a in alerts if a['type'] == 'emergency']
        critical_alerts = [a for a in alerts if a['type'] == 'critical']
        warning_alerts = [a for a in alerts if a['type'] == 'warning']
        action_alerts = [a for a in alerts if a['type'] == 'action_needed']
        info_alerts = [a for a in alerts if a['type'] == 'info']
        
        print(f"\n{'='*60}")
        print(f"📊 TRAINING ANALYSIS ALERT - Step {step}")
        print(f"{'='*60}")
        
        # Emergency alerts (red)
        for alert in emergency_alerts:
            print(f"🚨 EMERGENCY: {alert['message']}")
            if alert.get('details'):
                for detail in alert['details']:
                    print(f"   • {detail}")
            if alert.get('action'):
                print(f"   → {alert['action']}")
        
        # Critical alerts (red)
        for alert in critical_alerts:
            print(f"🔥 CRITICAL: {alert['message']}")
            if alert.get('action'):
                print(f"   → {alert['action']}")
        
        # Warning alerts (yellow)
        for alert in warning_alerts:
            print(f"⚠️  WARNING: {alert['message']}")
            if alert.get('details'):
                for detail in alert['details'][:2]:  # Limit details
                    print(f"   • {detail}")
        
        # Action needed (blue)
        for alert in action_alerts:
            print(f"🔧 ACTION NEEDED: {alert['message']}")
            if alert.get('details'):
                for detail in alert['details'][:3]:
                    print(f"   • {detail}")
        
        # Info alerts (green)
        for alert in info_alerts:
            print(f"💡 INFO: {alert['message']}")
        
        print(f"{'='*60}\n")
        
        # Store alert for history
        self.alert_history.append({
            'step': step,
            'alert_count': len(alerts),
            'emergency_count': len(emergency_alerts),
            'critical_count': len(critical_alerts)
        })


# Usage example in your training loop
def enhanced_training_loop(self):
    """Example training loop with clean monitoring"""
    
    # Initialize monitor
    monitor = TrainingMonitor()
    
    for epoch in range(self.num_epochs):
        for batch_idx, batch in enumerate(self.dataloader):
            
            # Your clean training step
            metrics = self.training_step(batch)
            
            # Process metrics and generate alerts
            alerts = monitor.process_training_metrics(
                step=batch_idx + epoch * len(self.dataloader),
                metrics=metrics,
                analyzer=getattr(self, 'analyzer', None)
            )
            
            # Simple logging
            if batch_idx % 100 == 0:
                status = "🔥" if any(a['type'] == 'critical' for a in alerts) else "✅"
                print(f"{status} Epoch {epoch}, Step {batch_idx}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"GradNorm={metrics['grad_norm']:.2e}, "
                      f"LR={metrics['lr']:.2e}")
            
            # Save analysis reports when generated
            if metrics.get('has_analysis'):
                report_path = f'analysis_reports/report_epoch_{epoch}_step_{batch_idx}.txt'
                with open(report_path, 'w') as f:
                    f.write(metrics['analysis_report'])
            
            # Optional: Generate plots periodically
            if batch_idx % 1000 == 0 and hasattr(self, 'analyzer'):
                self.analyzer.plot_training_metrics(
                    f'plots/training_epoch_{epoch}_step_{batch_idx}.png'
                )
    
    # Final summary
    if hasattr(self, 'analyzer'):
        print("\n🏁 Training completed. Generating final analysis...")
        final_report = self.analyzer.generate_comprehensive_report()
        
        # Save final analysis
        with open('final_analysis_report.txt', 'w') as f:
            f.write(final_report)
        
        self.analyzer.export_metrics('final_metrics.json')
        self.analyzer.cleanup()
        print("📊 Final analysis saved to 'final_analysis_report.txt'")


# Simple alert-only integration (even cleaner)
def minimal_integration_example(self):
    """Minimal integration - just add these 3 lines to your existing training loop"""
    
    # In your existing training loop, just add:
    for batch_idx, batch in enumerate(self.dataloader):
        metrics = self.training_step(batch)  # Your existing function
        
        # Add just these 3 lines for monitoring:
        if batch_idx == 0:  # Initialize once
            self.monitor = TrainingMonitor()
        
        alerts = self.monitor.process_training_metrics(batch_idx, metrics, getattr(self, 'analyzer', None))
        
        # That's it! Alerts will be displayed automatically when issues are detected