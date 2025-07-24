import torch
import torch.optim as optim
from custom import CosineAnnealingWarmRestartsWithDecay,EarlyStopping
from custom import GradientClipperWithNormTracking,EMAModel,MinSNRVLoss

class AdvancedDiffusionTrainer:
    """
    Complete advanced training setup incorporating all improvements
    """
    def __init__(self, model, base_lr=1e-4, use_v_parameterization=True):
        self.model = model
        self.use_v_param = use_v_parameterization
        
        # Advanced noise scheduler
        #self.noise_scheduler = ImprovedNoiseScheduler()
        
        # Min-SNR loss
        self.criterion = MinSNRVLoss(gamma=5.0)
        
        # EMA model for stable sampling
        self.ema_model = EMAModel(model, decay=0.9999)
        
        # Multi-component optimizer
        self.optimizer = self._create_advanced_optimizer(base_lr)
        self.early_stop = EarlyStopping(patience=20, min_delta=1e-4, restore_best_weights=True)
        
        # Advanced scheduler
        # self.scheduler = WarmupCosineWithRestartsScheduler(
        #     self.optimizer, warmup_steps=2000, total_steps=200000
        # )
        self.scheduler = CosineAnnealingWarmRestartsWithDecay(0.1, T_0=1000, T_mult=2, eta_min=1e-6, decay_factor=0.9)
        
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