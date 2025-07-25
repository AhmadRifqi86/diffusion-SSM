import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from train.custom import CosineAnnealingWarmRestartsWithDecay, EarlyStopping
from train.custom import EMAModel, MinSNRVLoss, GradientClipperWithNormTracking
from models.diffuse import UShapeMambaDiffusion
from models.blocks import CrossAttention
import logging
import tqdm
import os


class AdvancedDiffusionTrainer:
    """
    Complete advanced training setup incorporating all improvements
    """
    def __init__(self, model: UShapeMambaDiffusion, base_lr=1e-4, use_v_parameterization=True, checkpoint_dir="checkpoints"):
        self.model = model
        self.use_v_param = use_v_parameterization

        # Min-SNR loss
        self.criterion = MinSNRVLoss(gamma=5.0)

        # EMA model for stable sampling
        self.ema_model = EMAModel(model, decay=0.9999)

        # Multi-component optimizer
        self.optimizer = self._create_advanced_optimizer(base_lr)
        self.early_stop = EarlyStopping(patience=20, min_delta=1e-4, restore_best_weights=True)

        # Advanced scheduler, consider using sequentialLR
        self.scheduler = CosineAnnealingWarmRestartsWithDecay(
            self.optimizer, T_0=1000, T_mult=2, eta_min=1e-6, decay=0.9)

        # Gradient management
        self.grad_clipper = GradientClipperWithNormTracking(max_norm=1.0)
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

        # Loss tracking
        self.loss_history = []
        self.val_loss_history = []
        self.learning_rates = []
        self.best_loss = float('inf')

        # Internal tracking
        self.start_epoch = 0
        self.checkpoint_dir = checkpoint_dir
        self.logger = logging.getLogger(__name__)

    def _create_advanced_optimizer(self, base_lr):
        """Create optimizer with structured parameter groups by module type and role."""
        param_groups = []
        used = set()

        def add_group(name, params, lr_mult=1.0, wd=0.01):
            """Add a parameter group for optimizer."""
            filtered = [p for p in params if p.requires_grad and id(p) not in used]
            for p in filtered:
                used.add(id(p))
            if filtered:
                param_groups.append({'params': filtered, 'lr': base_lr * lr_mult, 'weight_decay': wd})

        # --- Named module patterns and their custom LR / weight decay
        pattern_config = {
            'mamba_block':     (1.5, 0.01),
            'CrossAttention':  (1.3, 0.01),  # by module type
            'time_embed':      (2.0, 0.005),
            'scale_shift':     (2.0, 0.005),
            'vae_proj':        (0.2, 0.0),
            'context_proj':    (0.2, 0.0),
            'timestep_embedding': (0.2, 0.0),
        }

        # --- Group known patterns (by name or type)
        for name, module in self.model.named_modules():
            for key, (lr_mult, wd) in pattern_config.items():
                if key in name or (key == 'CrossAttention' and isinstance(module, CrossAttention)):
                    add_group(name, module.parameters(), lr_mult=lr_mult, wd=wd)
                    break  # prevent double-assignment

        # --- U-Net block structure (decaying LR)
        if hasattr(self.model, 'unet'):
            for i, block in enumerate(self.model.unet.down_blocks):
                add_group(f"unet.down_blocks.{i}", block.parameters(), lr_mult=(0.95 ** i), wd=0.01)
            for i, block in enumerate(self.model.unet.up_blocks):
                add_group(f"unet.up_blocks.{i}", block.parameters(), lr_mult=(0.95 ** i), wd=0.01)
            add_group("unet.middle_block", self.model.unet.middle_block.parameters(), lr_mult=0.7, wd=0.01)

        # --- Catch-all fallback
        remaining = [p for p in self.model.parameters() if p.requires_grad and id(p) not in used]
        if remaining:
            param_groups.append({'params': remaining, 'lr': base_lr, 'weight_decay': 0.01})

        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    def checkpoint(self, save_path, epoch, val_loss):
        """
        Save a full training checkpoint at a specific path.
        `save_path` must be a full path like 'checkpoints/cp_epoch10.pt' or 'checkpoints/best_model.pt'
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
        }

        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint saved to {save_path}")

    def training_step(self, batch):
        """Single training step with autocast, EMA, Min-SNR, grad clipping"""
        images, text_prompts = batch
        timesteps = torch.randint(
            0, self.model.noise_scheduler.num_train_timesteps,
            (images.shape[0],), device=images.device
        )
        images = images.to(next(self.model.parameters()).device)
        timesteps = timesteps.to(next(self.model.parameters()).device)

        with autocast(device_type="cuda"):
            predicted_noise, noise, latents = self.model(images, timesteps, text_prompts)
            target = self.model.noise_scheduler.get_v_target(latents, noise, timesteps) if self.use_v_param else noise
            snr = self.model.noise_scheduler.snr[timesteps].to(images.device)
            loss = self.criterion(predicted_noise, target, timesteps, snr)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        grad_norm = self.grad_clipper.clip_gradients(self.model)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema_model.update()
        self.scheduler.step()

        self.loss_history.append(loss.item())
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

        return {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def validate(self, val_loader):
        """Validation loop for one epoch"""
        self.model.eval()
        total_val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images, text_prompts = batch
                timesteps = torch.randint(
                    0, self.model.noise_scheduler.num_train_timesteps,
                    (images.shape[0],), device=images.device
                )
                images = images.to(next(self.model.parameters()).device)
                timesteps = timesteps.to(next(self.model.parameters()).device)
                with autocast(device_type="cuda"):
                    predicted_noise, noise, latents = self.model(images, timesteps, text_prompts)
                    target = self.model.noise_scheduler.get_v_target(latents, noise, timesteps) if self.use_v_param else noise
                    snr = self.model.noise_scheduler.snr[timesteps].to(images.device)
                    loss = self.criterion(predicted_noise, target, timesteps, snr)

                total_val_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_val_loss / max(1, num_batches)
        self.val_loss_history.append(avg_val_loss)

        if avg_val_loss < self.best_loss:
            self.best_loss = avg_val_loss

        self.model.train()
        return avg_val_loss
    
    def train(self, dataloader, val_dataloader, num_epochs, checkpoint_dir="checkpoints"):
        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            print(f"📘 Epoch {epoch}/{num_epochs}")
            self.model.train()
            epoch_losses = []

            for batch in dataloader:
                stats = self.training_step(batch)
                epoch_losses.append(stats['loss'])

            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"📉 Avg Train Loss: {avg_train_loss:.4f}")

            # Validation
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_loss = self.validate(val_batch)
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"🔎 Validation Loss: {avg_val_loss:.4f}")

            # Save per-epoch checkpoint
            epoch_cp_path = os.path.join(checkpoint_dir, f"cp_epoch{epoch}.pt")
            self.checkpoint(epoch_cp_path, epoch, avg_val_loss)

            # Save best checkpoint if improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_cp_path = os.path.join(checkpoint_dir, "best_model.pt")
                self.checkpoint(best_cp_path, epoch, avg_val_loss)# Early stopping

            if self.early_stop(avg_val_loss):
                print("⏹️ Early stopping triggered.")
                break
            

