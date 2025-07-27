import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from train.dataloader import collate_fn, create_datasets_with_indices
from train.custom import CosineAnnealingWarmRestartsWithDecay, EarlyStopping
from train.custom import EMAModel, MinSNRVLoss, GradientClipperWithNormTracking
from models.diffuse import UShapeMambaDiffusion
from models.blocks import CrossAttention
import logging
from train.factory import OptimizerSchedulerFactory
from tqdm import tqdm
import os
from torch.utils.data import DataLoader


class AdvancedDiffusionTrainer:  #Resuming nya belom kalau pake indices dataset
    """
    Complete advanced training setup incorporating all improvements
    """
    def __init__(self, model: UShapeMambaDiffusion, config):
        self.model = model
        self.config = config or {}
        self.use_v_param = config.Train.use_v_parameterization
        self.criterion = MinSNRVLoss(gamma=5.0)
        self.ema_model = EMAModel(model, decay=0.9999)

        # Multi-component optimizer
        self.early_stop = EarlyStopping(patience=20, min_delta=1e-4, restore_best_weights=True)
        
        self.optimizer = OptimizerSchedulerFactory.create_advanced_optimizer(self.model, config)
        self.scheduler = OptimizerSchedulerFactory.create_advanced_scheduler(self.optimizer, config)

        # Gradient management
        self.grad_clipper = GradientClipperWithNormTracking(max_norm=1.0)
        self.scaler = GradScaler(enabled=torch.cuda.is_available()) # if dtype for train is FP16, else no need for GradScale 

        # Loss tracking
        self.loss_history = []
        self.val_loss_history = []
        self.learning_rates = []
        self.best_loss = float('inf')

        # Internal tracking
        self.start_epoch = 0
        self.checkpoint_dir = config.Train.Checkpoint.checkpoint_dir #ganti jadi config aja
        self.logger = logging.getLogger(__name__)


    def checkpoint(self, save_path, epoch, val_loss, train_indices=None, val_indices=None):
        """
        Save a full training checkpoint at a specific path.
        `save_path` must be a full path like 'checkpoints/cp_epoch10.pt' or 'checkpoints/best_model.pt'
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {  #add best_val_loss
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if hasattr(self, 'scaler') else None,
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_loss': self.best_loss,
            'lr': self.learning_rates,
            'train_indices': train_indices,
            'val_indices': val_indices
        }

        torch.save(checkpoint, save_path)
        #print(f"✅ Checkpoint saved to {save_path}")
    
    def resume(self, checkpoint_path):
        """
        Resume training from a saved checkpoint.
        `checkpoint_path` must be a full path like 'checkpoints/cp_epoch10.pt' or 'checkpoints/best_model.pt'
        """
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint file {checkpoint_path} does not exist. Cannot resume training.")
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_loss = checkpoint.get('best_loss', checkpoint.get('val_loss', 1.0))
        self.learning_rates = checkpoint.get('lr', [])
        print(f"✅ Resumed training from {checkpoint_path} at epoch {self.start_epoch}, last loss: {self.best_loss:.4f}")
        return checkpoint

    def training_step(self, batch): #perlu ditest: checkpointing dengan best_val_loss setelah resume
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
            # Ensure timesteps are on the same device as snr, kode nya bloated apa karena loss juga ya? 
            # snr = self.model.noise_scheduler.snr[timesteps.to(self.model.noise_scheduler.snr.device)].to(images.device)
            # loss = self.criterion(predicted_noise, target, timesteps, snr)
        
        predicted_noise = predicted_noise.float()
        target = target.float()
        snr = self.model.noise_scheduler.snr[timesteps.to(self.model.noise_scheduler.snr.device)].to(images.device).float()
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

    def validate(self, val_loader, epoch, train_indices=None, val_indices=None):
        """Validation loop for one epoch"""
        self.ema_model.apply_shadow()  # <<< Use EMA weights
        self.model.eval()
        total_val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images, text_prompts = batch
                device = next(self.model.parameters()).device
                images = images.to(device)

                timesteps = torch.randint(
                    0, self.model.noise_scheduler.num_train_timesteps,
                    (images.shape[0],), device=device
                )

                with autocast(device_type="cuda"):
                    predicted_noise, noise, latents = self.model(images, timesteps, text_prompts)
                    target = (
                        self.model.noise_scheduler.get_v_target(latents, noise, timesteps)
                        if self.use_v_param else noise
                    )
                    snr = self.model.noise_scheduler.snr[
                        timesteps.to(self.model.noise_scheduler.snr.device)
                    ].to(device)
                    loss = self.criterion(predicted_noise, target, timesteps, snr)

                total_val_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_val_loss / max(1, num_batches)
        self.val_loss_history.append(avg_val_loss)

        if avg_val_loss < self.best_loss:
            self.best_loss = avg_val_loss
            if self.config.Train.Checkpoint.enabled:
                best_cp_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                self.checkpoint(best_cp_path, epoch, avg_val_loss, train_indices, val_indices)
                print(f"🌟 Best model saved to {best_cp_path}")

        self.model.train()
        self.ema_model.restore()  # <<< Restore raw model weights
        return avg_val_loss
    
    def train(self, config, val_loader=None):
        device = next(self.model.parameters()).device
        #best_val_loss = self.best_loss
        #self.config = config or {}
        num_epochs = config.Train.num_epochs
        checkpoint_path = config.Train.Checkpoint.checkpoint_path
        self.checkpoint_dir = config.Train.Checkpoint.checkpoint_dir
        start_epoch = 1

        train_indices = None
        val_indices = None

        # Resume from checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path) and config.Train.Checkpoint.enabled:
            #checkpoint_path = os.path.abspath(checkpoint_path)
            print(f"📦 Resuming from checkpoint: {checkpoint_path}")
            checkpoint = self.resume(checkpoint_path)
            train_indices = checkpoint.get('train_indices')
            val_indices = checkpoint.get('val_indices')
            start_epoch = self.start_epoch
            print(f"🔄 Checkpoint loaded. Resuming from epoch {start_epoch}")
        else:
            print("🚀 Starting training from scratch.")

        # Create datasets and dataloaders inside train()
        train_dataset, val_dataset, train_indices, val_indices = create_datasets_with_indices(
            config, train_indices, val_indices
        )

        print(f"📊 Created train dataset with {len(train_dataset)} samples")
        if val_dataset:
            print(f"📊 Created val dataset with {len(val_dataset)} samples")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.Train.batch_size,
            shuffle=True,
            num_workers=config.Train.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        # Only use external val_loader if passed
        if val_loader is None and val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.Train.batch_size,
                shuffle=False,
                num_workers=config.Train.num_workers,
                pin_memory=True,
                collate_fn=collate_fn
            )

        # ============ Training Loop ============
        prev_cp_path = None  # Track previous checkpoint path for deletion
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n📘 Epoch {epoch}/{num_epochs}")
            self.model.train()
            epoch_losses = []

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
            for batch in progress_bar:
                stats = self.training_step(batch)
                epoch_losses.append(stats['loss'])

            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"📉 Avg Train Loss: {avg_train_loss:.6f}")
            print(f"📈 Learning Rate: {stats['lr']:.2e}")

            avg_val_loss = 0.0
            if val_loader:
                #self.model.eval()
                val_losses = []
                with torch.no_grad():
                    val_loss = self.validate(val_loader, epoch, train_indices, val_indices)
                    val_losses.append(val_loss)
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"🔎 Validation Loss: {avg_val_loss:.6f}")
            #Save best checkpoint if validation loss improves

            # Save checkpoint every epoch
            if self.checkpoint_dir and config.Train.Checkpoint.enabled:
                if epoch >= 0:
                    cp_path = os.path.join(self.checkpoint_dir, f"cp_epoch{epoch}.pt")
                    self.checkpoint(cp_path, epoch, avg_val_loss, train_indices, val_indices)
                    # Delete the previous checkpoint if it exists
                    if prev_cp_path and os.path.exists(prev_cp_path):
                        try:
                            os.remove(prev_cp_path)
                        except Exception as e:
                            print(f"Warning: Failed to delete previous checkpoint {prev_cp_path}: {e}")
                    print(f"✅ Checkpoint saved to {cp_path}")
                    prev_cp_path = cp_path  # Update to current checkpoint path

            # Early stopping
            if self.early_stop(avg_val_loss, self.model):
                print("⏹️ Early stopping triggered.")
                continue



