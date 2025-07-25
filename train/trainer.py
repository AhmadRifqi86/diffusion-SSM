import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from train.custom import CosineAnnealingWarmRestartsWithDecay, EarlyStopping
from train.custom import EMAModel, MinSNRVLoss, GradientClipperWithNormTracking
from models.diffuse import UShapeMambaDiffusion
from models.blocks import CrossAttention
import logging
from train.factory import OptimizerSchedulerFactory
from tqdm import tqdm
import os


class AdvancedDiffusionTrainer:
    """
    Complete advanced training setup incorporating all improvements
    """
    def __init__(self, model: UShapeMambaDiffusion, config):
        self.model = model
        self.use_v_param = config.use_v_parameterization
        self.criterion = MinSNRVLoss(gamma=5.0)
        self.ema_model = EMAModel(model, decay=0.9999)

        # Multi-component optimizer
        self.early_stop = EarlyStopping(patience=20, min_delta=1e-4, restore_best_weights=True)
        
        self.optimizer = OptimizerSchedulerFactory.create_advanced_optimizer(self.model, self.config)
        self.scheduler = OptimizerSchedulerFactory.create_advanced_scheduler(self.optimizer, self.config)

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
        self.checkpoint_dir = config.checkpoint_dir #ganti jadi config aja
        self.logger = logging.getLogger(__name__)


    def checkpoint(self, save_path, epoch, val_loss):
        """
        Save a full training checkpoint at a specific path.
        `save_path` must be a full path like 'checkpoints/cp_epoch10.pt' or 'checkpoints/best_model.pt'
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {  #checkpointing early stopping and train_indices
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if hasattr(self, 'scaler') else None,
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'lr': self.learning_rates
        }

        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint saved to {save_path}")
    
    def resume(self, checkpoint_path):
        """
        Resume training from a saved checkpoint.
        `checkpoint_path` must be a full path like 'checkpoints/cp_epoch10.pt' or 'checkpoints/best_model.pt'
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        #self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('val_loss', float('inf'))
        self.learning_rates = checkpoint.get('lr', [])
        print(f"✅ Resumed training from {checkpoint_path} at epoch {self.start_epoch}, best loss: {self.best_loss:.4f}")

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
                    # Ensure timesteps are on the same device as snr
                    snr = self.model.noise_scheduler.snr[timesteps.to(self.model.noise_scheduler.snr.device)].to(images.device)
                    loss = self.criterion(predicted_noise, target, timesteps, snr)

                total_val_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_val_loss / max(1, num_batches)
        self.val_loss_history.append(avg_val_loss)

        if avg_val_loss < self.best_loss:
            self.best_loss = avg_val_loss

        self.model.train()
        return avg_val_loss
    
    def train(self, data_loader, val_dataloader, config):  #nanti param ini ganti config
        best_val_loss = float('inf')
        start_epoch = 0
        num_epochs = config.get('num_epochs', 1000)
        checkpoint_dir = config.get('checkpoint_dir', None)
        
        if config.get('resume_from_checkpoint'): #apa ganti pake checkpoint path aja?
            if os.path.exists(config['resume_from_checkpoint']):
                print(f"Resuming from checkpoint: {config['resume_from_checkpoint']}")
                self.resume(config['resume_from_checkpoint'])
                start_epoch = self.start_epoch
            else:
                print(f"Checkpoint {config['resume_from_checkpoint']} not found, starting fresh.")
        #print("len(data_loader):", len(data_loader))
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"📘 Epoch {epoch}/{num_epochs}")
            self.model.train()
            epoch_losses = []

            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                stats = self.training_step(batch)
                epoch_losses.append(stats['loss'])
                #print(f"Learning Rate: {stats['lr']:.2e}") #karena learning rate nya sekarang per-step, bukan per-epoch
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"📉 Avg Train Loss: {avg_train_loss:.2e}")
            # printing final learning rate for the epoch
            #print(f"Avg Train Loss: {avg_train_loss:.4f}")

            # Validation
            self.model.eval()
            val_losses = []
            avg_val_loss = 0.0
            if val_dataloader is not None:
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_loss = self.validate(val_batch)
                        val_losses.append(val_loss.item())

                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"🔎 Validation Loss: {avg_val_loss:.4f}")
    
            # Save per-epoch checkpoint
            if checkpoint_dir is not None:
                if epoch > 0:
                    prev_checkpoint_path = os.path.join(checkpoint_dir, f"cp_epoch{epoch-1}.pt")
                    os.remove(prev_checkpoint_path)

                epoch_cp_path = os.path.join(checkpoint_dir, f"cp_epoch{epoch}.pt")
                self.checkpoint(epoch_cp_path, epoch, avg_val_loss)
            else:
                print("No checkpoint directory specified, skipping checkpoint save.")
                # Save best checkpoint if improved
            # if avg_val_loss < best_val_loss:
            #     best_val_loss = avg_val_loss
            #     best_cp_path = os.path.join(checkpoint_dir, "best_model.pt")
            #     self.checkpoint(best_cp_path, epoch, avg_val_loss)# Early stopping

            if self.early_stop(avg_val_loss, self.model):
                print("⏹️ Early stopping triggered.")
                continue

