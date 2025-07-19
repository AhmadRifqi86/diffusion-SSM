import torch
torch.autograd.set_detect_anomaly(True)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import json
from PIL import Image
import os
from typing import List
import logging
import traceback
import random
from tqdm import tqdm
from torchvision import transforms
from model_v2 import UShapeMambaDiffusion
import numpy as np
from collections import deque


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, image_size=512):
        self.num_samples = num_samples
        self.image_size = image_size
        self.captions = [
            "A random caption.",
            "Another random caption.",
            "Yet another caption."
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)
        # Fix: Use Python's random instead of torch.randint to avoid device issues
        #caption = self.captions[random.randint(0, len(self.captions) - 1)]
        caption = random.choice(self.captions)
        return {'image': image, 'caption': caption}

class COCODataset(Dataset):
    def __init__(self, annotations_path, image_dir, image_size=512):
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        self.image_dir = image_dir
        self.image_size = image_size
        
        # Create image_id to captions mapping
        self.image_captions = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_captions:
                self.image_captions[img_id] = []
            self.image_captions[img_id].append(ann['caption'])
        
        # Filter images with captions
        self.images = [img for img in data['images'] 
                      if img['id'] in self.image_captions]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print(f"Loaded {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except:
            image = torch.randn(3, self.image_size, self.image_size)
        
        captions = self.image_captions[img_info['id']]
        #caption = captions[torch.randint(0, len(captions), (1,)).item()]
        caption = captions[torch.randint(0, len(captions), (1,), device='cpu').item()]
        
        return {'image': image, 'caption': caption}

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

def create_scheduler(optimizer, config):
    """Create scheduler based on config"""
    scheduler_type = config.get('scheduler', 'phase')
    
    if scheduler_type == 'phase':
        return PhaseScheduler(
            optimizer,
            config['num_epochs'],
            warmup_ratio=config.get('warmup_ratio', 0.2),
            decay_ratio=config.get('decay_ratio', 0.2)
        )
    elif scheduler_type == 'cosine':
        # Use PyTorch's built-in cosine annealing
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=config.get('eta_min', 0)
        )
    elif scheduler_type == 'linear':
        return LinearScheduler(
            optimizer,
            config['num_epochs'],
            warmup_epochs=config.get('warmup_epochs', 0),
            end_lr_factor=config.get('end_lr_factor', 0.1)
        )
    elif scheduler_type == 'step':
        # Step decay scheduler
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 50),
            gamma=config.get('gamma', 0.5)
        )
    elif scheduler_type == 'exponential':
        # Exponential decay scheduler
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get('gamma', 0.95)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def train_model(model, train_loader, val_loader, device, config):
    """Training loop with 3-phase schedule and early stopping"""
    
    # Setup training components
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Create scheduler based on config
    scheduler = create_scheduler(optimizer, config)
    
    scaler = GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('patience', 20),
        min_delta=config.get('min_delta', 1e-4)
    )
    
    # Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Training history
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # Loss smoothing for monitoring
    loss_window = deque(maxlen=50)
    
    best_loss = float('inf')
    
    # Log scheduler information
    scheduler_type = config.get('scheduler', 'phase')
    logger.info(f"Starting training with {config['num_epochs']} epochs")
    logger.info(f"Scheduler: {scheduler_type}")
    
    if scheduler_type == 'phase':
        logger.info(f"Phase 1 (Warmup): epochs 1-{scheduler.warmup_epochs}")
        logger.info(f"Phase 2 (Main): epochs {scheduler.warmup_epochs+1}-{scheduler.warmup_epochs+scheduler.main_epochs}")
        logger.info(f"Phase 3 (Decay): epochs {scheduler.warmup_epochs+scheduler.main_epochs+1}-{config['num_epochs']}")
    elif scheduler_type == 'cosine':
        logger.info(f"Cosine annealing with T_max={config['num_epochs']}, eta_min={config.get('eta_min', 0)}")
    elif scheduler_type == 'linear':
        logger.info(f"Linear decay with {scheduler.warmup_epochs} warmup epochs")
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Determine current phase for logging
        if scheduler_type == 'phase':
            if epoch < scheduler.warmup_epochs:
                phase = "Warmup"
            elif epoch < scheduler.warmup_epochs + scheduler.main_epochs:
                phase = "Main"
            else:
                phase = "Decay"
        else:
            phase = scheduler_type.title()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} ({phase})")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch['image'].to(device)
                captions = batch['caption']
                
                # Sample random timesteps
                batch_size = images.shape[0]
                timesteps = torch.randint(
                    0, model.noise_scheduler.num_train_timesteps, 
                    (batch_size,), device=device
                )
                
                # Forward pass with mixed precision
                with autocast():
                    predicted_noise, target_noise, _ = model(images, timesteps, captions)
                    loss = F.mse_loss(predicted_noise, target_noise)
                
                # Backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                num_batches += 1
                loss_window.append(loss.item())
                
                # Update progress bar with more info
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = total_loss / num_batches
                smoothed_loss = np.mean(loss_window) if loss_window else avg_loss
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'smoothed': f'{smoothed_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}\n{traceback.format_exc()}")
                continue
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        # Update learning rate
        if hasattr(scheduler, 'step') and callable(getattr(scheduler, 'step')):
            current_lr = scheduler.step()
        else:
            # For PyTorch built-in schedulers
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total_loss = 0
            val_num_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        images = batch['image'].to(device)
                        captions = batch['caption']
                        
                        batch_size = images.shape[0]
                        timesteps = torch.randint(
                            0, model.noise_scheduler.num_train_timesteps, 
                            (batch_size,), device=device
                        )
                        
                        with autocast():
                            predicted_noise, target_noise, _ = model(images, timesteps, captions)
                            loss = F.mse_loss(predicted_noise, target_noise)
                        
                        val_total_loss += loss.item()
                        val_num_batches += 1
                        
                    except Exception as e:
                        logger.error(f"Error in validation: {e}")
                        continue
            
            val_loss = val_total_loss / val_num_batches
            val_losses.append(val_loss)
        
        # Logging with phase information
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} ({phase}): "
                   f"Train Loss: {avg_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f if val_loss else 'N/A'}, "
                   f"LR: {current_lr:.2e}")
        
        # Save best checkpoint
        if val_loss and val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'loss': val_loss,
                'config': config,
                'phase': phase
            }, 'best_checkpoint.pt')
            logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
        
        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'loss': avg_loss,
                'val_loss': val_loss,
                'config': config,
                'phase': phase,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates
            }, f'checkpoint_epoch_{epoch+1}.pt')
        
        # Early stopping check
        if val_loss and early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            if early_stopping.restore_best_model(model):
                logger.info("Restored best model weights")
            break
    
    # Final logging
    logger.info(f"Training completed. Best validation loss: {best_loss:.4f}")
    
    # Save final training history
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_loss': best_loss,
        'final_epoch': len(train_losses)
    }, 'training_history.pt')
    
    return train_losses, val_losses, learning_rates

def main(): #test annotation nya gaada
    # Configuration with training schedule parameters
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 250,  # Increased for better training schedule
        'batch_size': 4,
        'image_size': 256,
        'num_workers': 2,
        'patience': 20,  # Early stopping patience
        'min_delta': 1e-4,  # Minimum improvement for early stopping
        
        # Scheduler configuration
        'scheduler': 'phase',  # Options: 'phase', 'cosine', 'linear', 'step', 'exponential'
        
        # Phase scheduler parameters
        'warmup_ratio': 0.2,  # 20% warmup for phase scheduler
        'decay_ratio': 0.2,   # 20% decay for phase scheduler
        
        # Cosine scheduler parameters (PyTorch built-in)
        'eta_min': 1e-6,      # Minimum learning rate for cosine annealing
        
        # Linear scheduler parameters
        'warmup_epochs': 50,  # Number of warmup epochs for linear
        'end_lr_factor': 0.1,  # Final LR = base_lr * end_lr_factor
        
        # Step scheduler parameters
        'step_size': 50,       # Step size for step scheduler
        'gamma': 0.5,          # Learning rate decay factor
        
        'train_annotations': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/annotations/captions_train2017.json',
        'train_image_dir': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/train2017',
        'val_annotations': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/annotations/captions_val2017.json',
        'val_image_dir': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/val2017',
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    # train_dataset = COCODataset(
    #     config['train_annotations'],
    #     config['train_image_dir'],
    #     config['image_size']
    # )
    train_dataset = DummyDataset(num_samples=800, image_size=config['image_size'])
    print("After create DummyDataset")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    # Validation dataset (optional)
    val_loader = None
    if os.path.exists(config['val_annotations']):
        val_dataset = COCODataset(
            config['val_annotations'],
            config['val_image_dir'],
            config['image_size']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
    
    # Create model
    model = UShapeMambaDiffusion(
        vae_model_name="stabilityai/sd-vae-ft-mse",
        clip_model_name="openai/clip-vit-base-patch32",
        use_openai_clip=False
    ).to(device)
    
    # Enable gradient checkpointing for memory efficiency
    #model.gradient_checkpointing_enable()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Using scheduler: {config['scheduler']}")
    #torch.autograd.set_detect_anomaly(True)
    # Train
    train_losses, val_losses, learning_rates = train_model(model, train_loader, val_loader, device, config)

if __name__ == "__main__":
    main() 