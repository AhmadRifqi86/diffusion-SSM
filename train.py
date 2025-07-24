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
from custom import Lion, CosineAnnealingWarmRestartsWithDecay, LinearScheduler, PhaseScheduler, EarlyStopping

#pip install -e . --no-build-isolation   #Clone repo mamba-ssm habis itu pip install tanpa build isolation

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

def create_optimizer(params, config):
    """
    Create optimizer based on config. Supports 'adamw', 'lion', etc.
    """
    opt_type = config.get('optimizer', 'adamw').lower()
    lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    if opt_type == 'lion':
        betas = config.get('betas', (0.9, 0.99))
        return Lion(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt_type == 'adamw':
        betas = config.get('betas', (0.9, 0.999))
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


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
        print(f"sched_1 total_iters: {int(config.get('warmup_ratio', 0.1)*config.get('num_epochs'))}")
        sched_1 = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config['init_lr'] / config['learning_rate'],
            end_factor=1.0,
            total_iters=int(config.get('warmup_ratio', 0.1)*config.get('num_epochs'))
        )
        print(f"sched_2 starting from epoch: {int(config.get('warmup_ratio', 0.1)*config.get('num_epochs'))}")
        sched_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=config.get('eta_min', 0)
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[sched_1, sched_2],
            #milestones=[config.get('warmup_epochs', 10)]
            milestones=[int(config.get('warmup_epochs', 10)*config.get('num_epochs'))]
        )

    elif scheduler_type == 'linear': #maybe change this to reduceLRonPlateau
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
    elif scheduler_type == 'cosine-decay':
        print(f"sched_1 total_iters: {int(config.get('warmup_ratio', 0.1)*config.get('num_epochs'))}")
        sched_1 = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config['init_lr'] / config['learning_rate'],
            end_factor=1.0,
            total_iters=int(config.get('warmup_ratio', 0.1)*config.get('num_epochs'))
        )
        print(f"sched_2 starting from epoch: {int(config.get('warmup_ratio', 0.1)*config.get('num_epochs'))}")
        sched_2 = CosineAnnealingWarmRestartsWithDecay(
            optimizer,
            T_0=config.get('T_0', 20),
            T_mult=config.get('T_mult', 1),
            eta_min=config.get('eta_min', 1e-6),
            decay=config.get('decay', 0.9),
            freq_mult=config.get('freq_mult', 1.0)
        )
        #print(f"milestones: {int(config.get('warmup_epochs', 10)*config.get('num_epochs'))}")
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[sched_1, sched_2],
            #milestones=[config.get('warmup_epochs', 10)]
            milestones=[1+int(config.get('warmup_ratio', 0.1)*config.get('num_epochs'))]
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def train_model(model, train_loader, val_loader, device, config, train_indices=None, val_indices=None):
    """Training loop with 3-phase schedule and early stopping"""
    
    # Setup training components, make configurable
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.get('learning_rate',1e-7),
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
    
    # Checkpointing setup
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    enable_checkpointing = config.get('enable_checkpointing', True)
    resume_from_checkpoint = config.get('resume_from_checkpoint', None)
    
    # Resume training if checkpoint exists
    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        # Load checkpoint to CPU first to avoid GPU memory spike
        checkpoint = torch.load(resume_from_checkpoint, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Handle different scheduler types
        if checkpoint.get('scheduler_state_dict') is not None:
            if hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif hasattr(scheduler, 'current_epoch'):
                # For custom schedulers
                scheduler.current_epoch = checkpoint.get('scheduler_epoch', 0)
        else:
            # For custom schedulers without state_dict, restore current_epoch
            if hasattr(scheduler, 'current_epoch'):
                scheduler.current_epoch = checkpoint.get('scheduler_epoch', 0)
        
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        learning_rates = checkpoint.get('learning_rates', [])
        best_loss = checkpoint.get('best_loss', float('inf'))
        
        # Restore early stopping state
        if 'early_stopping_state' in checkpoint:
            early_stopping.best_loss = checkpoint['early_stopping_state']['best_loss']
            early_stopping.counter = checkpoint['early_stopping_state']['counter']
            early_stopping.best_weights = checkpoint['early_stopping_state']['best_weights']
        
        # Restore dataset indices
        if 'train_indices' in checkpoint:
            train_indices = checkpoint['train_indices']
            val_indices = checkpoint['val_indices']
        
        logger.info(f"Resumed from epoch {start_epoch}")
        logger.info(f"Previous best loss: {best_loss:.4f}")
    
    # Log scheduler information
    scheduler_type = config.get('scheduler', 'phase')
    logger.info(f"Starting training with {config['num_epochs']} epochs")
    logger.info(f"Scheduler: {scheduler_type}")
    logger.info(f"Checkpointing: {'Enabled' if enable_checkpointing else 'Disabled'}")
    
    if scheduler_type == 'phase':
        logger.info(f"Phase 1 (Warmup): epochs 1-{scheduler.warmup_epochs}")
        logger.info(f"Phase 2 (Main): epochs {scheduler.warmup_epochs+1}-{scheduler.warmup_epochs+scheduler.main_epochs}")
        logger.info(f"Phase 3 (Decay): epochs {scheduler.warmup_epochs+scheduler.main_epochs+1}-{config['num_epochs']}")
    elif scheduler_type == 'cosine':
        logger.info(f"Cosine annealing with T_max={config['num_epochs']}, eta_min={config.get('eta_min', 0)}")
    elif scheduler_type == 'linear':
        logger.info(f"Linear decay with {scheduler.warmup_epochs} warmup epochs")
    elif scheduler_type == 'cosine-decay':
        logger.info(f"Custom Cosine Annealing with T_0={config.get('T_0', 10)}, decay={config.get('decay', 0.9)}")
    
    for epoch in range(start_epoch, config['num_epochs']):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #mungkin ganti jadi 0.5
                
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
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} ({phase}): "
                   f"Train Loss: {avg_loss:.4f}, "
                   f"Val Loss: {val_loss_str}, "
                   f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best checkpoint
        if val_loss and val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'scheduler_epoch': getattr(scheduler, 'current_epoch', None),
                'scaler_state_dict': scaler.state_dict(),
                'loss': val_loss,
                'config': config,
                'phase': phase,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
                'best_loss': best_loss,
                'early_stopping_state': {
                    'best_loss': early_stopping.best_loss,
                    'counter': early_stopping.counter,
                    'best_weights': early_stopping.best_weights
                },
                'train_indices': train_indices,
                'val_indices': val_indices
            }, os.path.join(checkpoint_dir, 'best_checkpoint.pt'))
            logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
        
        # Save checkpoint after every epoch if enabled
        if enable_checkpointing:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            
            # Delete previous epoch's checkpoint file if it exists
            if epoch > 0:  # Don't delete on first epoch
                prev_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
                if os.path.exists(prev_checkpoint_path):
                    os.remove(prev_checkpoint_path)
                    logger.info(f"Deleted previous checkpoint: {prev_checkpoint_path}")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'scheduler_epoch': getattr(scheduler, 'current_epoch', None),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
                'val_loss': val_loss,
                'config': config,
                'phase': phase,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
                'best_loss': best_loss,
                'early_stopping_state': {
                    'best_loss': early_stopping.best_loss,
                    'counter': early_stopping.counter,
                    'best_weights': early_stopping.best_weights
                },
                'train_indices': train_indices,
                'val_indices': val_indices
            }, checkpoint_path)
            
            # Save latest checkpoint for easy resuming
            latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
            
            # Delete previous latest checkpoint if it exists
            if os.path.exists(latest_checkpoint_path):
                os.remove(latest_checkpoint_path)
                
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'scheduler_epoch': getattr(scheduler, 'current_epoch', None),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
                'val_loss': val_loss,
                'config': config,
                'phase': phase,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
                'best_loss': best_loss,
                'early_stopping_state': {
                    'best_loss': early_stopping.best_loss,
                    'counter': early_stopping.counter,
                    'best_weights': early_stopping.best_weights
                },
                'train_indices': train_indices,
                'val_indices': val_indices
            }, latest_checkpoint_path)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Regular checkpoint every 10 epochs (if not already saved)
        # elif (epoch + 1) % 10 == 0:
        #     checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            
        #     # Delete previous epoch-specific checkpoint if it exists
        #     if os.path.exists(checkpoint_path):
        #         os.remove(checkpoint_path)
                
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        #         'scheduler_epoch': getattr(scheduler, 'current_epoch', None),
        #         'scaler_state_dict': scaler.state_dict(),
        #         'loss': avg_loss,
        #         'val_loss': val_loss,
        #         'config': config,
        #         'phase': phase,
        #         'train_losses': train_losses,
        #         'val_losses': val_losses,
        #         'learning_rates': learning_rates,
        #         'best_loss': best_loss,
        #         'early_stopping_state': {
        #             'best_loss': early_stopping.best_loss,
        #             'counter': early_stopping.counter,
        #             'best_weights': early_stopping.best_weights
        #         },
        #         'train_indices': train_indices,
        #         'val_indices': val_indices
        #     }, checkpoint_path)
        
        # Early stopping check
        if val_loss and early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            if early_stopping.restore_best_model(model):
                logger.info("Restored best model weights")
            continue
    
    # Final logging
    logger.info(f"Training completed. Best validation loss: {best_loss:.4f}")
    
    # Save final training history
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_loss': best_loss,
        'final_epoch': len(train_losses)
    }, os.path.join(checkpoint_dir, 'training_history.pt'))
    
    return train_losses, val_losses, learning_rates

def create_datasets_with_indices(config, train_indices=None, val_indices=None):
    """Create datasets with optional indices for subsetting"""
    train_dataset = None
    val_dataset = None
    
    # Create training dataset
    if os.path.exists(config['train_annotations']):
        full_train_dataset = COCODataset(
            config['train_annotations'],
            config['train_image_dir'],
            config['image_size']
        )
        
        if train_indices is not None:
            # Use provided indices (for resuming)
            train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        elif config.get('train_subset_size') is not None:
            # Create new random subset
            subset_size = min(config['train_subset_size'], len(full_train_dataset))
            train_indices = random.sample(range(len(full_train_dataset)), subset_size)
            train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        else:
            # Use full dataset
            train_dataset = full_train_dataset
    else:
        train_dataset = DummyDataset(num_samples=40, image_size=config['image_size'])
        train_indices = None
    
    # Create validation dataset
    if os.path.exists(config['val_annotations']):
        full_val_dataset = COCODataset(
            config['val_annotations'],
            config['val_image_dir'],
            config['image_size']
        )
        
        if val_indices is not None:
            # Use provided indices (for resuming)
            val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)
        elif config.get('val_subset_size') is not None:
            # Create new random subset
            subset_size = min(config['val_subset_size'], len(full_val_dataset))
            val_indices = random.sample(range(len(full_val_dataset)), subset_size)
            val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)
        else:
            # Use full dataset
            val_dataset = full_val_dataset
    else:
        val_dataset = None
        val_indices = None
    
    return train_dataset, val_dataset, train_indices, val_indices

def main(): #test annotation nya gaada
    # Configuration with training schedule parameters
    config = {
        'init_lr': 1e-7,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'num_epochs': 200,  # Increased for better training schedule
        #'warmup_epochs': 0.15,  # Warmup phase for scheduler
        'optimizer': 'adamw',  # Options: 'adamw', 'lion'
        'batch_size': 8,
        'image_size': 256,
        'num_workers': 2,
        'patience': 20,  # Early stopping patience
        'min_delta': 1e-4,  # Minimum improvement for early stopping
        
        # Scheduler configuration
        'scheduler': 'cosine-decay',  # Options: 'phase', 'cosine', 'linear', 'step', 'exponential'
        'T_0': 20,
        # Phase scheduler parameters
        'warmup_ratio': 0.05,  # 10% warmup for phase scheduler
        'decay_ratio': 0.2,   # 20% decay for phase scheduler
        
        # Cosine scheduler parameters (PyTorch built-in)
        'eta_min': 1e-6,      # Minimum learning rate for cosine annealing
        
        # Linear scheduler parameters
        #'warmup_epochs': 0.2,  # Number of warmup epochs for linear
        'end_lr_factor': 0.1,  # Final LR = base_lr * end_lr_factor
        
        # Step scheduler parameters
        'step_size': 50,       # Step size for step scheduler
        'gamma': 0.5,          # Learning rate decay factor
        
        'train_annotations': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/annotations/captions_train2017.json',
        #'train_annotations': 'random',
        'train_image_dir': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/train2017',
        'val_annotations': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/annotations/captions_val2017.json',
        #'val_annotations': 'val',
        'val_image_dir': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/val2017',
        # Subset sizes (set to None for full dataset)
        'train_subset_size': 10000,
        'val_subset_size': 500,
        
        # Checkpointing configuration
        'enable_checkpointing': True,
        #'enable_checkpointing': False,
        'checkpoint_dir': 'checkpoints_cosineDecay', #Ganti jadi checkpoints_phasesched, checkpoints_customlr_1, 
        #'resume_from_checkpoint':None,
        'resume_from_checkpoint': 'checkpoints_cosineDecay/checkpoint_epoch_94.pt', # Set to path of checkpoint to resume from
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if resuming from checkpoint
    train_indices = None
    val_indices = None
    if config['resume_from_checkpoint'] and os.path.exists(config['resume_from_checkpoint']):
        print(f"Loading dataset indices from checkpoint: {config['resume_from_checkpoint']}")
        checkpoint = torch.load(config['resume_from_checkpoint'], map_location='cpu')
        train_indices = checkpoint.get('train_indices')
        val_indices = checkpoint.get('val_indices')
        print(f"Restored train indices: {len(train_indices) if train_indices else 'None'}")
        print(f"Restored val indices: {len(val_indices) if val_indices else 'None'}")
    
    # Create datasets
    train_dataset, val_dataset, train_indices, val_indices = create_datasets_with_indices(
        config, train_indices, val_indices
    )
    
    print(f"Created train dataset with {len(train_dataset)} samples")
    if val_dataset:
        print(f"Created val dataset with {len(val_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Validation dataset (optional)
    val_loader = None
    if val_dataset is not None:
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
        clip_model_name="openai/clip-vit-base-patch32"
    ).to(device)
    
    # Enable gradient checkpointing for memory efficiency
    #model.gradient_checkpointing_enable()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Using scheduler: {config['scheduler']}")
    #torch.autograd.set_detect_anomaly(True)
    # Train
    train_losses, val_losses, learning_rates = train_model(model, train_loader, val_loader, device, config, train_indices, val_indices)

    # Test the trained model: generate a few samples
    print("\nTesting the trained model with sample text prompts...")
    test_prompts = [
        "A beautiful landscape with mountains",
        "A futuristic city at night",
        "A cat sitting on a sofa"
    ]
    model.eval()
    with torch.no_grad():
        generated_images = model.sample(test_prompts, num_inference_steps=20)
    print(f"Generated images shape: {generated_images.shape}")
    # Optionally, save or visualize the images here
    # Example: save the first image using torchvision.utils.save_image if torchvision is available
    try:
        from torchvision.utils import save_image
        for i, img in enumerate(generated_images):
            save_image((img.clamp(-1, 1) + 1) / 2, f"generated_sample_{i}.png")
        print("Sample images saved as generated_sample_*.png")
    except ImportError:
        print("torchvision not available, skipping image saving.")

if __name__ == "__main__":
    main() 



#Pas warmup pake linearLR, LR nya berubah dari 1e-7 ke 1.05e-7, sehingga pas epoch 40, lr nya bukan 5e-5, experiment using smaller decay and other 
# learning rate scheduler
#Next to do: Pake EMA (exponential moving average) untuk model, bisa pake ema.py dari mamba-ssm
#And using DDIMSampler, Considering using RMSProp


# config = {
#         'init_lr': 1e-7,
#         'learning_rate': 5e-5,
#         'weight_decay': 0.01,
#         'num_epochs': 200,  # Increased for better training schedule
#         #'warmup_epochs': 0.15,  # Warmup phase for scheduler
#         'optimizer': 'adamw',  # Options: 'adamw', 'lion'
#         'batch_size': 8,
#         'image_size': 256,
#         'num_workers': 2,
#         'patience': 20,  # Early stopping patience
#         'min_delta': 1e-4,  # Minimum improvement for early stopping
        
#         # Scheduler configuration
#         'scheduler': 'cosine-decay',  # Options: 'phase', 'cosine', 'linear', 'step', 'exponential'
#         'T_0': 20,
#         # Phase scheduler parameters
#         'warmup_ratio': 0.05,  # 10% warmup for phase scheduler
#         'decay_ratio': 0.2,   # 20% decay for phase scheduler
        
#         # Cosine scheduler parameters (PyTorch built-in)
#         'eta_min': 1e-6,      # Minimum learning rate for cosine annealing
        
#         # Linear scheduler parameters
#         'warmup_epochs': 0.2,  # Number of warmup epochs for linear
#         'end_lr_factor': 0.1,  # Final LR = base_lr * end_lr_factor
        
#         # Step scheduler parameters
#         'step_size': 50,       # Step size for step scheduler
#         'gamma': 0.5,          # Learning rate decay factor
        
#         'train_annotations': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/annotations/captions_train2017.json',
#         'train_image_dir': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/train2017',
#         'val_annotations': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/annotations/captions_val2017.json',
#         'val_image_dir': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/val2017',
#         # Subset sizes (set to None for full dataset)
#         'train_subset_size': 10000,
#         'val_subset_size': 500,
        
#         # Checkpointing configuration
#         'enable_checkpointing': True,
#         'checkpoint_dir': 'checkpoints_cosineDecay', #Ganti jadi checkpoints_phasesched, checkpoints_customlr_1, 
#         'resume_from_checkpoint': 'checkpoints_cosineDecay/checkpoint_epoch_38.pt', # Set to path of checkpoint to resume from
#     }