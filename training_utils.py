import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import json
from PIL import Image
import os
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import wandb
from torchvision import transforms
import numpy as np

class COCODiffusionDataset(Dataset):
    """COCO dataset for diffusion training"""
    def __init__(self, coco_annotations_path: str, image_dir: str, 
                 image_size: int = 512, max_caption_length: int = 77):
        with open(coco_annotations_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.image_dir = image_dir
        self.image_size = image_size
        self.max_caption_length = max_caption_length
        
        # Create image_id to captions mapping
        self.image_captions = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_captions:
                self.image_captions[img_id] = []
            self.image_captions[img_id].append(ann['caption'])
        
        # Filter images that have captions
        self.images = [img for img in self.coco_data['images'] 
                      if img['id'] in self.image_captions]
        
        # Transform for images
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print(f"Loaded {len(self.images)} images with captions")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder if image loading fails
            image = torch.randn(3, self.image_size, self.image_size)
        
        # Get random caption
        captions = self.image_captions[img_info['id']]
        caption = captions[torch.randint(0, len(captions), (1,)).item()]
        
        return {
            'image': image,
            'caption': caption,
            'image_id': img_info['id']
        }

class DiffusionTrainer:
    """Production-ready diffusion model trainer"""
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Training components
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  #biar bisa pake scheduler lain
            self.optimizer, 
            T_max=config['num_epochs']
        )
        
        # Mixed precision
        self.scaler = GradScaler() #biar bisa single precision training
        
        # Logging
        self.setup_logging()
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        
    def setup_logging(self):
        """Setup logging and wandb"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'diffusion-mamba'),
                config=self.config
            )
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            captions = batch['caption']
            
            # Sample random timesteps
            batch_size = images.shape[0]
            timesteps = torch.randint(
                0, self.model.noise_scheduler.num_train_timesteps, 
                (batch_size,), device=self.device
            )
            
            # Forward pass with mixed precision
            with autocast():
                predicted_noise, target_noise, _ = self.model(images, timesteps, captions)
                loss = F.mse_loss(predicted_noise, target_noise)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
            
            # Log to wandb
            if self.config.get('use_wandb', False) and batch_idx % 100 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                images = batch['image'].to(self.device)
                captions = batch['caption']
                
                batch_size = images.shape[0]
                timesteps = torch.randint(
                    0, self.model.noise_scheduler.num_train_timesteps, 
                    (batch_size,), device=self.device
                )
                
                with autocast():
                    predicted_noise, target_noise, _ = self.model(images, timesteps, captions)
                    loss = F.mse_loss(predicted_noise, target_noise)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, 'best_checkpoint.pt')
        
        # Keep only last N checkpoints
        if epoch > 10:
            old_checkpoint = f'checkpoint_epoch_{epoch-10}.pt'
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    
    def train(self, train_dataloader, val_dataloader=None):
        """Main training loop"""
        best_loss = float('inf')
        start_epoch = 0
        
        # Load checkpoint if exists
        if os.path.exists('best_checkpoint.pt'):
            start_epoch, best_loss = self.load_checkpoint('best_checkpoint.pt')
            logging.info(f"Resuming from epoch {start_epoch} with loss {best_loss}")
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            logging.info(f"Starting epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Training
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = None
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step()
            
            # Logging
            logging.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f if val_loss else 'N/A'}")
            
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss_epoch': train_loss,
                    'val_loss_epoch': val_loss if val_loss else 0,
                    'learning_rate_epoch': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_loss is not None and val_loss < best_loss
            if is_best:
                best_loss = val_loss
            
            self.save_checkpoint(epoch + 1, train_loss, is_best)
            
            # Generate sample images
            if epoch % 10 == 0:
                self.generate_samples(epoch)
    
    def generate_samples(self, epoch):
        """Generate sample images for monitoring"""
        self.model.eval()
        with torch.no_grad():
            sample_prompts = [
                "A beautiful sunset over mountains",
                "A cat sitting in a garden",
                "A futuristic city skyline",
                "A peaceful lake with trees"
            ]
            
            images = self.model.sample(
                sample_prompts, 
                num_inference_steps=20,
                height=512, 
                width=512
            )
            
            # Save or log images
            if self.config.get('use_wandb', False):
                for i, (prompt, image) in enumerate(zip(sample_prompts, images)):
                    wandb.log({
                        f"sample_image_{i}": wandb.Image(
                            image.cpu(), 
                            caption=prompt
                        )
                    })
            
            # Save to disk
            os.makedirs('samples', exist_ok=True)
            for i, (prompt, image) in enumerate(zip(sample_prompts, images)):
                # Convert to PIL and save
                image_pil = transforms.ToPILImage()(image.cpu())
                image_pil.save(f'samples/epoch_{epoch}_sample_{i}.png')

def create_dataloaders(config):
    """Create training and validation dataloaders"""
    # Training dataset
    train_dataset = COCODiffusionDataset(
        coco_annotations_path=config['train_annotations'],
        image_dir=config['train_image_dir'],
        image_size=config['image_size']
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    # Validation dataset (if provided)
    val_dataloader = None
    if config.get('val_annotations') and config.get('val_image_dir'):
        val_dataset = COCODiffusionDataset(
            coco_annotations_path=config['val_annotations'],
            image_dir=config['val_image_dir'],
            image_size=config['image_size']
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
    
    return train_dataloader, val_dataloader

# Example training configuration
TRAINING_CONFIG = {
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'num_epochs': 100,
    'batch_size': 4,
    'image_size': 512,
    'num_workers': 4,
    'train_annotations': 'path/to/coco/annotations.json',
    'train_image_dir': 'path/to/coco/images',
    'val_annotations': 'path/to/coco/val_annotations.json',
    'val_image_dir': 'path/to/coco/val_images',
    'use_wandb': True,
    'wandb_project': 'diffusion-mamba-coco'
} 