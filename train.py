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

def train_model(model, train_loader, val_loader, device, config):
    """Training loop with proper error handling and logging"""
    
    # Setup training components
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )
    
    scaler = GradScaler()
    
    # Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    best_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
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
                
                total_loss = total_loss + loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
                
            except Exception as e:
                #logger.error(f"Error in batch {batch_idx}: {e}")
                logging.error(f"Error in batch {batch_idx}: {e}\n{traceback.format_exc()}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        
        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total_loss = 0
            
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
                        
                    except Exception as e:
                        logger.error(f"Error in validation: {e}")
                        continue
            
            val_loss = val_total_loss / len(val_loader)
        
        # Logging
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f if val_loss else 'N/A'}")
        
        # Save checkpoint
        if val_loss and val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'config': config
            }, 'best_checkpoint.pt')
        
        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, f'checkpoint_epoch_{epoch+1}.pt')

def main():
    # Configuration
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 100,
        'batch_size': 2,
        'image_size': 256,
        'num_workers': 2,
        'train_annotations': 'path/to/coco/annotations.json',
        'train_image_dir': 'path/to/coco/images',
        'val_annotations': 'path/to/coco/val_annotations.json',
        'val_image_dir': 'path/to/coco/val_images',
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    # train_dataset = COCODataset(
    #     config['train_annotations'],
    #     config['train_image_dir'],
    #     config['image_size']
    # )train_dataset = DummyDataset(num_samples=1000, image_size=config['image_size'])


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
    #torch.autograd.set_detect_anomaly(True)
    # Train
    train_model(model, train_loader, val_loader, device, config)

if __name__ == "__main__":
    main() 