
import torch
import torch.utils.data as data
import random
import json
from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import Dataset, DataLoader

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