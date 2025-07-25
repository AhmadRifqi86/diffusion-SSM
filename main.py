import torch
from torch.utils.data import DataLoader
from models.diffuse import UShapeMambaDiffusion
from train.trainer import AdvancedDiffusionTrainer
from train.dataloader import create_datasets_with_indices
import os

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    return images, captions

def main():
    config = {
        'init_lr': 1e-7,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'num_epochs': 200,
        'optimizer': 'adamw',
        'batch_size': 4,
        'image_size': 256,
        'num_workers': 2,
        'patience': 20,
        'min_delta': 1e-4,
        'scheduler': 'cosine-decay',
        'T_0': 20,
        'warmup_ratio': 0.05,
        'decay_ratio': 0.2,
        'eta_min': 1e-6,
        'end_lr_factor': 0.1,
        'step_size': 50,
        'gamma': 0.5,
        'train_annotations': 'random',
        #'train_annotations': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/annotations/captions_train2017.json',
        'train_image_dir': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/train2017',
        #'val_annotations': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/annotations/captions_val2017.json',
        'val_annotations': 'val',
        'val_image_dir': '/home/arifadh/Desktop/Skripsi-Magang-Proyek/coco2017/val2017',
        'train_subset_size': 10000,
        'val_subset_size': 500,
        'enable_checkpointing': False,
        'checkpoint_dir': 'checkpoints_cosineDecay',
        'resume_from_checkpoint': None,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_indices = None
    val_indices = None
    if config['resume_from_checkpoint'] and os.path.exists(config['resume_from_checkpoint']):
        print(f"Loading dataset indices from checkpoint: {config['resume_from_checkpoint']}")
        checkpoint = torch.load(config['resume_from_checkpoint'], map_location='cpu')
        train_indices = checkpoint.get('train_indices')
        val_indices = checkpoint.get('val_indices')
        print(f"Restored train indices: {len(train_indices) if train_indices else 'None'}")
        print(f"Restored val indices: {len(val_indices) if val_indices else 'None'}")

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
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn
        )

    model = UShapeMambaDiffusion(
        vae_model_name="stabilityai/sd-vae-ft-mse",
        clip_model_name="openai/clip-vit-base-patch32"
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Using AdvancedDiffusionTrainer")

    trainer = AdvancedDiffusionTrainer(
        model,
        base_lr=config['learning_rate'],
        use_v_parameterization=True,
        checkpoint_dir=config['checkpoint_dir']
    )
    trainer.train(train_loader, val_loader, config=config)

if __name__ == "__main__":
    main()
