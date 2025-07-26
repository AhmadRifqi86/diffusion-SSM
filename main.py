import torch
from models.diffuse import UShapeMambaDiffusion
from train.trainer import AdvancedDiffusionTrainer
from train.dataloader import create_datasets_with_indices
import os
from train.factory import OptimizerSchedulerFactory



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔌 Using device: {device}")
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    #print(f"📂 Loading configuration from: {config_path}")
    config = OptimizerSchedulerFactory.load_config(config_path)
    model = OptimizerSchedulerFactory.create_model(config_path)

    print(f"📦 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("🚀 Using AdvancedDiffusionTrainer")

    # Create trainer
    trainer = AdvancedDiffusionTrainer(model,config=config)

    # Let trainer handle dataset restoration and checkpoint logic
    trainer.train(config=config)


# def main():

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     train_indices = None
#     val_indices = None
#     if config['resume_from_checkpoint'] and os.path.exists(config['resume_from_checkpoint']):
#         print(f"Loading dataset indices from checkpoint: {config['resume_from_checkpoint']}")
#         checkpoint = torch.load(config['resume_from_checkpoint'], map_location='cpu')
#         train_indices = checkpoint.get('train_indices')
#         val_indices = checkpoint.get('val_indices')
#         print(f"Restored train indices: {len(train_indices) if train_indices else 'None'}")
#         print(f"Restored val indices: {len(val_indices) if val_indices else 'None'}")

#     train_dataset, val_dataset, train_indices, val_indices = create_datasets_with_indices(
#         config, train_indices, val_indices
#     )

#     print(f"Created train dataset with {len(train_dataset)} samples")
#     if val_dataset:
#         print(f"Created val dataset with {len(val_dataset)} samples")

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         num_workers=config['num_workers'],
#         pin_memory=True,
#         collate_fn=collate_fn
#     )

#     val_loader = None
#     if val_dataset is not None:
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=config['batch_size'],
#             shuffle=False,
#             num_workers=config['num_workers'],
#             pin_memory=True,
#             collate_fn=collate_fn
#         )

#     model = UShapeMambaDiffusion(
#         vae_model_name="stabilityai/sd-vae-ft-mse",
#         clip_model_name="openai/clip-vit-base-patch32"
#     ).to(device)

#     print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
#     print(f"Using AdvancedDiffusionTrainer")

#     trainer = AdvancedDiffusionTrainer(
#         model,
#         base_lr=config['learning_rate'],
#         use_v_parameterization=True,
#         checkpoint_dir=config['checkpoint_dir']
#     )
#     trainer.train(train_loader, val_loader, config=config)

if __name__ == "__main__":
    main()
