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

if __name__ == "__main__":
    main()
