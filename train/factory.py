import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import torch.optim.lr_scheduler as torch_sched
import inspect
from tools.debug import debug_log
# Example: from your custom scheduler implementation
from train.custom import CosineAnnealingWarmRestartsWithDecay
from models.diffuse import UShapeMambaDiffusion
import os
from omegaconf import OmegaConf


OPTIMIZER_REGISTRY = {
    "adamw": optim.AdamW,
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    # Add more as needed
}

class OptimizerSchedulerFactory:
    @staticmethod
    def load_config(config):
        #print(f"🔍 Loading configuration from: {config}")
        if isinstance(config, str) and (config.endswith(".yaml") or config.endswith(".yml")):
            if not os.path.exists(config):
                raise FileNotFoundError(f"YAML config file not found: {config}")
            return OmegaConf.load(config)
        if isinstance(config, dict):
            return OmegaConf.create(config)
        if isinstance(config, OmegaConf):
            return config
        raise TypeError("Config must be a path to .yaml, a dict, or an OmegaConf object.")
    #staticmethod to create model
    @staticmethod
    def create_model(config):
        config = OptimizerSchedulerFactory.load_config(config)
        return UShapeMambaDiffusion(config).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    @staticmethod
    def get_amp_type(self,config):
        if config.Optimizer.get("Autocast", True):
            amp_dtype = config.Optimizer.get("amp_dtype", "fp16").lower()
            if amp_dtype == "fp16":
                mixed_precision_dtype = torch.float16
                scaler = GradScaler()
            elif amp_dtype == "bf16":
                mixed_precision_dtype = torch.bfloat16
                scaler = None  # no need for GradScaler
            else:
                raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")
        else:
            mixed_precision_dtype = torch.float32
            scaler = None
        
        return mixed_precision_dtype,scaler

    @staticmethod
    def create_advanced_optimizer(model, config):
        opt_cfg = config['Optimizer']
        opt_name = opt_cfg.get('name', 'adamw').lower()
        opt_params = opt_cfg.get(opt_name.capitalize(), {})  # e.g., 'Adamw'

        base_lr = opt_params.get('base_lr', 1e-4)
        weight_decay = opt_params.get('weight_decay', 0.01)
        print(f"Using optimizer: {opt_name.upper()}, base_lr: {base_lr:.2e}")

        param_groups = []
        used = set()

        def add_group(name, params, lr_mult=1.0, wd=weight_decay):
            filtered = [p for p in params if p.requires_grad and id(p) not in used]
            for p in filtered:
                used.add(id(p))
            if filtered:
                param_groups.append({
                    'params': filtered,
                    'lr': base_lr * lr_mult,
                    'weight_decay': wd,
                    'name': name
                })

        # Custom param groups
        for key, group_cfg in opt_cfg.get('ParamGroups', {}).items():
            lr_mult = group_cfg.get('lr_scale', 1.0)
            wd = group_cfg.get('weight_decay', weight_decay)
            for name, module in model.named_modules():
                if key in name or module.__class__.__name__ == key:
                    add_group(name, module.parameters(), lr_mult=lr_mult, wd=wd)
                    break

        # Optional unet grouping
        if hasattr(model, 'unet'):
            for i, block in enumerate(model.unet.down_blocks):
                add_group(f"unet.down_blocks.{i}", block.parameters(), lr_mult=(0.95 ** i))
            for i, block in enumerate(model.unet.up_blocks):
                add_group(f"unet.up_blocks.{i}", block.parameters(), lr_mult=(0.95 ** i))
            add_group("unet.middle_block", model.unet.middle_block.parameters(), lr_mult=0.7)

        # Catch-all fallback
        remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in used]
        if remaining:
            param_groups.append({'params': remaining, 'lr': base_lr, 'weight_decay': weight_decay, 'name': 'default'})

        # Instantiate optimizer from registry
        optimizer_cls = OPTIMIZER_REGISTRY.get(opt_name)
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # Filter allowed args to avoid unexpected kwargs
        kwargs = {k: v for k, v in opt_params.items() if k not in ['base_lr', 'weight_decay']}
        return optimizer_cls(param_groups, **kwargs)
    
    @staticmethod
    def create_advanced_scheduler(optimizer, config):
        #config = OptimizerSchedulerFactory.load_config(config)
        sequence = config['Scheduler'].get('sequence', [])
        if not sequence:
            raise ValueError("Scheduler sequence is not defined in config.")

        schedulers = []
        milestones = []
        total_iters_accum = 0
        
        for i, sched_cfg in enumerate(sequence):
            name = sched_cfg['name']
            sched_kwargs = {k: v for k, v in sched_cfg.items() if k != 'name'}
            total_iters_scale = config.Train.Dataset.train_subset / (config.Train.batch_size * config.Optimizer.grad_acc)

            scheduler_cls = globals().get(name, None)
            if scheduler_cls is None:
                scheduler_cls = getattr(torch.optim.lr_scheduler, os.name, None)
                if scheduler_cls is None:
                    raise ValueError(f"Unknown scheduler class: {name}")

            total_iters = sched_kwargs.get('total_iters', None)
            if total_iters is not None:
                total_iters = int(total_iters * total_iters_scale)
                sched_kwargs['total_iters'] = total_iters
            if 'T_0' in sched_kwargs:
                sched_kwargs['T_0'] = int(sched_kwargs['T_0'] * total_iters_scale)
                debug_log(f"T_0 : {sched_kwargs['T_0']}")
                
            constructor_args = inspect.signature(scheduler_cls.__init__).parameters
            debug_log(f"scheduler_name: {name} has total_iters: {total_iters} steps")

            if total_iters is not None:
                if 'total_iters' not in constructor_args:
                    sched_kwargs.pop('total_iters')  # use for milestone only
            else:
                if name == "LinearLR":
                    raise ValueError("LinearLR requires 'total_iters' in config.")

            scheduler = scheduler_cls(optimizer, **sched_kwargs)
            schedulers.append(scheduler)

            if 'start_factor' in constructor_args and 'start_factor' not in sched_kwargs:
                base_lr = config['Optimizer']['Adamw']['base_lr']
                init_lr = config['Optimizer'].get('init_lr', base_lr * 0.01)
                sched_kwargs['start_factor'] = init_lr / base_lr    

            if total_iters is not None and i < len(sequence) - 1:
                total_iters_accum += total_iters
                milestones.append(total_iters_accum)

        if len(schedulers) == 1:
            return schedulers[0]

        return SequentialLR(
            optimizer,
            schedulers=schedulers,
            milestones=milestones
        )


