import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import torch.optim.lr_scheduler as torch_sched
import inspect
# Example: from your custom scheduler implementation
from train.custom import CosineAnnealingWarmRestartsWithDecay
import os
from omegaconf import OmegaConf

class OptimizerSchedulerFactory:
    @staticmethod
    def load_config(config):
        if isinstance(config, str) and (config.endswith(".yaml") or config.endswith(".yml")):
            if not os.path.exists(config):
                raise FileNotFoundError(f"YAML config file not found: {config}")
            return OmegaConf.load(config)
        if isinstance(config, dict):
            return OmegaConf.create(config)
        if isinstance(config, OmegaConf):
            return config
        raise TypeError("Config must be a path to .yaml, a dict, or an OmegaConf object.")

    @staticmethod
    def create_advanced_optimizer(model, config):
        config = OptimizerSchedulerFactory.load_config(config)
        base_lr = config['Optimizer']['Adamw']['base_lr']
        param_groups = []
        used = set()

        def add_group(name, params, lr_mult=1.0, wd=0.01):
            filtered = [p for p in params if p.requires_grad and id(p) not in used]
            for p in filtered:
                used.add(id(p))
            if filtered:
                param_groups.append({
                    'params': filtered,
                    'lr': base_lr * lr_mult,
                    'weight_decay': wd
                })

        # Custom param groups
        for key, cfg in config['Optimizer']['ParamGroups'].items():
            lr_mult = cfg.get('lr_scale', 1.0)
            wd = cfg.get('weight_decay', 0.01)
            for name, module in model.named_modules():
                if key in name or module.__class__.__name__ == key:
                    add_group(name, module.parameters(), lr_mult=lr_mult, wd=wd)
                    break

        # Optional unet group
        if hasattr(model, 'unet'):
            for i, block in enumerate(model.unet.down_blocks):
                add_group(f"unet.down_blocks.{i}", block.parameters(), lr_mult=(0.95 ** i), wd=0.01)
            for i, block in enumerate(model.unet.up_blocks):
                add_group(f"unet.up_blocks.{i}", block.parameters(), lr_mult=(0.95 ** i), wd=0.01)
            add_group("unet.middle_block", model.unet.middle_block.parameters(), lr_mult=0.7, wd=0.01)

        # Catch-all fallback
        remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in used]
        if remaining:
            param_groups.append({'params': remaining, 'lr': base_lr, 'weight_decay': 0.01})

        return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    @staticmethod
    def create_advanced_scheduler(optimizer, config):
        config = OptimizerSchedulerFactory.load_config(config)
        sequence = config['Scheduler'].get('sequence', [])
        if not sequence:
            raise ValueError("Scheduler sequence is not defined in config.")

        schedulers = []
        milestones = []
        total_iters_accum = 0

        for i, sched_cfg in enumerate(sequence):
            name = sched_cfg['name']
            sched_kwargs = {k: v for k, v in sched_cfg.items() if k != 'name'}

            scheduler_cls = getattr(torch.optim.lr_scheduler, name, None)
            if scheduler_cls is None:
                scheduler_cls = globals().get(name, None)
            if scheduler_cls is None:
                raise ValueError(f"Unknown scheduler class: {name}")

            total_iters = sched_kwargs.get('total_iters', None)
            constructor_args = inspect.signature(scheduler_cls.__init__).parameters

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
                init_lr = config['Optimizer']['Adamw'].get('init_lr', base_lr * 0.01)
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

