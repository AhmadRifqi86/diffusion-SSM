import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

class DiffusionMambaAnalyzer:
    def __init__(self, model):
        self.model = model
        self.metrics_history = defaultdict(list)
        self.hooks = []
        self.activations = {}
        
    def register_hooks(self):
        """Register hooks for activation analysis"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Hook mamba blocks
        for i, layer in enumerate(self.model.unet.layers):
            if hasattr(layer, 'main_block'):
                handle = layer.main_block.register_forward_hook(
                    get_activation(f'mamba_block_{i}')
                )
                self.hooks.append(handle)
        
        # Hook cross-attention layers
        for i, layer in enumerate(self.model.unet.layers):
            if hasattr(layer, 'cross_attn'):
                handle = layer.cross_attn.register_forward_hook(
                    get_activation(f'cross_attn_{i}')
                )
                self.hooks.append(handle)
    
    def analyze_gradient_flow(self):
        """Analyze gradient flow through different components"""
        gradient_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                
                # Categorize by component
                if 'mamba' in name.lower():
                    component = 'mamba'
                elif 'cross_attn' in name.lower() or 'attention' in name.lower():
                    component = 'cross_attention'
                elif 'time' in name.lower():
                    component = 'time_embed'
                elif 'context' in name.lower():
                    component = 'context_proj'
                else:
                    component = 'other'
                
                if component not in gradient_stats:
                    gradient_stats[component] = {'grad_norms': [], 'param_norms': []}
                
                gradient_stats[component]['grad_norms'].append(grad_norm)
                gradient_stats[component]['param_norms'].append(param_norm)
        
        # Compute statistics
        for component, stats in gradient_stats.items():
            mean_grad = np.mean(stats['grad_norms'])
            std_grad = np.std(stats['grad_norms'])
            mean_param = np.mean(stats['param_norms'])
            
            print(f"{component}: grad_norm={mean_grad:.6f}±{std_grad:.6f}, "
                  f"param_norm={mean_param:.6f}, "
                  f"grad/param_ratio={mean_grad/mean_param:.8f}")
            
            self.metrics_history[f'{component}_grad_norm'].append(mean_grad)
            self.metrics_history[f'{component}_grad_std'].append(std_grad)
    
    def analyze_mamba_states(self):
        """Analyze mamba state utilization and entropy"""
        if not self.activations:
            print("No activations recorded. Call register_hooks() first.")
            return
        
        mamba_stats = {}
        for name, activation in self.activations.items():
            if 'mamba' in name.lower():
                # Compute activation statistics
                mean_act = activation.mean().item()
                std_act = activation.std().item()
                
                # Compute entropy (measure of state utilization)
                act_probs = F.softmax(activation.flatten(), dim=0)
                entropy = -(act_probs * torch.log(act_probs + 1e-8)).sum().item()
                
                mamba_stats[name] = {
                    'mean': mean_act,
                    'std': std_act,
                    'entropy': entropy
                }
                
                print(f"{name}: mean={mean_act:.4f}, std={std_act:.4f}, entropy={entropy:.4f}")
        
        return mamba_stats
    
    def analyze_cross_attention_patterns(self):
        """Analyze cross-attention alignment quality"""
        attention_stats = {}
        
        for name, activation in self.activations.items():
            if 'cross_attn' in name.lower():
                # Simple attention pattern analysis
                attn_mean = activation.mean().item()
                attn_std = activation.std().item()
                
                # Check for attention collapse (all attention on one token)
                attn_max = activation.max().item()
                attn_min = activation.min().item()
                attention_range = attn_max - attn_min
                
                attention_stats[name] = {
                    'mean': attn_mean,
                    'std': attn_std,
                    'range': attention_range
                }
                
                print(f"{name}: mean={attn_mean:.4f}, std={attn_std:.4f}, range={attention_range:.4f}")
        
        return attention_stats
    
    def analyze_timestep_bias(self, timesteps, losses):
        """Analyze if model has bias towards certain timesteps"""
        timestep_loss_map = defaultdict(list)
        
        for t, loss in zip(timesteps, losses):
            timestep_loss_map[t.item()].append(loss.item())
        
        # Compute average loss per timestep
        timestep_avg_loss = {}
        for t, loss_list in timestep_loss_map.items():
            timestep_avg_loss[t] = np.mean(loss_list)
        
        # Plot timestep bias
        timesteps_sorted = sorted(timestep_avg_loss.keys())
        losses_sorted = [timestep_avg_loss[t] for t in timesteps_sorted]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timesteps_sorted, losses_sorted)
        plt.xlabel('Timestep')
        plt.ylabel('Average Loss')
        plt.title('Loss Distribution Across Timesteps')
        plt.show()
        
        # Check for problematic timesteps
        loss_values = list(timestep_avg_loss.values())
        mean_loss = np.mean(loss_values)
        std_loss = np.std(loss_values)
        
        problematic_timesteps = []
        for t, loss in timestep_avg_loss.items():
            if loss > mean_loss + 2 * std_loss:
                problematic_timesteps.append((t, loss))
        
        if problematic_timesteps:
            print("Problematic timesteps (high loss):")
            for t, loss in problematic_timesteps:
                print(f"  Timestep {t}: loss={loss:.4f}")
        
        return timestep_avg_loss
    
    def diagnose_capacity_bottleneck(self):
        """Diagnose if model has capacity bottlenecks"""
        print("\n=== CAPACITY BOTTLENECK DIAGNOSIS ===")
        
        # Check parameter utilization
        total_params = sum(p.numel() for p in self.model.parameters())
        learning_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Learning parameters: {learning_params:,}")
        print(f"Parameter utilization: {learning_params/total_params:.2%}")
        
        # Check gradient magnitudes
        component_grads = defaultdict(list)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if 'mamba' in name.lower():
                    component_grads['mamba'].append(param.grad.norm().item())
                elif 'cross_attn' in name.lower():
                    component_grads['cross_attn'].append(param.grad.norm().item())
        
        for component, grads in component_grads.items():
            mean_grad = np.mean(grads)
            if mean_grad < 1e-6:
                print(f"WARNING: {component} has very small gradients ({mean_grad:.2e})")
                print("  Possible capacity bottleneck or learning rate too low")
    
    def generate_training_report(self):
        """Generate comprehensive training analysis report"""
        print("\n" + "="*50)
        print("DIFFUSION-MAMBA TRAINING ANALYSIS REPORT")
        print("="*50)
        
        self.analyze_gradient_flow()
        print("\n" + "-"*30)
        self.analyze_mamba_states()
        print("\n" + "-"*30)
        self.analyze_cross_attention_patterns()
        print("\n" + "-"*30)
        self.diagnose_capacity_bottleneck()
        print("\n" + "="*50)
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# # Usage in your training loop:
# def enhanced_training_step(self, batch, analyzer=None):
#     """Enhanced training step with analysis"""
#     images, text_prompts = batch
#     timesteps = torch.randint(0, self.model.noise_scheduler.num_train_timesteps,
#                             (images.shape[0],), device=images.device)
    
#     with autocast(device_type="cuda"):
#         predicted_noise, noise, latents = self.model(images, timesteps, text_prompts)
#         target = self.model.noise_scheduler.get_v_target(latents, noise, timesteps) if self.use_v_param else noise
        
#         snr = self.model.noise_scheduler.snr[timesteps.to(self.model.noise_scheduler.snr.device)].to(images.device).float()
#         loss = self.criterion(predicted_noise, target, timesteps, snr)
#         loss = loss / self.gradient_accumulation_steps
    
#     self.scaler.scale(loss).backward()
    
#     # Analysis every 100 steps
#     if analyzer and (self.current_step % 100 == 0):
#         analyzer.analyze_gradient_flow()
#         analyzer.analyze_timestep_bias(timesteps, [loss.item()] * len(timesteps))
    
#     # Rest of your training step...
#     return {'loss': loss.item() * self.gradient_accumulation_steps}

# # Integration example:
# analyzer = DiffusionMambaAnalyzer(model)
# analyzer.register_hooks()

# # During training
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         metrics = enhanced_training_step(batch, analyzer)
        
#         # Generate report every 10 epochs
#         if epoch % 10 == 0 and batch_idx == 0:
#             analyzer.generate_training_report()


# import torch
# import torch.nn.functional as F
# import numpy as np
# from collections import defaultdict
# import matplotlib.pyplot as plt

class BlockAnalyzer:  #Low level per-block analyzer, track when val_loss is reaching somewhere between 0.2-0.4
    def __init__(self, model):
        self.model = model
        self.gradient_history = defaultdict(list)
        self.weight_importance = defaultdict(list)
        self.learning_effectiveness = defaultdict(list)
        self.activation_stats = defaultdict(list)
        self.hooks = []
        
    def track_gradient_metrics(self):
        """Track comprehensive gradient-based learning metrics"""
        block_metrics = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 1. Gradient Signal Strength
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                
                # 2. Gradient-to-Parameter Ratio (learning effectiveness)
                grad_param_ratio = grad_norm / (param_norm + 1e-8)
                
                # 3. Gradient Variance (stability indicator)
                grad_flat = param.grad.flatten()
                grad_var = torch.var(grad_flat).item()
                grad_mean = torch.mean(grad_flat).item()
                
                # 4. Weight Update Magnitude
                lr = self.get_effective_lr(name)
                update_magnitude = (lr * grad_norm)
                
                # 5. Signal-to-Noise Ratio in gradients
                grad_abs_mean = torch.mean(torch.abs(grad_flat)).item()
                snr = grad_abs_mean / (torch.std(grad_flat).item() + 1e-8)
                
                block_name = self.categorize_parameter(name)
                if block_name not in block_metrics:
                    block_metrics[block_name] = {
                        'grad_norms': [], 'param_norms': [], 'grad_param_ratios': [],
                        'grad_vars': [], 'update_magnitudes': [], 'snrs': []
                    }
                
                block_metrics[block_name]['grad_norms'].append(grad_norm)
                block_metrics[block_name]['param_norms'].append(param_norm)
                block_metrics[block_name]['grad_param_ratios'].append(grad_param_ratio)
                block_metrics[block_name]['grad_vars'].append(grad_var)
                block_metrics[block_name]['update_magnitudes'].append(update_magnitude)
                block_metrics[block_name]['snrs'].append(snr)
        
        # Analyze and store results
        for block_name, metrics in block_metrics.items():
            avg_grad_norm = np.mean(metrics['grad_norms'])
            avg_ratio = np.mean(metrics['grad_param_ratios'])
            avg_update = np.mean(metrics['update_magnitudes'])
            avg_snr = np.mean(metrics['snrs'])
            
            self.gradient_history[f'{block_name}_grad_norm'].append(avg_grad_norm)
            self.learning_effectiveness[f'{block_name}_ratio'].append(avg_ratio)
            
            # Diagnose learning issues
            if avg_grad_norm < 1e-6:
                print(f"⚠️  {block_name}: Very small gradients ({avg_grad_norm:.2e}) - may need higher LR")
            elif avg_grad_norm > 1e-2:
                print(f"⚠️  {block_name}: Large gradients ({avg_grad_norm:.2e}) - may need lower LR or clipping")
            
            if avg_ratio < 1e-5:
                print(f"⚠️  {block_name}: Poor learning effectiveness ({avg_ratio:.2e}) - weights barely changing")
            
            if avg_snr < 1.0:
                print(f"⚠️  {block_name}: Low gradient SNR ({avg_snr:.2f}) - noisy learning signal")
        
        return block_metrics
    
    def track_weight_importance(self):
        """Track weight importance using multiple methods"""
        importance_metrics = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Method 1: Gradient-Weight Product (GWP)
                # Indicates which weights have strong gradient signal
                gwp = torch.abs(param * param.grad).mean().item()
                
                # Method 2: Fisher Information approximation
                # Second moment of gradients
                fisher_diag = (param.grad ** 2).mean().item()
                
                # Method 3: Weight Magnitude Distribution
                weight_std = param.std().item() 
                weight_mean_abs = torch.abs(param).mean().item()
                
                # Method 4: Activation-based importance (if we have activations)
                # This requires forward hooks - simplified version
                activation_sensitivity = self.compute_activation_sensitivity(name, param)
                
                block_name = self.categorize_parameter(name)
                if block_name not in importance_metrics:
                    importance_metrics[block_name] = {
                        'gwp': [], 'fisher': [], 'weight_std': [], 
                        'weight_mean_abs': [], 'activation_sens': []
                    }
                
                importance_metrics[block_name]['gwp'].append(gwp)
                importance_metrics[block_name]['fisher'].append(fisher_diag)
                importance_metrics[block_name]['weight_std'].append(weight_std)
                importance_metrics[block_name]['weight_mean_abs'].append(weight_mean_abs)
                importance_metrics[block_name]['activation_sens'].append(activation_sensitivity)
        
        # Store aggregated importance scores
        for block_name, metrics in importance_metrics.items():
            avg_gwp = np.mean(metrics['gwp'])
            avg_fisher = np.mean(metrics['fisher'])
            
            self.weight_importance[f'{block_name}_gwp'].append(avg_gwp)
            self.weight_importance[f'{block_name}_fisher'].append(avg_fisher)
            
            print(f"{block_name}: GWP={avg_gwp:.6f}, Fisher={avg_fisher:.6f}")
        
        return importance_metrics
    
    def analyze_lr_weight_relationship(self, loss_history):
        """Analyze relationship between learning rate and weight changes"""
        lr_effects = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None and len(self.gradient_history) > 1:
                block_name = self.categorize_parameter(name)
                lr = self.get_effective_lr(name)
                
                # Calculate weight change rate
                current_norm = param.norm().item()
                if hasattr(self, 'prev_param_norms') and block_name in self.prev_param_norms:
                    weight_change = abs(current_norm - self.prev_param_norms[block_name])
                    
                    # LR effectiveness: weight_change per unit learning rate
                    lr_effectiveness = weight_change / (lr + 1e-8)
                    
                    if block_name not in lr_effects:
                        lr_effects[block_name] = []
                    lr_effects[block_name].append(lr_effectiveness)
                
                # Store current norms for next iteration
                if not hasattr(self, 'prev_param_norms'):
                    self.prev_param_norms = {}
                self.prev_param_norms[block_name] = current_norm
        
        return lr_effects
    
    def analyze_loss_weight_correlation(self, current_loss):
        """Analyze how loss changes correlate with weight changes"""
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) < 10:
            return {}
        
        correlations = {}
        recent_loss_trend = np.diff(self.loss_history[-10:])  # Last 10 loss changes
        
        for block_name in ['mamba_block', 'cross_attention', 'time_embed']:
            if f'{block_name}_grad_norm' in self.gradient_history:
                recent_grad_trend = np.diff(self.gradient_history[f'{block_name}_grad_norm'][-10:])
                
                if len(recent_grad_trend) == len(recent_loss_trend):
                    # Correlation between loss change and gradient change
                    correlation = np.corrcoef(recent_loss_trend, recent_grad_trend)[0, 1]
                    correlations[block_name] = correlation
                    
                    # Interpretation
                    if abs(correlation) > 0.7:
                        direction = "positively" if correlation > 0 else "negatively"
                        print(f"{block_name}: Strongly {direction} correlated with loss (r={correlation:.3f})")
                    elif abs(correlation) < 0.3:
                        print(f"{block_name}: Weakly correlated with loss (r={correlation:.3f}) - may need attention")
        
        return correlations
    
    def compute_activation_sensitivity(self, param_name, param):
        """Compute sensitivity based on activation patterns"""
        # Simplified version - in practice, you'd use forward hooks
        # This is a placeholder that returns gradient-based sensitivity
        if param.grad is not None:
            return torch.abs(param.grad).mean().item()
        return 0.0
    
    def categorize_parameter(self, param_name):
        """Categorize parameters by block type"""
        name_lower = param_name.lower()
        if 'mamba' in name_lower:
            return 'mamba_block'
        elif 'cross_attn' in name_lower or 'attention' in name_lower:
            return 'cross_attention'
        elif 'time' in name_lower:
            return 'time_embed'
        elif 'context' in name_lower:
            return 'context_proj'
        elif 'input' in name_lower:
            return 'input_proj'
        elif 'scale' in name_lower or 'shift' in name_lower:
            return 'scale_shift'
        else:
            return 'other'
    
    def get_effective_lr(self, param_name):
        """Get effective learning rate for a parameter"""
        # This should be implemented based on your optimizer setup
        # Placeholder implementation
        base_lr = 1e-5  # Your base learning rate
        
        # Apply param group scaling
        block_name = self.categorize_parameter(param_name)
        lr_scales = {
            'mamba_block': 1.8,
            'cross_attention': 1.5,
            'time_embed': 2.2,
            'scale_shift': 2.2,
            'context_proj': 0.7,
            'input_proj': 0.7
        }
        
        return base_lr * lr_scales.get(block_name, 1.0)
    
    def diagnose_block_health(self):
        """Comprehensive block health diagnosis"""
        print("\n" + "="*60)
        print("BLOCK HEALTH DIAGNOSIS")
        print("="*60)
        
        health_report = {}
        
        for block_name in ['mamba_block', 'cross_attention', 'time_embed', 'context_proj']:
            print(f"\n--- {block_name.upper()} ---")
            
            # Check gradient health
            if f'{block_name}_grad_norm' in self.gradient_history:
                recent_grads = self.gradient_history[f'{block_name}_grad_norm'][-5:]
                grad_trend = "increasing" if recent_grads[-1] > recent_grads[0] else "decreasing"
                grad_stability = np.std(recent_grads) / (np.mean(recent_grads) + 1e-8)
                
                print(f"  Gradient trend: {grad_trend}")
                print(f"  Gradient stability: {grad_stability:.3f} (lower is better)")
                
                # Diagnosis
                if grad_stability > 1.0:
                    print("  ⚠️  High gradient instability - consider lower LR or gradient clipping")
                elif np.mean(recent_grads) < 1e-6:
                    print("  ⚠️  Very small gradients - consider higher LR")
                else:
                    print("  ✅ Gradient health looks good")
            
            # Check learning effectiveness
            if f'{block_name}_ratio' in self.learning_effectiveness:
                recent_ratios = self.learning_effectiveness[f'{block_name}_ratio'][-5:]
                avg_ratio = np.mean(recent_ratios)
                
                print(f"  Learning effectiveness: {avg_ratio:.8f}")
                
                if avg_ratio < 1e-6:
                    print("  ⚠️  Poor learning effectiveness - weights barely changing")
                    health_report[block_name] = 'needs_higher_lr'
                elif avg_ratio > 1e-3:
                    print("  ⚠️  Very high learning rate - may be unstable")
                    health_report[block_name] = 'needs_lower_lr'
                else:
                    print("  ✅ Learning effectiveness looks good")
                    health_report[block_name] = 'healthy'
        
        return health_report
    
    def suggest_lr_adjustments(self):
        """Suggest learning rate adjustments based on analysis"""
        health_report = self.diagnose_block_health()
        suggestions = {}
        
        print(f"\n--- LEARNING RATE ADJUSTMENT SUGGESTIONS ---")
        
        for block_name, health_status in health_report.items():
            current_lr_scale = {
                'mamba_block': 1.8,
                'cross_attention': 1.5, 
                'time_embed': 2.2,
                'context_proj': 0.7
            }.get(block_name, 1.0)
            
            if health_status == 'needs_higher_lr':
                new_scale = current_lr_scale * 1.5
                suggestions[block_name] = new_scale
                print(f"  {block_name}: Increase lr_scale from {current_lr_scale} to {new_scale:.2f}")
            
            elif health_status == 'needs_lower_lr':
                new_scale = current_lr_scale * 0.7
                suggestions[block_name] = new_scale
                print(f"  {block_name}: Decrease lr_scale from {current_lr_scale} to {new_scale:.2f}")
            
            else:
                suggestions[block_name] = current_lr_scale
                print(f"  {block_name}: Keep current lr_scale at {current_lr_scale}")
        
        return suggestions


# # Usage in training loop:
# def enhanced_training_with_block_analysis(self, batch):
#     """Training step with comprehensive block analysis"""
#     # Your existing training code...
#     images, text_prompts = batch
#     # ... forward pass, loss computation ...
    
#     # Initialize analyzer if not exists
#     if not hasattr(self, 'block_analyzer'):
#         self.block_analyzer = BlockAnalyzer(self.model)
    
#     # Backward pass
#     loss.backward()
    
#     # Analyze every 50 steps
#     if self.current_step % 50 == 0:
#         # Track all metrics
#         grad_metrics = self.block_analyzer.track_gradient_metrics()
#         importance_metrics = self.block_analyzer.track_weight_importance()
#         lr_effects = self.block_analyzer.analyze_lr_weight_relationship(self.loss_history)
#         correlations = self.block_analyzer.analyze_loss_weight_correlation(loss.item())
        
#         # Generate suggestions every 500 steps
#         if self.current_step % 500 == 0:
#             suggestions = self.block_analyzer.suggest_lr_adjustments()
            
#             # Log to wandb or your preferred logger
#             import wandb
#             wandb.log({
#                 "block_analysis/suggestions": suggestions,
#                 "block_analysis/grad_metrics": grad_metrics,
#                 "block_analysis/correlations": correlations
#             })
    
#     # Continue with optimization step...
#     return {'loss': loss.item()}

# # Example of implementing suggestions:
# def update_optimizer_from_suggestions(optimizer, suggestions):
#     """Update optimizer learning rates based on analysis"""
#     for param_group in optimizer.param_groups:
#         group_name = param_group.get('name', 'default')
#         if group_name in suggestions:
#             old_lr = param_group['lr']
#             new_lr_scale = suggestions[group_name]
#             param_group['lr'] = param_group['base_lr'] * new_lr_scale
#             print(f"Updated {group_name} LR from {old_lr:.2e} to {param_group['lr']:.2e}")