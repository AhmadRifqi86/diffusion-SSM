import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json

class ComprehensiveDiffusionMambaAnalyzer:
    """
    Unified analyzer for comprehensive diffusion-mamba model analysis and troubleshooting.
    Combines activation analysis, gradient flow, learning effectiveness, and training diagnostics.
    """
    
    def __init__(self, model, base_lr: float = 1e-5):
        self.model = model
        self.base_lr = base_lr
        
        # Tracking dictionaries
        self.metrics_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
        self.weight_importance = defaultdict(list)
        self.learning_effectiveness = defaultdict(list)
        self.activation_stats = defaultdict(list)
        self.loss_history = []
        self.timestep_losses = defaultdict(list)
        
        # Hook management
        self.hooks = []
        self.activations = {}
        self.prev_param_norms = {}
        
        # Analysis state
        self.step_count = 0
        self.analysis_interval = 50
        self.report_interval = 500
        
        # Learning rate scales for different components
        self.lr_scales = {
            'mamba_block': 1.8,
            'cross_attention': 1.5,
            'time_embed': 2.2,
            'scale_shift': 2.2,
            'context_proj': 0.7,
            'input_proj': 0.7,
            'output_proj': 0.8,
            'norm': 1.0,
            'other': 1.0
        }
    
    def register_hooks(self):
        """Register comprehensive hooks for activation and gradient analysis"""
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    output = output[0]  # Take first element if tuple
                self.activations[name] = output.detach()
                
                # Store additional activation statistics
                self._store_activation_stats(name, output.detach())
            return hook
        
        def get_gradient_hook(name):
            def hook(grad):
                self._store_gradient_stats(name, grad)
                return grad
            return hook
        
        # Hook different layer types
        layer_counts = defaultdict(int)
        
        for name, module in self.model.named_modules():
            # Mamba blocks
            if 'mamba' in name.lower() and hasattr(module, 'forward'):
                layer_name = f'mamba_block_{layer_counts["mamba"]}'
                handle = module.register_forward_hook(get_activation(layer_name))
                self.hooks.append(handle)
                layer_counts["mamba"] += 1
            
            # Cross-attention layers
            elif 'cross_attn' in name.lower() or 'cross_attention' in name.lower():
                layer_name = f'cross_attn_{layer_counts["cross_attn"]}'
                handle = module.register_forward_hook(get_activation(layer_name))
                self.hooks.append(handle)
                layer_counts["cross_attn"] += 1
            
            # Time embedding layers
            elif 'time' in name.lower() and 'embed' in name.lower():
                layer_name = f'time_embed_{layer_counts["time_embed"]}'
                handle = module.register_forward_hook(get_activation(layer_name))
                self.hooks.append(handle)
                layer_counts["time_embed"] += 1
        
        # Register gradient hooks for parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(get_gradient_hook(name))
    
    def _store_activation_stats(self, name: str, activation: torch.Tensor):
        """Store comprehensive activation statistics"""
        with torch.no_grad():
            # Basic statistics
            mean_act = activation.mean().item()
            std_act = activation.std().item()
            max_act = activation.max().item()
            min_act = activation.min().item()
            
            # Advanced statistics
            # 1. Dead neuron ratio (activations close to zero)
            dead_ratio = (torch.abs(activation) < 1e-6).float().mean().item()
            
            # 2. Saturation ratio (activations at extreme values)
            if activation.numel() > 0:
                sat_high = (activation > (mean_act + 3 * std_act)).float().mean().item()
                sat_low = (activation < (mean_act - 3 * std_act)).float().mean().item()
                saturation_ratio = sat_high + sat_low
            else:
                saturation_ratio = 0.0
            
            # 3. Activation entropy (diversity measure)
            act_flat = activation.flatten()
            if len(act_flat) > 1:
                act_probs = F.softmax(act_flat, dim=0)
                entropy = -(act_probs * torch.log(act_probs + 1e-8)).sum().item()
            else:
                entropy = 0.0
            
            # 4. Effective rank (measure of activation diversity)
            if activation.dim() >= 2:
                act_2d = activation.view(activation.shape[0], -1)
                try:
                    _, s, _ = torch.svd(act_2d)
                    s_normalized = s / s.sum()
                    effective_rank = torch.exp(-torch.sum(s_normalized * torch.log(s_normalized + 1e-8))).item()
                except:
                    effective_rank = 0.0
            else:
                effective_rank = 0.0
            
            stats = {
                'mean': mean_act,
                'std': std_act,
                'max': max_act,
                'min': min_act,
                'dead_ratio': dead_ratio,
                'saturation_ratio': saturation_ratio,
                'entropy': entropy,
                'effective_rank': effective_rank
            }
            
            for key, value in stats.items():
                self.activation_stats[f'{name}_{key}'].append(value)
    
    def _store_gradient_stats(self, param_name: str, grad: torch.Tensor):
        """Store comprehensive gradient statistics"""
        if grad is None:
            return
        
        with torch.no_grad():
            # Basic gradient statistics
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()
            
            # Advanced gradient metrics
            # 1. Gradient sparsity
            grad_sparsity = (torch.abs(grad) < 1e-8).float().mean().item()
            
            # 2. Gradient clipping indicator
            grad_max = grad.abs().max().item()
            
            # 3. Gradient signal-to-noise ratio
            grad_abs_mean = torch.abs(grad).mean().item()
            grad_snr = grad_abs_mean / (grad_std + 1e-8)
            
            block_name = self.categorize_parameter(param_name)
            
            # Store metrics
            metrics = {
                'grad_norm': grad_norm,
                'grad_mean': grad_mean,
                'grad_std': grad_std,
                'grad_sparsity': grad_sparsity,
                'grad_max': grad_max,
                'grad_snr': grad_snr
            }
            
            for key, value in metrics.items():
                self.gradient_history[f'{block_name}_{key}'].append(value)
    
    def analyze_step(self, loss: float, timesteps: Optional[torch.Tensor] = None):
        """Main analysis function to call at each training step"""
        self.step_count += 1
        self.loss_history.append(loss)
        
        # Store timestep-specific losses
        if timesteps is not None:
            for t in timesteps:
                self.timestep_losses[t.item()].append(loss)
        
        # Perform analysis at specified intervals
        if self.step_count % self.analysis_interval == 0:
            self._analyze_gradient_flow()
            self._analyze_learning_effectiveness()
            self._analyze_weight_importance()
            
            if self.activations:
                self._analyze_activation_patterns()
        
        # Generate comprehensive report
        if self.step_count % self.report_interval == 0:
            return self.generate_comprehensive_report()
        
        return None
    
    def _analyze_gradient_flow(self):
        """Analyze gradient flow through different components"""
        gradient_stats = defaultdict(lambda: defaultdict(list))
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                
                # Learning metrics
                grad_param_ratio = grad_norm / (param_norm + 1e-8)
                lr = self.get_effective_lr(name)
                update_magnitude = lr * grad_norm
                relative_update = update_magnitude / (param_norm + 1e-8)
                
                # Gradient health metrics
                grad_var = param.grad.var().item()
                grad_kurtosis = self._compute_kurtosis(param.grad)
                
                component = self.categorize_parameter(name)
                
                gradient_stats[component]['grad_norms'].append(grad_norm)
                gradient_stats[component]['param_norms'].append(param_norm)
                gradient_stats[component]['grad_param_ratios'].append(grad_param_ratio)
                gradient_stats[component]['update_magnitudes'].append(update_magnitude)
                gradient_stats[component]['relative_updates'].append(relative_update)
                gradient_stats[component]['grad_vars'].append(grad_var)
                gradient_stats[component]['grad_kurtosis'].append(grad_kurtosis)
        
        # Store aggregated statistics
        for component, stats in gradient_stats.items():
            for metric, values in stats.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    self.metrics_history[f'{component}_{metric}_mean'].append(mean_val)
                    self.metrics_history[f'{component}_{metric}_std'].append(std_val)
    
    def _analyze_learning_effectiveness(self):
        """Analyze how effectively each component is learning"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                component = self.categorize_parameter(name)
                current_norm = param.norm().item()
                
                # Track weight change rates
                if component in self.prev_param_norms:
                    weight_change = abs(current_norm - self.prev_param_norms[component])
                    lr = self.get_effective_lr(name)
                    lr_effectiveness = weight_change / (lr + 1e-8)
                    
                    self.learning_effectiveness[f'{component}_weight_change'].append(weight_change)
                    self.learning_effectiveness[f'{component}_lr_effectiveness'].append(lr_effectiveness)
                
                self.prev_param_norms[component] = current_norm
    
    def _analyze_weight_importance(self):
        """Analyze weight importance using multiple methods"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                component = self.categorize_parameter(name)
                
                # Multiple importance measures
                # 1. Gradient-Weight Product
                gwp = torch.abs(param * param.grad).mean().item()
                
                # 2. Fisher Information approximation
                fisher_info = (param.grad ** 2).mean().item()
                
                # 3. Weight magnitude statistics
                weight_l1 = torch.abs(param).mean().item()
                weight_l2 = param.norm().item()
                
                # 4. Parameter utilization (how spread out are the weights)
                param_entropy = self._compute_entropy(param)
                
                # 5. Gradient alignment (consistency of gradient direction)
                if f'{component}_prev_grad' in self.weight_importance:
                    prev_grad = self.weight_importance[f'{component}_prev_grad'][-1]
                    current_grad = param.grad.flatten()
                    if len(prev_grad) == len(current_grad):
                        grad_alignment = F.cosine_similarity(
                            prev_grad.unsqueeze(0), 
                            current_grad.unsqueeze(0)
                        ).item()
                        self.weight_importance[f'{component}_grad_alignment'].append(grad_alignment)
                
                # Store metrics
                self.weight_importance[f'{component}_gwp'].append(gwp)
                self.weight_importance[f'{component}_fisher'].append(fisher_info)
                self.weight_importance[f'{component}_weight_l1'].append(weight_l1)
                self.weight_importance[f'{component}_weight_l2'].append(weight_l2)
                self.weight_importance[f'{component}_param_entropy'].append(param_entropy)
                self.weight_importance[f'{component}_prev_grad'] = [param.grad.flatten().clone()]
    
    def _analyze_activation_patterns(self):
        """Analyze activation patterns for model health"""
        if not self.activations:
            return
        
        for name, activation in self.activations.items():
            # Skip if activation is None or empty
            if activation is None or activation.numel() == 0:
                continue
            
            # Pattern analysis
            # 1. Information flow (activation magnitude trends)
            act_magnitude = activation.norm().item()
            
            # 2. Feature diversity (how many features are active)  
            active_features = (torch.abs(activation) > activation.std() * 0.1).float().mean().item()
            
            # 3. Batch consistency (how consistent are activations across batch)
            if activation.dim() >= 2:
                batch_consistency = 1.0 - activation.std(dim=0).mean().item() / (activation.mean().abs().item() + 1e-8)
            else:
                batch_consistency = 1.0
            
            # Store activation pattern metrics
            self.activation_stats[f'{name}_magnitude'].append(act_magnitude)
            self.activation_stats[f'{name}_active_features'].append(active_features)
            self.activation_stats[f'{name}_batch_consistency'].append(batch_consistency)
    
    def analyze_timestep_bias(self):
        """Analyze bias towards certain timesteps"""
        if not self.timestep_losses:
            return {}
        
        timestep_avg_loss = {}
        for t, losses in self.timestep_losses.items():
            timestep_avg_loss[t] = np.mean(losses)
        
        if len(timestep_avg_loss) < 5:
            return timestep_avg_loss
        
        # Statistical analysis
        loss_values = list(timestep_avg_loss.values())
        mean_loss = np.mean(loss_values)
        std_loss = np.std(loss_values)
        
        # Identify problematic timesteps
        problematic_high = [(t, loss) for t, loss in timestep_avg_loss.items() 
                           if loss > mean_loss + 2 * std_loss]
        problematic_low = [(t, loss) for t, loss in timestep_avg_loss.items() 
                          if loss < mean_loss - 2 * std_loss]
        
        return {
            'timestep_avg_loss': timestep_avg_loss,
            'mean_loss': mean_loss,
            'std_loss': std_loss,
            'problematic_high': problematic_high,
            'problematic_low': problematic_low
        }
    
    def diagnose_training_health(self) -> Dict[str, Any]:
        """Comprehensive training health diagnosis"""
        diagnosis = {
            'gradient_health': {},
            'learning_effectiveness': {},
            'activation_health': {},
            'overall_status': 'unknown'
        }
        
        # Gradient health check
        for component in ['mamba_block', 'cross_attention', 'time_embed']:
            grad_norm_key = f'{component}_grad_norms_mean'
            if grad_norm_key in self.metrics_history and self.metrics_history[grad_norm_key]:
                recent_grad_norm = np.mean(self.metrics_history[grad_norm_key][-5:])
                
                if recent_grad_norm < 1e-7:
                    diagnosis['gradient_health'][component] = 'vanishing_gradients'
                elif recent_grad_norm > 1e-1:
                    diagnosis['gradient_health'][component] = 'exploding_gradients'
                else:
                    diagnosis['gradient_health'][component] = 'healthy'
        
        # Learning effectiveness check
        for component in ['mamba_block', 'cross_attention', 'time_embed']:
            effectiveness_key = f'{component}_lr_effectiveness'
            if effectiveness_key in self.learning_effectiveness and self.learning_effectiveness[effectiveness_key]:
                recent_effectiveness = np.mean(self.learning_effectiveness[effectiveness_key][-5:])
                
                if recent_effectiveness < 1e-6:
                    diagnosis['learning_effectiveness'][component] = 'poor_learning'
                elif recent_effectiveness > 1e-2:
                    diagnosis['learning_effectiveness'][component] = 'excessive_learning'
                else:
                    diagnosis['learning_effectiveness'][component] = 'good_learning'
        
        # Activation health check
        for name in self.activations.keys():
            dead_ratio_key = f'{name}_dead_ratio'
            if dead_ratio_key in self.activation_stats and self.activation_stats[dead_ratio_key]:
                recent_dead_ratio = np.mean(self.activation_stats[dead_ratio_key][-3:])
                
                if recent_dead_ratio > 0.5:
                    diagnosis['activation_health'][name] = 'dead_neurons'
                elif recent_dead_ratio > 0.2:
                    diagnosis['activation_health'][name] = 'some_dead_neurons'
                else:
                    diagnosis['activation_health'][name] = 'healthy_activations'
        
        # Overall status
        issues = []
        for category, status_dict in diagnosis.items():
            if category != 'overall_status':
                for component, status in status_dict.items():
                    if status in ['vanishing_gradients', 'exploding_gradients', 'poor_learning', 
                                'excessive_learning', 'dead_neurons']:
                        issues.append(f"{category}.{component}: {status}")
        
        if not issues:
            diagnosis['overall_status'] = 'healthy'
        elif len(issues) <= 2:
            diagnosis['overall_status'] = 'minor_issues'
        else:
            diagnosis['overall_status'] = 'needs_attention'
        
        diagnosis['issues'] = issues
        return diagnosis
    
    def suggest_optimizations(self) -> Dict[str, Any]:
        """Suggest specific optimizations based on analysis"""
        suggestions = {
            'learning_rates': {},
            'architecture': [],
            'training': [],
            'priority': 'medium'
        }
        
        diagnosis = self.diagnose_training_health()
        
        # Learning rate suggestions
        for component in ['mamba_block', 'cross_attention', 'time_embed']:
            current_scale = self.lr_scales.get(component, 1.0)
            
            # Check gradient health
            grad_status = diagnosis['gradient_health'].get(component, 'unknown')
            learning_status = diagnosis['learning_effectiveness'].get(component, 'unknown')
            
            if grad_status == 'vanishing_gradients' or learning_status == 'poor_learning':
                new_scale = min(current_scale * 2.0, 5.0)  # Cap at 5x
                suggestions['learning_rates'][component] = {
                    'current': current_scale,
                    'suggested': new_scale,
                    'reason': f'{grad_status} or {learning_status}'
                }
            elif grad_status == 'exploding_gradients' or learning_status == 'excessive_learning':
                new_scale = max(current_scale * 0.5, 0.1)  # Floor at 0.1x
                suggestions['learning_rates'][component] = {
                    'current': current_scale,
                    'suggested': new_scale,
                    'reason': f'{grad_status} or {learning_status}'
                }
        
        # Architecture suggestions
        dead_neuron_blocks = [name for name, status in diagnosis['activation_health'].items() 
                             if 'dead' in status]
        if dead_neuron_blocks:
            suggestions['architecture'].append(
                f"Consider adding residual connections or adjusting initialization for: {dead_neuron_blocks}"
            )
        
        # Training suggestions
        if len(diagnosis['issues']) > 3:
            suggestions['training'].append("Consider warmup period or different learning rate schedule")
            suggestions['priority'] = 'high'
        
        # Loss trend analysis
        if len(self.loss_history) > 100:
            recent_trend = np.polyfit(range(50), self.loss_history[-50:], 1)[0]
            if recent_trend > 0:  # Loss increasing
                suggestions['training'].append("Loss is increasing - consider lower learning rate or gradient clipping")
                suggestions['priority'] = 'high'
            elif abs(recent_trend) < 1e-6:  # Loss plateauing
                suggestions['training'].append("Loss plateauing - consider higher learning rate or learning rate scheduling")
        
        return suggestions
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE DIFFUSION-MAMBA ANALYSIS REPORT")
        report.append(f"Step: {self.step_count}")
        report.append("=" * 80)
        
        # Training health diagnosis
        diagnosis = self.diagnose_training_health()
        report.append(f"\nOVERALL HEALTH: {diagnosis['overall_status'].upper()}")
        
        if diagnosis['issues']:
            report.append(f"\nISSUES DETECTED ({len(diagnosis['issues'])}):")
            for issue in diagnosis['issues']:
                report.append(f"  ⚠️  {issue}")
        
        # Component-wise analysis
        report.append(f"\nCOMPONENT ANALYSIS:")
        for component in ['mamba_block', 'cross_attention', 'time_embed']:
            report.append(f"\n--- {component.upper().replace('_', ' ')} ---")
            
            # Gradient info
            grad_status = diagnosis['gradient_health'].get(component, 'unknown')
            learn_status = diagnosis['learning_effectiveness'].get(component, 'unknown')
            
            report.append(f"  Gradient Health: {grad_status}")
            report.append(f"  Learning Effectiveness: {learn_status}")
            
            # Recent metrics
            grad_norm_key = f'{component}_grad_norms_mean'
            if grad_norm_key in self.metrics_history and self.metrics_history[grad_norm_key]:
                recent_grad = self.metrics_history[grad_norm_key][-1]
                report.append(f"  Recent Gradient Norm: {recent_grad:.2e}")
        
        # Suggestions
        suggestions = self.suggest_optimizations()
        report.append(f"\nOPTIMIZATION SUGGESTIONS (Priority: {suggestions['priority'].upper()}):")
        
        if suggestions['learning_rates']:
            report.append("  Learning Rate Adjustments:")
            for component, lr_info in suggestions['learning_rates'].items():
                report.append(f"    {component}: {lr_info['current']:.2f} → {lr_info['suggested']:.2f} "
                            f"({lr_info['reason']})")
        
        for suggestion in suggestions['architecture'] + suggestions['training']:
            report.append(f"  • {suggestion}")
        
        # Timestep analysis
        timestep_analysis = self.analyze_timestep_bias()
        if timestep_analysis and 'problematic_high' in timestep_analysis:
            if timestep_analysis['problematic_high']:
                report.append(f"\nPROBLEMATIC TIMESTEPS (High Loss):")
                for t, loss in timestep_analysis['problematic_high'][:5]:
                    report.append(f"  Timestep {t}: {loss:.4f}")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def plot_training_metrics(self, save_path: Optional[str] = None):
        """Plot comprehensive training metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Loss history
        if self.loss_history:
            axes[0, 0].plot(self.loss_history)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
        
        # Plot 2: Gradient norms by component
        components = ['mamba_block', 'cross_attention', 'time_embed']
        for component in components:
            key = f'{component}_grad_norms_mean'
            if key in self.metrics_history:
                axes[0, 1].plot(self.metrics_history[key], label=component.replace('_', ' '))
        axes[0, 1].set_title('Gradient Norms by Component')
        axes[0, 1].set_xlabel('Analysis Step')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # Plot 3: Learning effectiveness
        for component in components:
            key = f'{component}_lr_effectiveness'
            if key in self.learning_effectiveness:
                axes[0, 2].plot(self.learning_effectiveness[key], label=component.replace('_', ' '))
        axes[0, 2].set_title('Learning Effectiveness')
        axes[0, 2].set_xlabel('Analysis Step')
        axes[0, 2].set_ylabel('LR Effectiveness')
        axes[0, 2].legend()
        axes[0, 2].set_yscale('log')
        
        # Plot 4: Weight importance (GWP)
        for component in components:
            key = f'{component}_gwp'
            if key in self.weight_importance:
                axes[1, 0].plot(self.weight_importance[key], label=component.replace('_', ' '))
        axes[1, 0].set_title('Weight Importance (GWP)')
        axes[1, 0].set_xlabel('Analysis Step')
        axes[1, 0].set_ylabel('Gradient-Weight Product')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        
        # Plot 5: Activation health (dead neuron ratio)
        activation_names = [name for name in self.activation_stats.keys() if '_dead_ratio' in name]
        for name in activation_names[:3]:  # Limit to first 3 for readability
            if name in self.activation_stats:
                clean_name = name.replace('_dead_ratio', '')
                axes[1, 1].plot(self.activation_stats[name], label=clean_name)
        axes[1, 1].set_title('Dead Neuron Ratio')
        axes[1, 1].set_xlabel('Analysis Step')
        axes[1, 1].set_ylabel('Dead Ratio')
        axes[1, 1].legend()
        
        # Plot 6: Timestep loss distribution
        timestep_analysis = self.analyze_timestep_bias()
        if timestep_analysis and 'timestep_avg_loss' in timestep_analysis:
            timesteps = list(timestep_analysis['timestep_avg_loss'].keys())
            losses = list(timestep_analysis['timestep_avg_loss'].values())
            axes[1, 2].scatter(timesteps, losses, alpha=0.6)
            axes[1, 2].set_title('Loss by Timestep')
            axes[1, 2].set_xlabel('Timestep')
            axes[1, 2].set_ylabel('Average Loss')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_metrics(self, filepath: str):
        """Export all metrics to JSON for external analysis"""
        export_data = {
            'step_count': self.step_count,
            'loss_history': self.loss_history,
            'metrics_history': dict(self.metrics_history),
            'gradient_history': dict(self.gradient_history),
            'weight_importance': dict(self.weight_importance),
            'learning_effectiveness': dict(self.learning_effectiveness),
            'activation_stats': dict(self.activation_stats),
            'diagnosis': self.diagnose_training_health(),
            'suggestions': self.suggest_optimizations()
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            else:
                return obj
        
        export_data = convert_arrays(export_data)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    # Utility methods
    def categorize_parameter(self, param_name: str) -> str:
        """Categorize parameters by component type"""
        name_lower = param_name.lower()
        if 'mamba' in name_lower:
            return 'mamba_block'
        elif 'cross_attn' in name_lower or 'attention' in name_lower:
            return 'cross_attention'
        elif 'time' in name_lower and 'embed' in name_lower:
            return 'time_embed'
        elif 'context' in name_lower:
            return 'context_proj'
        elif 'input' in name_lower:
            return 'input_proj'
        elif 'output' in name_lower:
            return 'output_proj'
        elif 'scale' in name_lower or 'shift' in name_lower:
            return 'scale_shift'
        elif 'norm' in name_lower:
            return 'norm'
        else:
            return 'other'
    
    def get_effective_lr(self, param_name: str) -> float:
        """Get effective learning rate for a parameter"""
        component = self.categorize_parameter(param_name)
        return self.base_lr * self.lr_scales.get(component, 1.0)
    
    def _compute_kurtosis(self, tensor: torch.Tensor) -> float:
        """Compute kurtosis (measure of tail heaviness)"""
        if tensor.numel() < 4:
            return 0.0
        
        with torch.no_grad():
            mean = tensor.mean()
            std = tensor.std()
            if std == 0:
                return 0.0
            
            normalized = (tensor - mean) / std
            kurtosis = torch.mean(normalized ** 4) - 3.0  # Excess kurtosis
            return kurtosis.item()
    
    def _compute_entropy(self, tensor: torch.Tensor) -> float:
        """Compute entropy of parameter distribution"""
        if tensor.numel() == 0:
            return 0.0
        
        with torch.no_grad():
            # Convert to probability distribution
            tensor_flat = tensor.flatten()
            # Use histogram to estimate distribution
            hist = torch.histc(tensor_flat, bins=50, min=tensor_flat.min(), max=tensor_flat.max())
            hist = hist / hist.sum()  # Normalize
            
            # Compute entropy
            entropy = -(hist * torch.log(hist + 1e-8)).sum()
            return entropy.item()
    
    def cleanup(self):
        """Remove all hooks and clear stored data"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}


# """
# HOW TO INTERPRET ANALYSIS RESULTS AND IMPROVE TRAINING:

# 1. GRADIENT HEALTH INDICATORS:
#    - Vanishing gradients (< 1e-7): Increase learning rate, check initialization
#    - Exploding gradients (> 1e-1): Decrease learning rate, add gradient clipping
#    - High gradient variance: Add gradient clipping, reduce batch size
#    - Low gradient SNR (< 1.0): Increase batch size, check data quality

# 2. LEARNING EFFECTIVENESS INDICATORS:
#    - Poor learning (lr_effectiveness < 1e-6): Increase learning rate significantly
#    - Excessive learning (lr_effectiveness > 1e-2): Decrease learning rate
#    - Inconsistent weight changes: Add weight decay, check optimizer settings

# 3. ACTIVATION HEALTH INDICATORS:
#    - High dead neuron ratio (> 0.5): 
#      * Change activation function (try Swish, GELU instead of ReLU)
#      * Adjust initialization (Xavier/Kaiming)
#      * Add residual connections
#    - High saturation ratio (> 0.3):
#      * Use activation functions with better gradients (Swish, GELU)
#      * Add batch normalization or layer normalization
#    - Low activation entropy:
#      * Model may be underutilizing capacity
#      * Consider wider layers or different architecture

# 4. COMPONENT-SPECIFIC ISSUES:

#    MAMBA BLOCKS:
#    - Poor gradient flow: Mamba blocks are complex; may need higher learning rates
#    - Dead activations: Check state initialization and hidden dimensions
#    - Low importance scores: May indicate poor integration with other components

#    CROSS-ATTENTION:
#    - Attention collapse: All attention on few tokens
#      * Add attention dropout
#      * Use different attention mechanisms
#    - Poor alignment with text: Cross-attention not learning text-image relationships
#      * Check text encoding quality
#      * Increase cross-attention learning rate

#    TIME EMBEDDING:
#    - Poor time conditioning: Model not learning time-dependent behavior
#      * Increase time embedding dimensions
#      * Use different time encoding (sinusoidal, learned)
#      * Higher learning rate for time components

# 5. TRAINING DYNAMICS:
#    - Loss plateauing: Increase learning rate or change schedule
#    - Loss oscillating: Decrease learning rate or add gradient clipping
#    - Timestep bias: Some timesteps much harder than others
#      * Use timestep-aware loss weighting
#      * Check noise schedule

# 6. OPTIMIZATION ACTIONS BASED ON PRIORITY:

#    HIGH PRIORITY (Immediate Action Required):
#    - Exploding/vanishing gradients → Adjust LR immediately
#    - Loss increasing → Stop training, diagnose issue
#    - >50% dead neurons → Change architecture/initialization

#    MEDIUM PRIORITY (Adjust in Next Training Run):
#    - Learning effectiveness issues → Tune learning rates
#    - Moderate activation problems → Adjust architecture
#    - Timestep bias → Modify loss weighting

#    LOW PRIORITY (Monitor and Optimize):
#    - Minor gradient inconsistencies
#    - Small dead neuron ratios
#    - Slight component imbalances

# 7. RECOMMENDED INTERVENTION SEQUENCE:
#    1. Fix gradient problems first (clipping, LR adjustment)
#    2. Address activation health (initialization, architecture)
#    3. Balance component learning rates
#    4. Fine-tune based on loss dynamics
#    5. Optimize for specific timestep performance

# 8. MONITORING BEST PRACTICES:
#    - Check analysis every 50-100 steps
#    - Generate full reports every 500-1000 steps
#    - Save metrics for long-term trend analysis
#    - Plot metrics regularly to catch issues early

# 9. COMMON PATTERNS TO WATCH FOR:
#    - Mamba blocks learning slower than attention → Increase mamba LR
#    - Time embedding not adapting → Check time encoding method
#    - Cross-attention dominating → Balance component learning rates
#    - Consistent timestep bias → Adjust noise schedule or loss weighting

# 10. EMERGENCY STOPS:
#     Stop training immediately if:
#     - Gradients explode beyond 1e2
#     - Loss increases for >100 steps
#     - >80% neurons become dead
#     - NaN values appear in any metric
# """