import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import numpy as np

# Import pre-trained models
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
#import clip


#TODO: Misah-misahin model definition jadi beberapa bagian: Block (mamba, cross-attn dan scaleshift), ushape(timestep embedding, downblock, upblock, ushape), pretrained

class ParallelScan:  
    """
    Implements parallel scan (prefix sum) operation for Mamba's selective scan
    Based on the parallel scan algorithm from the Mamba paper
    """
    
    @staticmethod
    def parallel_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
        """
        Parallel scan in log space for numerical stability
        
        Args:
            log_coeffs: Log of coefficients [batch, length, dim]
            log_values: Log of values [batch, length, dim]
            
        Returns:
            Scanned values in log space
        """
        batch_size, seq_len, dim = log_coeffs.shape
        
        # For very short sequences, use sequential scan
        if seq_len <= 4:
            return ParallelScan._sequential_scan_log(log_coeffs, log_values)
        
        # Number of parallel scan levels
        levels = int(np.ceil(np.log2(seq_len)))
        
        # Work in log space for numerical stability
        coeffs = log_coeffs.clone()
        values = log_values.clone()
        
        # Up-sweep (reduce) phase
        for level in range(levels):
            step = 2 ** (level + 1)
            if step - 1 >= seq_len:
                break
                
            indices = torch.arange(step - 1, seq_len, step, device=log_coeffs.device)
            
            if len(indices) > 0:
                left_idx = indices - 2 ** level
                right_idx = indices
                
                valid_mask = (left_idx >= 0) & (right_idx < seq_len)
                if valid_mask.any():
                    left_idx = left_idx[valid_mask]
                    right_idx = right_idx[valid_mask]
                    
                    new_coeffs = coeffs[:, left_idx] + coeffs[:, right_idx]
                    new_values = torch.logaddexp(
                        values[:, left_idx],
                        coeffs[:, right_idx] + values[:, right_idx]
                    )
                    
                    coeffs[:, right_idx] = new_coeffs
                    values[:, right_idx] = new_values
        
        # Down-sweep (distribute) phase
        for level in range(levels - 1, -1, -1):
            step = 2 ** (level + 1)
            if step - 1 >= seq_len:
                continue
                
            indices = torch.arange(step - 1, seq_len, step, device=log_coeffs.device)
            
            if len(indices) > 0:
                left_idx = indices - 2 ** level
                right_idx = indices
                
                valid_mask = (left_idx >= 0) & (right_idx < seq_len)
                if valid_mask.any():
                    left_idx = left_idx[valid_mask]
                    right_idx = right_idx[valid_mask]
                    
                    temp_values = values[:, right_idx].clone()
                    values[:, right_idx] = torch.logaddexp(
                        values[:, left_idx],
                        coeffs[:, right_idx] + values[:, right_idx]
                    )
                    values[:, left_idx] = temp_values
        
        return values
    
    @staticmethod
    def _sequential_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
        """Sequential scan for short sequences"""
        batch_size, seq_len, dim = log_coeffs.shape
        scanned_values = log_values.clone()
        
        for b in range(batch_size):
            for d in range(dim):
                running_sum = scanned_values[b, 0, d]
                for i in range(1, seq_len):
                    running_sum = torch.logaddexp(
                        running_sum,
                        log_coeffs[b, i, d] + scanned_values[b, i, d]
                    )
                    scanned_values[b, i, d] = running_sum
        
        return scanned_values

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model core module
    Implements the selective scan mechanism from Mamba
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # Selective scan parameters
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.rand(self.d_inner, d_state)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Activation
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # Convolution for local dependencies
        x_conv = self.conv1d(x_inner.transpose(-1, -2))[..., :seq_len].transpose(-1, -2)
        x_conv = self.act(x_conv)
        
        # Selective scan parameters
        x_dbl = self.x_proj(x_conv)
        delta, B, C = torch.split(x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)
        
        # Delta (time step) processing
        delta = F.softplus(self.dt_proj(delta))
        
        # Selective scan
        y = self.selective_scan(x_conv, delta, B, C)
        
        # Skip connection and output projection
        y = y * self.act(z)
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(self, x: torch.Tensor, delta: torch.Tensor, 
                      B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Perform selective scan using parallel scan algorithm
        """
        batch_size, seq_len, d_inner = x.shape
        
        # Get A matrix
        A = -torch.exp(self.A_log.float())
        
        # Discretize
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, x)
        
        # Parallel scan in log space for numerical stability
        log_coeffs = torch.log(deltaA + 1e-12)
        log_values = torch.log(torch.abs(deltaB_u) + 1e-12)
        
        # Reshape for parallel scan
        log_coeffs = log_coeffs.view(batch_size, seq_len, -1)
        log_values = log_values.view(batch_size, seq_len, -1)
        
        # Apply parallel scan
        scanned_log_values = ParallelScan.parallel_scan_log(log_coeffs, log_values)
        
        # Reshape back
        scanned_values = torch.exp(scanned_log_values).view(batch_size, seq_len, d_inner, self.d_state)
        
        # Apply C matrix and D skip connection
        y = torch.einsum('bldn,bln->bld', scanned_values, C)
        y = y + x * self.D
        
        return y

class MambaBlock(nn.Module):
    """
    Enhanced Mamba block with proper structure from the diagram
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.ssm = SelectiveSSM(dim, d_state, d_conv, expand)
        self.linear_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        # Layer norm first
        x_norm = self.norm(x)
        # Selective scan
        x_ssm = self.ssm(x_norm)
        # Final linear layer
        x_out = self.linear_out(x_ssm)
        return x_out

class CrossAttention(nn.Module):
    """
    Cross-attention module for conditioning on text embeddings
    """
    def __init__(self, dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, context=None):
        b, n, _ = x.shape
        h = self.heads
        
        # If no context, use self-attention
        if context is None:
            context = x
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = q.view(b, n, h, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, h, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, h, self.dim_head).transpose(1, 2)
        
        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        
        return self.to_out(out)

class ScaleShift(nn.Module):
    """
    Scale and Shift module for adaptive conditioning, now supports optional timestep embedding.
    """
    def __init__(self, dim, context_dim):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim, dim * 2)
        )
        self.to_scale_shift_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim * 2, dim * 2)
        )

    def forward(self, x, context, timestep_emb=None):
        if timestep_emb is not None:
            # Concatenate context and timestep embedding
            cond = torch.cat([context, timestep_emb], dim=-1)
            scale_shift = self.to_scale_shift_time(cond)
        else:
            scale_shift = self.to_scale_shift(context)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return x * (scale + 1) + shift

class MainBlockParallel(nn.Module):
    """
    Main block implementing the architecture from the second image, now supports optional timestep embedding.
    """
    def __init__(self, dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        self.dim = dim
        self.cross_attn = CrossAttention(dim, context_dim, heads, dim_head)
        self.scale_shift_1 = ScaleShift(dim, context_dim)
        self.norm_1 = nn.LayerNorm(dim)
        self.mamba_block = MambaBlock(dim)
        self.scale_shift_2 = ScaleShift(dim, context_dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.scale_1 = nn.Parameter(torch.ones(1))
        self.scale_2 = nn.Parameter(torch.ones(1))

    def forward(self, x, context, timestep_emb=None):
        residual = x
        # If timestep_emb is None, add it to input before block
        # if timestep_emb is None:
        #     x = x + 0  # No-op, but placeholder for logic if needed
        # First path: Cross Attention
        x_attn = self.cross_attn(x, context)
        x_attn = self.scale_shift_1(x_attn, context.mean(dim=1), timestep_emb)
        x_attn = self.norm_1(x_attn)
        x_attn = x_attn * self.scale_1
        # Second path: Mamba Block
        x_mamba = self.mamba_block(x)
        x_mamba = self.scale_shift_2(x_mamba, context.mean(dim=1), timestep_emb)
        x_mamba = self.norm_2(x_mamba)
        x_mamba = x_mamba * self.scale_2
        output = residual + x_attn + x_mamba
        return output

class MainBlockSerial(nn.Module):
    """
    Main block implementing the architecture from the second image, now supports optional timestep embedding.
    """
    def __init__(self, dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        self.dim = dim
        self.cross_attn = CrossAttention(dim, context_dim, heads, dim_head)
        self.scale_shift_1 = ScaleShift(dim, context_dim)
        self.norm_1 = nn.LayerNorm(dim)
        self.mamba_block = MambaBlock(dim)
        self.scale_shift_2 = ScaleShift(dim, context_dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.scale_1 = nn.Parameter(torch.ones(1))
        self.scale_2 = nn.Parameter(torch.ones(1))

    def forward(self, x, context, timestep_emb=None):
        residual_1 = x
        # if timestep_emb is None:
        #     x = x + 0  # No-op, but placeholder for logic if needed
        x_mamba = self.mamba_block(x)
        x_mamba = self.scale_shift_1(x_mamba, context.mean(dim=1), timestep_emb)
        x_mamba = self.norm_1(x_mamba)
        x_mamba = x_mamba * self.scale_1
        attn_inp = x_mamba + residual_1
        x_attn = self.cross_attn(attn_inp, context)
        x_attn = self.scale_shift_2(x_attn, context.mean(dim=1), timestep_emb)
        x_attn = self.norm_2(x_attn)
        x_attn = x_attn * self.scale_2
        output = x_attn + attn_inp
        return output

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, context_dim, time_in_mainblock=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.to_main_block = nn.Linear(out_channels, out_channels)
        self.main_block = MainBlockSerial(out_channels, context_dim)
        self.from_main_block = nn.Linear(out_channels, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
        self.time_in_mainblock = time_in_mainblock

    def forward(self, x, t, context=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        time_emb = self.time_mlp(t)
        if not self.time_in_mainblock:
            h = h + time_emb[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        if context is not None:
            B, C, H, W = h.shape
            h_flat = h.flatten(2).transpose(1, 2)
            h_flat = self.to_main_block(h_flat)
            if self.time_in_mainblock:
                h_flat = self.main_block(h_flat, context, time_emb)
            else:
                h_flat = self.main_block(h_flat, context)
            h_flat = self.from_main_block(h_flat)
            h = h_flat.transpose(1, 2).reshape(B, C, H, W)
        return self.downsample(h), h

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, context_dim, time_in_mainblock=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.to_main_block = nn.Linear(out_channels, out_channels)
        self.main_block = MainBlockSerial(out_channels, context_dim)
        self.from_main_block = nn.Linear(out_channels, out_channels)
        self.time_in_mainblock = time_in_mainblock

    def forward(self, x, skip, t, context=None):
        h = self.upsample(x)
        h = torch.cat([h, skip], dim=1)
        h = self.conv1(h)
        h = self.norm1(h)
        h = F.silu(h)
        time_emb = self.time_mlp(t)
        if not self.time_in_mainblock:
            h = h + time_emb[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        if context is not None:
            B, C, H, W = h.shape
            h_flat = h.flatten(2).transpose(1, 2)
            h_flat = self.to_main_block(h_flat)
            if self.time_in_mainblock:
                h_flat = self.main_block(h_flat, context, time_emb)
            else:
                h_flat = self.main_block(h_flat, context)
            h_flat = self.from_main_block(h_flat)
            h = h_flat.transpose(1, 2).reshape(B, C, H, W)
        return h

class MiddleBlock(nn.Module):
    def __init__(self, channels, time_dim, context_dim, time_in_mainblock=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels)
        )
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
        # Main block processing
        self.to_main_block = nn.Linear(channels, channels)
        self.main_block = MainBlockSerial(channels, context_dim)
        self.from_main_block = nn.Linear(channels, channels)
        self.time_in_mainblock = time_in_mainblock
        
    def forward(self, x, t, context=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t)
        if not self.time_in_mainblock:
            h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # Apply Main Block
        if context is not None:
            B, C, H, W = h.shape
            h_flat = h.flatten(2).transpose(1, 2)
            h_flat = self.to_main_block(h_flat)
            if self.time_in_mainblock:
                h_flat = self.main_block(h_flat, context, time_emb)
            else:
                h_flat = self.main_block(h_flat, context)
            h_flat = self.from_main_block(h_flat)
            h = h_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return h

class NoiseScheduler:
    """Simple linear noise scheduler for DDPM"""
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Pre-compute values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, original_samples, timesteps):
        """Add noise to samples according to timesteps"""
        device = timesteps.device  # 🔥 fix
        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)

        noise = torch.randn_like(original_samples)
        noisy_samples = sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise

        return noisy_samples, noise

    def step(self, model_output, timesteps, sample):
        """Single denoising step"""
        device = timesteps.device  # 🔥 fix
        alpha = self.alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)
        alpha_prev = self.alphas_cumprod_prev.to(device)[timesteps].view(-1, 1, 1, 1)
        beta = self.betas.to(device)[timesteps].view(-1, 1, 1, 1)

        pred_original_sample = (sample - torch.sqrt(1 - alpha) * model_output) / torch.sqrt(alpha)
        pred_sample_direction = torch.sqrt(1 - alpha_prev) * model_output
        pred_prev_sample = torch.sqrt(alpha_prev) * pred_original_sample + pred_sample_direction

        return pred_prev_sample


class UShapeMamba(nn.Module):
    def __init__(self, 
                 in_channels=4,  # Stable Diffusion VAE latent channels
                 model_channels=320,
                 time_embed_dim=1280,
                 context_dim=768):  # CLIP text encoder dimension
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimestepEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Context projection for CLIP embeddings
        self.context_proj = nn.Linear(context_dim, context_dim)
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Down blocks
        self.down_blocks = nn.ModuleList([
            DownBlock(model_channels, model_channels, time_embed_dim, context_dim),
            DownBlock(model_channels, model_channels * 2, time_embed_dim, context_dim),
            DownBlock(model_channels * 2, model_channels * 4, time_embed_dim, context_dim),
            DownBlock(model_channels * 4, model_channels * 8, time_embed_dim, context_dim),
        ])
        
        # Middle block
        self.middle_block = MiddleBlock(model_channels * 8, time_embed_dim, context_dim)
        
        # Up blocks
        self.up_blocks = nn.ModuleList([
            UpBlock(model_channels * 8, model_channels * 4, time_embed_dim, context_dim),
            UpBlock(model_channels * 4, model_channels * 2, time_embed_dim, context_dim),
            UpBlock(model_channels * 2, model_channels, time_embed_dim, context_dim),
            UpBlock(model_channels, model_channels, time_embed_dim, context_dim),
        ])
        
        # Output projection
        self.output_proj = nn.Conv2d(model_channels, in_channels, 3, padding=1)
        
    def forward(self, x, timesteps, context=None):
        # Time embedding
        t = self.time_embed(timesteps)
        
        # Process context
        if context is not None:
            context = self.context_proj(context)
            if context.dim() == 2:  # If context is [batch, dim], expand to [batch, seq_len, dim]
                context = context.unsqueeze(1)
        
        # Input projection
        h = self.input_proj(x)
        
        # Down path
        skip_connections = []
        for block in self.down_blocks:
            h, skip = block(h, t, context)
            skip_connections.append(skip)
        
        # Middle
        h = self.middle_block(h, t, context)
        
        # Up path
        for i, block in enumerate(self.up_blocks):
            skip = skip_connections[-(i+1)]
            h = block(h, skip, t, context)
        
        # Output
        return self.output_proj(h)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all submodules that support it."""
        for module in self.modules():
            if hasattr(module, 'gradient_checkpointing_enable') and callable(module.gradient_checkpointing_enable):
                module.gradient_checkpointing_enable()
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True

class UShapeMambaDiffusion(nn.Module):
    def __init__(self, 
                 vae_model_name="stabilityai/sd-vae-ft-mse",
                 clip_model_name="openai/clip-vit-base-patch32",
                 model_channels=320,
                 use_openai_clip=False,
                 num_train_timesteps=1000):
        super().__init__()
        
        # Load pre-trained VAE
        self.vae = AutoencoderKL.from_pretrained(vae_model_name)
        
        # Load pre-trained CLIP encoder
        self.use_openai_clip = use_openai_clip
        # if use_openai_clip:
        #     # Use OpenAI CLIP
        #     self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        #     self.clip_tokenizer = clip.tokenize
        #     context_dim = self.clip_model.text_projection.out_features
        # else:
        #     # Use Hugging Face CLIP
        self.clip_text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        context_dim = self.clip_text_encoder.config.hidden_size
        
        # Get VAE latent channels
        vae_latent_channels = self.vae.config.latent_channels
        
        # U-Shape Mamba denoiser
        self.unet = UShapeMamba(
            in_channels=vae_latent_channels,
            model_channels=model_channels,
            context_dim=context_dim
        )
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(num_train_timesteps)
        
        # Freeze pre-trained components
        for param in self.vae.parameters():
            param.requires_grad = False
        
        if use_openai_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            for param in self.clip_text_encoder.parameters():
                param.requires_grad = False
    
    def encode_images(self, images):
        """Encode images to latent space using pre-trained VAE"""
        with torch.no_grad():
            # VAE encoder returns a distribution, we sample from it
            posterior = self.vae.encode(images).latent_dist
            latents = posterior.sample()
            # Scale latents according to VAE config
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def decode_latents(self, latents):
        """Decode latents to images using pre-trained VAE"""
        with torch.no_grad():
            # Unscale latents
            latents = latents / self.vae.config.scaling_factor
            # Decode
            images = self.vae.decode(latents).sample
        return images
    
    def encode_text(self, text_prompts):
        """Encode text prompts using pre-trained CLIP"""
        if self.use_openai_clip:
            # OpenAI CLIP
            with torch.no_grad():
                text_tokens = self.clip_tokenizer(text_prompts).to(next(self.parameters()).device)
                text_features = self.clip_model.encode_text(text_tokens)
                # Get the text features after projection
                text_features = text_features.float()
        else:
            # Hugging Face CLIP
            with torch.no_grad():
                text_inputs = self.clip_tokenizer(
                    text_prompts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                # Move each tensor in the dict to the correct device
                print("encode text using huggingface clip")
                text_inputs = {k: v.to(next(self.parameters()).device) for k, v in text_inputs.items()}
                
                text_features = self.clip_text_encoder(**text_inputs).last_hidden_state
        
        return text_features
    
    def forward(self, images, timesteps, text_prompts=None):
        """
        Forward pass for training - predicts noise
        """
        # Encode images to latent space
        latents = self.encode_images(images)
        
        # Add noise according to timesteps
        noisy_latents, noise = self.noise_scheduler.add_noise(latents, timesteps)
        
        # Encode text prompts if provided
        context = None
        if text_prompts is not None:
            context = self.encode_text(text_prompts)
        
        # Predict noise (not denoised latents)
        predicted_noise = self.unet(noisy_latents, timesteps, context)
        
        return predicted_noise, noise, latents
    
    def sample(self, text_prompts, num_inference_steps=50, guidance_scale=7.5, height=512, width=512):
        """
        Sample images from text prompts using proper DDPM sampling
        """
        device = next(self.parameters()).device
        batch_size = len(text_prompts)
        
        # Encode text
        text_embeddings = self.encode_text(text_prompts)
        
        # Create latent shape
        latent_height = height // 8  # VAE downsamples by 8
        latent_width = width // 8
        latent_shape = (batch_size, self.vae.config.latent_channels, latent_height, latent_width)
        
        # Initialize random noise
        latents = torch.randn(latent_shape, device=device)
        
        # Proper DDPM sampling
        timesteps = torch.linspace(self.noise_scheduler.num_train_timesteps - 1, 0, num_inference_steps).long()
        
        for t in timesteps:
            timesteps_batch = torch.full((batch_size,), t, device=device)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(latents, timesteps_batch, text_embeddings)
            
            # Denoising step
            latents = self.noise_scheduler.step(noise_pred, timesteps_batch, latents)
        
        # Decode to images
        images = self.decode_latents(latents)
        
        return images

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all submodules that support it."""
        if hasattr(self.unet, 'gradient_checkpointing_enable') and callable(self.unet.gradient_checkpointing_enable):
            self.unet.gradient_checkpointing_enable()
        if hasattr(self.unet, 'gradient_checkpointing'):
            self.unet.gradient_checkpointing = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import json
from PIL import Image
import os
from typing import List
import logging
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
                
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
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
        'image_size': 128,
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
    
    # Train
    train_model(model, train_loader, val_loader, device, config)

if __name__ == "__main__":
    main() 


