import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import numpy as np
import inspect

# Import pre-trained models
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
#import clip

DEBUG = False

#TODO: Misah-misahin model definition jadi beberapa bagian: Block (mamba, cross-attn dan scaleshift), ushape(timestep embedding, downblock, upblock, ushape), pretrained
def print_forward_shapes(forward_fn):
    def wrapper(self, *args, **kwargs):
        if not DEBUG:
            return forward_fn(self, *args, **kwargs)
        print(f"\n[{self.__class__.__name__}.forward] called")
        # Get parameter names (skip 'self')
        sig = inspect.signature(forward_fn)
        param_names = list(sig.parameters.keys())[1:]
        # Print positional args with names
        for i, arg in enumerate(args):
            name = param_names[i] if i < len(param_names) else f"arg{i}"
            if isinstance(arg, torch.Tensor):
                print(f"  {name}: shape={arg.shape}")
            else:
                print(f"  {name}: type={type(arg)}")
        # Print keyword args
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}")
            else:
                print(f"  {k}: type={type(v)}")
        # Call the original forward
        output = forward_fn(self, *args, **kwargs)
        # Print output shape(s)
        if isinstance(output, torch.Tensor):
            print(f"  Output: shape={output.shape}")
        elif isinstance(output, (tuple, list)):
            for idx, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"  Output[{idx}]: shape={out.shape}")
                else:
                    print(f"  Output[{idx}]: type={type(out)}")
        else:
            print(f"  Output: type={type(output)}")
        return output
    return wrapper

class ParallelScan:
    """
    Implements parallel scan (prefix sum) operation for Mamba's selective scan.
    This version avoids in-place modifications to support autograd.
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

        if seq_len <= 4:
            return ParallelScan._sequential_scan_log(log_coeffs, log_values)

        levels = int(np.ceil(np.log2(seq_len)))

        coeffs = log_coeffs.clone()
        values = log_values.clone()

        for level in range(levels):
            step = 2 ** (level + 1)
            if step - 1 >= seq_len:
                break

            indices = torch.arange(step - 1, seq_len, step, device=log_coeffs.device)

            if indices.numel() > 0:
                left_idx = indices - 2 ** level
                right_idx = indices

                valid_mask = (left_idx >= 0) & (right_idx < seq_len)
                left_idx = left_idx[valid_mask]
                right_idx = right_idx[valid_mask]

                if left_idx.numel() > 0:
                    left_coeffs = coeffs.index_select(1, left_idx)
                    right_coeffs = coeffs.index_select(1, right_idx)
                    left_values = values.index_select(1, left_idx)
                    right_values = values.index_select(1, right_idx)

                    new_coeffs = left_coeffs + right_coeffs
                    new_values = torch.logaddexp(left_values, right_coeffs + right_values)

                    scatter_shape = (batch_size, right_idx.size(0), dim)
                    scatter_indices = right_idx[None, :, None].expand(batch_size, -1, dim)

                    coeffs = coeffs.scatter(1, scatter_indices, new_coeffs)
                    values = values.scatter(1, scatter_indices, new_values)

        for level in reversed(range(levels)):
            step = 2 ** (level + 1)
            if step - 1 >= seq_len:
                continue

            indices = torch.arange(step - 1, seq_len, step, device=log_coeffs.device)

            if indices.numel() > 0:
                left_idx = indices - 2 ** level
                right_idx = indices

                valid_mask = (left_idx >= 0) & (right_idx < seq_len)
                left_idx = left_idx[valid_mask]
                right_idx = right_idx[valid_mask]

                if left_idx.numel() > 0:
                    left_values = values.index_select(1, left_idx)
                    right_values = values.index_select(1, right_idx)
                    right_coeffs = coeffs.index_select(1, right_idx)

                    new_right = torch.logaddexp(left_values, right_coeffs + right_values)

                    scatter_indices_right = right_idx[None, :, None].expand(batch_size, -1, dim)
                    scatter_indices_left = left_idx[None, :, None].expand(batch_size, -1, dim)

                    values = values.scatter(1, scatter_indices_right, new_right)
                    values = values.scatter(1, scatter_indices_left, right_values)

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
    Implements the selective scan mechanism from Mamba (serial version for debugging)
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        self.A_log = nn.Parameter(torch.log(torch.rand(self.d_inner, d_state)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    @print_forward_shapes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_inner.transpose(-1, -2))[..., :seq_len].transpose(-1, -2)
        x_conv = self.act(x_conv)

        x_dbl = self.x_proj(x_conv)
        delta, B, C = torch.split(x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)

        delta = F.softplus(self.dt_proj(delta))

        y = self.selective_scan(x_conv, delta, B, C)

        y = y * self.act(z)
        output = self.out_proj(y)
        return output

    def selective_scan(self, x: torch.Tensor, delta: torch.Tensor, 
                       B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Perform serial selective scan (easier to debug)
        """
        batch_size, seq_len, d_inner = x.shape
        A = -torch.exp(self.A_log.float())

        y = torch.zeros((batch_size, seq_len, d_inner), device=x.device, dtype=x.dtype)
        state = torch.zeros((batch_size, d_inner, self.d_state), device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            delta_t = delta[:, t]  # [batch, d_inner]
            B_t = B[:, t]          # [batch, d_state]
            C_t = C[:, t]          # [batch, d_state]
            x_t = x[:, t]          # [batch, d_inner]

            A_t = A.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, d_inner, d_state]
            decay = torch.exp(delta_t.unsqueeze(-1) * A_t)   # [batch, d_inner, d_state]
            Bu = B_t.unsqueeze(1) * x_t.unsqueeze(-1)        # [batch, d_inner, d_state]

            state = decay * state + Bu                       # [batch, d_inner, d_state]
            y_t = torch.sum(state * C_t.unsqueeze(1), dim=-1) + x_t * self.D  # [batch, d_inner]
            y[:, t] = y_t

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
    
    @print_forward_shapes
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
    Cross-attention module for conditioning on text embeddings.
    Handles x of shape [B, N, C] or [B, C, H, W].
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
    @print_forward_shapes
    def forward(self, x, context=None):
        # Handle both [B, C, H, W] and [B, N, C]
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
        elif x.dim() == 3:
            B, N, C = x.shape
        else:
            raise ValueError(f"Unsupported x shape in CrossAttention: {x.shape}")

        h = self.heads
        if context is None:
            context = x  # fallback to self-attention

        # Project to QKV
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head: [B, H, N, D]
        q = q.view(B, -1, h, self.dim_head).transpose(1, 2)
        k = k.view(B, -1, h, self.dim_head).transpose(1, 2)
        v = v.view(B, -1, h, self.dim_head).transpose(1, 2)

        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)

        # Merge heads: [B, N, H*D]
        out = out.transpose(1, 2).contiguous().view(B, -1, h * self.dim_head)

        return self.to_out(out)  # [B, N, C]

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

    @print_forward_shapes
    def forward(self, x, context, timestep_emb=None):
        if timestep_emb is not None:
            # Concatenate context and timestep embedding
            cond = torch.cat([context, timestep_emb], dim=-1)
            scale_shift = self.to_scale_shift_time(cond)
        else:
            scale_shift = self.to_scale_shift(context)
        scale, shift = scale_shift.chunk(2, dim=-1)

        # Reshape for broadcasting over sequence or spatial dimensions
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

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

    @print_forward_shapes
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

    @print_forward_shapes  
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
    
    @print_forward_shapes
    def forward(self, x, t, context=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        time_emb = self.time_mlp(t)
        #print(f"Downblock, h shape {h.shape}, t shape {time_emb.shape}")
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
    def __init__(self, in_channels, out_channels, skip_channels, time_embed_dim, context_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )

        self.norm = nn.GroupNorm(8, out_channels)
        self.attn = CrossAttention(dim=out_channels, context_dim=context_dim)
        self.mamba = MambaBlock(dim=out_channels)

    @print_forward_shapes
    def forward(self, x, skip, t, context=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        time_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)
        x = x + time_emb
        x = self.norm(x)
        x = self.attn(x, context)
        x = self.mamba(x)
        # Ensure output is 4D
        if x.dim() == 3:
            B, seq_len, C = x.shape
            H, W = skip.shape[2], skip.shape[3]
            assert seq_len == H * W, f"Cannot reshape: seq_len={seq_len}, H={H}, W={W}"
            x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


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

    @print_forward_shapes  
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
                 in_channels=4,
                 model_channels=160,
                 time_embed_dim=160,
                 context_dim=768):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimestepEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.context_proj = nn.Linear(context_dim, context_dim)
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        down_channels = []
        in_ch = model_channels

        for i in range(4):
            out_ch = model_channels * (2 ** i)
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, time_embed_dim, context_dim)
            )
            down_channels.append(out_ch)
            in_ch = out_ch

        self.middle_block = MiddleBlock(in_ch, time_embed_dim, context_dim)

        # Reverse for up path
        self.up_blocks = nn.ModuleList()
        skip_channels = list(reversed(down_channels))
        up_in_channels = in_ch

        for i, skip_ch in enumerate(skip_channels):
            out_ch = model_channels * (2 ** (2 - i)) if i < 3 else model_channels
            self.up_blocks.append(
                UpBlock(up_in_channels, out_ch, skip_ch, time_embed_dim, context_dim)
            )
            up_in_channels = out_ch

        self.output_proj = nn.Conv2d(model_channels, in_channels, 3, padding=1)

    @print_forward_shapes
    def forward(self, x, timesteps, context=None):
        t = self.time_embed(timesteps)

        if context is not None:
            context = self.context_proj(context)
            if context.dim() == 2:
                context = context.unsqueeze(1)

        h = self.input_proj(x)

        skip_connections = []
        for block in self.down_blocks:
            h, skip = block(h, t, context)
            skip_connections.append(skip)

        h = self.middle_block(h, t, context)
        #print("h shape after middle block", h.shape)
        for i, block in enumerate(self.up_blocks):
            skip = skip_connections[-(i + 1)]
            h = block(h, skip, t, context)

        return self.output_proj(h)



class UShapeMambaDiffusion(nn.Module):
    def __init__(self, 
                 vae_model_name="stabilityai/sd-vae-ft-mse",
                 clip_model_name="openai/clip-vit-base-patch32",
                 model_channels=160,
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
                #print("encode text using huggingface clip")
                text_inputs = {k: v.to(next(self.parameters()).device) for k, v in text_inputs.items()}
                
                text_features = self.clip_text_encoder(**text_inputs).last_hidden_state
        
        return text_features
    
    @print_forward_shapes
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

# Example usage
if __name__ == "__main__":
    # Create model with pre-trained components
    model = UShapeMambaDiffusion(
        vae_model_name="stabilityai/sd-vae-ft-mse",
        clip_model_name="openai/clip-vit-base-patch32",
        use_openai_clip=False  # Set to True to use OpenAI CLIP
    )
    
    # Example inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 512, 512)  # Input images
    timesteps = torch.randint(0, 1000, (batch_size,))
    text_prompts = ["A beautiful sunset", "A cat in a garden"]
    
    # Forward pass
    with torch.no_grad():
        denoised_latents, original_latents = model(images, timesteps, text_prompts)
        
        # Sample new images
        generated_images = model.sample(["A futuristic city", "A peaceful lake"], num_inference_steps=20)
    
    print(f"Input images shape: {images.shape}")
    print(f"Original latents shape: {original_latents.shape}")
    print(f"Denoised latents shape: {denoised_latents.shape}")
    print(f"Generated images shape: {generated_images.shape}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")