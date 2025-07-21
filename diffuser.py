import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import print_forward_shapes
from blocks import MainBlockSerial, MainBlockParallel


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
    def __init__(self, in_channels, out_channels, time_dim, context_dim, time_in_mainblock=False, use_shared_time_embedding=True):
        super().__init__()
        self.use_shared_time_embedding = use_shared_time_embedding
        
        if use_shared_time_embedding:    # Shared approach: Project from shared time embedding to block channels
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, out_channels)  # Project shared embedding to block channels
            )
        else:  # Separate approach: Create independent time embedding for this block
            self.time_mlp = nn.Sequential(
                TimestepEmbedding(time_dim),  # Create timestep embedding
                nn.Linear(time_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, out_channels)  # Project to block channels
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
    def __init__(self, in_channels, out_channels, skip_channels, time_embed_dim, context_dim, time_in_mainblock=False, use_shared_time_embedding=True):
        super().__init__()
        self.use_shared_time_embedding = use_shared_time_embedding
        
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if use_shared_time_embedding:  # Shared approach: Project from shared time embedding to block channels
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, out_channels)  # Project shared embedding to block channels
            )
        else:  # Separate approach: Create independent time embedding for this block
            self.time_mlp = nn.Sequential(
                TimestepEmbedding(time_embed_dim),  # Create timestep embedding
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, out_channels)  # Project to block channels
            )

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Main block processing - consistent with DownBlock and MiddleBlock
        self.to_main_block = nn.Linear(out_channels, out_channels)
        self.main_block = MainBlockSerial(out_channels, context_dim)
        self.from_main_block = nn.Linear(out_channels, out_channels)
        self.time_in_mainblock = time_in_mainblock
        
    @print_forward_shapes
    def forward(self, x, skip, t, context=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        
        # Process time embedding - consistent with DownBlock and MiddleBlock
        time_emb = self.time_mlp(t)

        if not self.time_in_mainblock:
            x = x + time_emb[:, :, None, None]
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)
        
        # Apply Main Block - consistent with DownBlock and MiddleBlock
        if context is not None:
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)
            x_flat = self.to_main_block(x_flat)

            if self.time_in_mainblock:
                x_flat = self.main_block(x_flat, context, time_emb)
            else:
                x_flat = self.main_block(x_flat, context)

            x_flat = self.from_main_block(x_flat)
            x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return x


class MiddleBlock(nn.Module):
    def __init__(self, channels, time_dim, context_dim, time_in_mainblock=False, use_shared_time_embedding=True):
        super().__init__()
        self.use_shared_time_embedding = use_shared_time_embedding
        
        if use_shared_time_embedding: # Shared approach: Project from shared time embedding to block channels
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, channels)  # Project shared embedding to block channels
            )
        else: # Separate approach: Create independent time embedding for this block
            self.time_mlp = nn.Sequential(
                TimestepEmbedding(time_dim),  # Create timestep embedding
                nn.Linear(time_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, channels)  # Project to block channels
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


class UShapeMamba(nn.Module):
    def __init__(self, 
                 in_channels=4,
                 model_channels=160,
                 time_embed_dim=160,
                 context_dim=768,
                 use_shared_time_embedding=True):  # New parameter to control time embedding approach
        super().__init__()
        
        # Time embedding configuration
        self.use_shared_time_embedding = use_shared_time_embedding
        
        if use_shared_time_embedding:
            # Shared time embedding approach (default)
            self.time_embed = nn.Sequential(
                TimestepEmbedding(model_channels),
                nn.Linear(model_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            print("Using SHARED time embedding approach")
        else:
            # Separate time embedding approach - no global time embedding
            print("Using SEPARATE time embedding approach")
        
        self.context_proj = nn.Linear(context_dim, context_dim)
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        down_channels = []
        in_ch = model_channels

        for i in range(4):
            out_ch = model_channels * (2 ** i)
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, time_embed_dim, context_dim, 
                         use_shared_time_embedding=use_shared_time_embedding)
            )
            down_channels.append(out_ch)
            in_ch = out_ch

        self.middle_block = MiddleBlock(in_ch, time_embed_dim, context_dim, 
                                       use_shared_time_embedding=use_shared_time_embedding)

        # Reverse for up path
        self.up_blocks = nn.ModuleList()
        skip_channels = list(reversed(down_channels))
        up_in_channels = in_ch

        for i, skip_ch in enumerate(skip_channels):
            out_ch = model_channels * (2 ** (2 - i)) if i < 3 else model_channels
            self.up_blocks.append(
                UpBlock(up_in_channels, out_ch, skip_ch, time_embed_dim, context_dim, 
                       time_in_mainblock=False, use_shared_time_embedding=use_shared_time_embedding)
            )
            up_in_channels = out_ch

        self.output_proj = nn.Conv2d(model_channels, in_channels, 3, padding=1)
        
    @print_forward_shapes
    def forward(self, x, timesteps, context=None):
        if self.use_shared_time_embedding:
            t = self.time_embed(timesteps)  # Shared approach: Use global time embedding
        else:
            t = timesteps  # Separate approach: Pass raw timesteps to each block
        
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
        for i, block in enumerate(self.up_blocks):
            skip = skip_connections[-(i + 1)]
            h = block(h, skip, t, context)
        
        return self.output_proj(h)