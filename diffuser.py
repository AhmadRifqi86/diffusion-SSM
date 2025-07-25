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

class AdaptiveGroupNorm(nn.Module): #Replace GroupNorm with AdaptiveGroupNorm
    """
    Adaptive Group Normalization that adapts to timestep
    Better than standard GroupNorm for diffusion models
    """
    def __init__(self, num_groups, num_channels, time_emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, num_channels * 2)
        )
        
    def forward(self, x, time_emb):
        x = self.norm(x)
        scale_shift = self.time_emb_proj(time_emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        return x * (scale + 1) + shift

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, context_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # self.norm1 = nn.GroupNorm(8, out_channels)
        # self.norm2 = nn.GroupNorm(8, out_channels)
        self.norm1 = AdaptiveGroupNorm(8, out_channels, time_dim)  #
        self.norm2 = AdaptiveGroupNorm(8, out_channels, time_dim)  #
        self.to_main_block = nn.Linear(out_channels, out_channels)
        self.main_block = MainBlockSerial(out_channels, context_dim)
        self.from_main_block = nn.Linear(out_channels, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
        
        
    @print_forward_shapes
    def forward(self, x, skip, t, context=None):

        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.norm1(x,t)
        x = F.silu(x)
        
        x = self.conv2(x)
        x = self.norm2(x,t)
        x = F.silu(x)
        
        # Apply Main Block - consistent with DownBlock and MiddleBlock
        if context is not None:
            #print("UpBlock Context shape: ", context.shape)
            #print("UpBlock Time shape: ", t.shape) 
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)
            x_flat = self.to_main_block(x_flat)
            #print("x_flat shape: ", x_flat.shape)
            x_flat = self.main_block(x_flat, context, t)
            x_flat = self.from_main_block(x_flat)
            x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, time_embed_dim, context_dim):
        super().__init__()
        
        
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = AdaptiveGroupNorm(8, out_channels, time_embed_dim)  # Use AdaptiveGroupNorm
        self.norm2 = AdaptiveGroupNorm(8, out_channels, time_embed_dim)
        # Main block processing - consistent with DownBlock and MiddleBlock
        self.to_main_block = nn.Linear(out_channels, out_channels)
        self.main_block = MainBlockSerial(out_channels, context_dim)
        self.from_main_block = nn.Linear(out_channels, out_channels)
        
    @print_forward_shapes
    def forward(self, x, skip, t, context=None):

        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.norm1(x,t)
        x = F.silu(x)
        x = self.conv2(x)
        x = self.norm2(x,t)
        x = F.silu(x)
        
        # Apply Main Block - consistent with DownBlock and MiddleBlock
        if context is not None:
            #print("UpBlock Context shape: ", context.shape)
            #print("UpBlock Time shape: ", t.shape) 
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)
            x_flat = self.to_main_block(x_flat)
            #print("x_flat shape: ", x_flat.shape)
            x_flat = self.main_block(x_flat, context, t)
            x_flat = self.from_main_block(x_flat)
            x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return x


class MiddleBlock(nn.Module):
    def __init__(self, channels, time_dim, context_dim,dropout):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        # self.norm1 = AdaptiveGroupNorm(8, channels)
        # self.norm2 = nn.GroupNorm(8, channels)
        self.norm1 = AdaptiveGroupNorm(8, channels, time_dim)  # Use AdaptiveGroupNorm
        self.norm2 = AdaptiveGroupNorm(8, channels, time_dim)
        # Main block processing
        self.to_main_block = nn.Linear(channels, channels)
        self.main_block = MainBlockSerial(channels, context_dim)
        self.from_main_block = nn.Linear(channels, channels)
        
    @print_forward_shapes  
    def forward(self, x, t, context=None):
        h = self.conv1(x)
        h = self.norm1(h,t)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h,t)
        h = F.silu(h)
        
        # Apply Main Block
        if context is not None:
            #print("MiddleBlock Context shape: ", context.shape)
            #print("MiddleBlock Time shape: ", t.shape)  
            B, C, H, W = h.shape
            h_flat = h.flatten(2).transpose(1, 2)
            h_flat = self.to_main_block(h_flat)
            #print("h_flat shape: ", h_flat.shape)
            h_flat = self.main_block(h_flat, context, t) #time embedding is 
            h_flat = self.from_main_block(h_flat)
            h = h_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return h


class UShapeMamba(nn.Module):
    def __init__(self, 
                 in_channels=4,
                 model_channels=160,
                 time_embed_dim=160,
                 context_dim=768,
                ):  # New parameter to control time embedding approach
        super().__init__()
        
        self.time_embed = nn.Sequential(
            TimestepEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.LayerNorm(time_embed_dim),
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
        t = self.time_embed(timesteps)  # Shared approach: Use global time embedding
        
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