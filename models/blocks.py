import torch
import torch.nn as nn
import torch.nn.functional as F
from debug import print_forward_shapes

# Import official Mamba implementation if available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("Using official Mamba implementation from mamba-ssm")
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not available, falling back to custom implementation")
    print("Install with: pip install mamba-ssm")

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model core module
    Uses official Mamba implementation when available, falls back to custom implementation
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        #self.norm = nn.RMSNorm(dim)
        
        # Try to use official Mamba if available
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_official = True
        else:
            # Fallback to custom implementation
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
            self.use_official = False
        
    @print_forward_shapes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_official:
            # Use official Mamba implementation
            return self.mamba(x)
        else:
            # Use custom implementation (fallback)
            return self._forward_custom(x)
    
    def _forward_custom(self, x: torch.Tensor) -> torch.Tensor:
        """
        Custom implementation (fallback when mamba-ssm is not available)
        """
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

class CrossAttention(nn.Module): #maybe put dropout here
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
    def __init__(self, dim, context_dim, use_zero_init=True):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim, dim * 2)
        )
        self.to_scale_shift_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim * 2, dim * 2)
        )
        if use_zero_init:
            nn.init.zeros_(self.to_scale_shift[-1].weight)
            nn.init.zeros_(self.to_scale_shift[-1].bias)

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

class MainBlockSerial(nn.Module):  #Maybe put dropout here
    """
    Main block implementing the architecture from the second image, now supports optional timestep embedding.
    Uses official Mamba if available, otherwise falls back to MambaBlock.
    """
    def __init__(self, dim, context_dim, heads=8, dim_head=64, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.cross_attn = CrossAttention(dim, context_dim, heads, dim_head)
        self.scale_shift_1 = ScaleShift(dim, context_dim)
        self.norm_1 = nn.RMSNorm(dim)
        if MAMBA_AVAILABLE:
            self.mamba_block = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba_block = MambaBlock(dim, d_state, d_conv, expand)
        self.scale_shift_2 = ScaleShift(dim, context_dim)
        self.norm_2 = nn.RMSNorm(dim)
        self.scale_1 = nn.Parameter(torch.ones(1))
        self.scale_2 = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(p = 0.05)
        self.dropout_2 = nn.Dropout(p = 0.1)  # Optional dropout after Mamba block
        
    @print_forward_shapes
    def forward(self, x, context, timestep_emb=None):
        residual_1 = x
        x_mamba = self.dropout(self.mamba_block(x))
        x_mamba = self.scale_shift_1(x_mamba, context.mean(dim=1), timestep_emb)
        x_mamba = self.norm_1(x_mamba)
        attn_inp = x_mamba * self.scale_1 + residual_1

        x_attn = self.cross_attn(attn_inp, context)
        x_attn = self.scale_shift_2(x_attn, context.mean(dim=1), timestep_emb) #RMSNorm after dropout?
        x_attn = self.norm_2(self.dropout_2(x_attn))
        output = x_attn * self.scale_2 + attn_inp
        return output



class MainBlockParallel(nn.Module):
    """
    Main block implementing the architecture from the second image, now supports optional timestep embedding.
    Uses official Mamba if available, otherwise falls back to MambaBlock.
    """
    def __init__(self, dim, context_dim, heads=8, dim_head=64, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.cross_attn = CrossAttention(dim, context_dim, heads, dim_head)
        self.scale_shift_1 = ScaleShift(dim, context_dim)
        self.norm_1 = nn.LayerNorm(dim)
        if MAMBA_AVAILABLE:
            self.mamba_block = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba_block = MambaBlock(dim, d_state, d_conv, expand)
        self.scale_shift_2 = ScaleShift(dim, context_dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.scale_1 = nn.Parameter(torch.ones(1))
        self.scale_2 = nn.Parameter(torch.ones(1))
        
    def forward(self, x, context, timestep_emb=None):
        residual = x
        x_attn = self.cross_attn(x, context)
        x_attn = self.scale_shift_1(x_attn, context.mean(dim=1), timestep_emb)
        x_attn = self.norm_1(x_attn)
        x_attn = x_attn * self.scale_1
        x_mamba = self.mamba_block(x)
        x_mamba = self.scale_shift_2(x_mamba, context.mean(dim=1), timestep_emb)
        x_mamba = self.norm_2(x_mamba)
        x_mamba = x_mamba * self.scale_2
        output = residual + x_attn + x_mamba
        return output

