import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import numpy as np

# Import pre-trained models
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import clip

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
    Complete Mamba block implementation with proper selective SSM
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.ssm = SelectiveSSM(dim, d_state, d_conv, expand)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        return x + residual

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
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Mamba processing
        self.to_mamba = nn.Linear(out_channels, out_channels)
        self.mamba = MambaBlock(out_channels)
        self.from_mamba = nn.Linear(out_channels, out_channels)
        
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
        
    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # Apply Mamba
        B, C, H, W = h.shape
        h_flat = h.flatten(2).transpose(1, 2)
        h_flat = self.to_mamba(h_flat)
        h_flat = self.mamba(h_flat)
        h_flat = self.from_mamba(h_flat)
        h = h_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return self.downsample(h), h

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
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
        
        # Mamba processing
        self.to_mamba = nn.Linear(out_channels, out_channels)
        self.mamba = MambaBlock(out_channels)
        self.from_mamba = nn.Linear(out_channels, out_channels)
        
    def forward(self, x, skip, t):
        h = self.upsample(x)
        h = torch.cat([h, skip], dim=1)
        
        h = self.conv1(h)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # Apply Mamba
        B, C, H, W = h.shape
        h_flat = h.flatten(2).transpose(1, 2)
        h_flat = self.to_mamba(h_flat)
        h_flat = self.mamba(h_flat)
        h_flat = self.from_mamba(h_flat)
        h = h_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return h

class MiddleBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels)
        )
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
        # Mamba processing
        self.to_mamba = nn.Linear(channels, channels)
        self.mamba = MambaBlock(channels)
        self.from_mamba = nn.Linear(channels, channels)
        
    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # Apply Mamba
        B, C, H, W = h.shape
        h_flat = h.flatten(2).transpose(1, 2)
        h_flat = self.to_mamba(h_flat)
        h_flat = self.mamba(h_flat)
        h_flat = self.from_mamba(h_flat)
        h = h_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return h

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
        self.context_proj = nn.Linear(context_dim, model_channels)
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Down blocks
        self.down_blocks = nn.ModuleList([
            DownBlock(model_channels, model_channels, time_embed_dim),
            DownBlock(model_channels, model_channels * 2, time_embed_dim),
            DownBlock(model_channels * 2, model_channels * 4, time_embed_dim),
            DownBlock(model_channels * 4, model_channels * 8, time_embed_dim),
        ])
        
        # Middle block
        self.middle_block = MiddleBlock(model_channels * 8, time_embed_dim)
        
        # Up blocks
        self.up_blocks = nn.ModuleList([
            UpBlock(model_channels * 8, model_channels * 4, time_embed_dim),
            UpBlock(model_channels * 4, model_channels * 2, time_embed_dim),
            UpBlock(model_channels * 2, model_channels, time_embed_dim),
            UpBlock(model_channels, model_channels, time_embed_dim),
        ])
        
        # Output projection
        self.output_proj = nn.Conv2d(model_channels, in_channels, 3, padding=1)
        
    def forward(self, x, timesteps, context=None):
        # Time embedding
        t = self.time_embed(timesteps)
        
        # Add context if provided
        if context is not None:
            context_emb = self.context_proj(context)
            t = t + context_emb
        
        # Input projection
        h = self.input_proj(x)
        
        # Down path
        skip_connections = []
        for block in self.down_blocks:
            h, skip = block(h, t)
            skip_connections.append(skip)
        
        # Middle
        h = self.middle_block(h, t)
        
        # Up path
        for i, block in enumerate(self.up_blocks):
            skip = skip_connections[-(i+1)]
            h = block(h, skip, t)
        
        # Output
        return self.output_proj(h)

class UShapeMambaDiffusion(nn.Module):
    def __init__(self, 
                 vae_model_name="stabilityai/sd-vae-ft-mse",
                 clip_model_name="openai/clip-vit-base-patch32",
                 model_channels=320,
                 use_openai_clip=False):
        super().__init__()
        
        # Load pre-trained VAE
        self.vae = AutoencoderKL.from_pretrained(vae_model_name)
        
        # Load pre-trained CLIP encoder
        self.use_openai_clip = use_openai_clip
        if use_openai_clip:
            # Use OpenAI CLIP
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
            self.clip_tokenizer = clip.tokenize
            context_dim = self.clip_model.text_projection.out_features
        else:
            # Use Hugging Face CLIP
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
                ).to(next(self.parameters()).device)
                
                text_features = self.clip_text_encoder(**text_inputs).pooler_output
        
        return text_features
    
    def forward(self, images, timesteps, text_prompts=None):
        """
        Forward pass for training/inference
        """
        # Encode images to latent space
        latents = self.encode_images(images)
        
        # Encode text prompts if provided
        context = None
        if text_prompts is not None:
            context = self.encode_text(text_prompts)
        
        # Denoise in latent space
        denoised_latents = self.unet(latents, timesteps, context)
        
        return denoised_latents, latents
    
    def sample(self, text_prompts, num_inference_steps=50, guidance_scale=7.5, height=512, width=512):
        """
        Sample images from text prompts using DDPM sampling
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
        
        # Simple DDPM sampling (you'd typically use a scheduler here)
        for t in torch.linspace(num_inference_steps-1, 0, num_inference_steps).long():
            timesteps = torch.full((batch_size,), t, device=device)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(latents, timesteps, text_embeddings)
            
            # Simple denoising step (replace with proper scheduler)
            alpha = 1.0 - t / num_inference_steps
            latents = alpha * latents + (1 - alpha) * noise_pred
        
        # Decode to images
        images = self.decode_latents(latents)
        
        return images

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