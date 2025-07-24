import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import inspect
from diffuser import UShapeMamba
from utils import print_forward_shapes
# Import pre-trained models
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer


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


class UShapeMambaDiffusion(nn.Module):
    def __init__(self, 
                 vae_model_name="stabilityai/sd-vae-ft-mse",
                 clip_model_name="openai/clip-vit-base-patch32",
                 model_channels=160,
                 num_train_timesteps=1000,
                 dropout=0.0,
                 use_shared_time_embedding=False):  # Removed use_openai_clip parameter
        super().__init__()
        
        # Load pre-trained VAE
        self.vae = AutoencoderKL.from_pretrained(vae_model_name)
        
        # Load Hugging Face CLIP encoder (only option now)
        print(f"Loading Hugging Face CLIP: {clip_model_name}")
        self.clip_text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        context_dim = self.clip_text_encoder.config.hidden_size
        
        # Get VAE latent channels
        vae_latent_channels = self.vae.config.latent_channels
        
        # U-Shape Mamba denoiser with configurable time embedding
        self.unet = UShapeMamba(
            in_channels=vae_latent_channels,
            model_channels=model_channels,
            context_dim=context_dim,
            dropout=dropout,
            use_shared_time_embedding=use_shared_time_embedding
        )
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(num_train_timesteps)
        
        # Freeze pre-trained components
        for param in self.vae.parameters():
            param.requires_grad = False
        
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
    
    def encode_text(self, text_prompts, max_length=None, padding=True):
        """Encode text prompts using Hugging Face CLIP"""
        with torch.no_grad():
            tokenizer_kwargs = {
                "padding": padding,
                "truncation": True,
                "return_tensors": "pt"
            }
            if max_length is not None:
                tokenizer_kwargs["max_length"] = max_length
            text_inputs = self.clip_tokenizer(
                text_prompts, 
                **tokenizer_kwargs
            )
        # Move each tensor in the dict to the correct device
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
        Sample images from text prompts using proper DDPM sampling with Classifier-Free Guidance (CFG)
        """
        device = next(self.parameters()).device
        batch_size = len(text_prompts)
        
        # Encode text (conditional)
        text_embeddings = self.encode_text(text_prompts)
        #print(f"Text embeddings shape: {text_embeddings.shape}")

        # Encode unconditional context (empty strings)
        uncond_embeddings = self.encode_text(
            [""] * batch_size, 
            max_length=text_embeddings.shape[1], 
            padding='max_length'
        )
        #print(f"Unconditional embeddings shape: {uncond_embeddings.shape}")
        # Create latent shape
        latent_height = height // 8  # VAE downsamples by 8
        latent_width = width // 8
        latent_shape = (batch_size, self.vae.config.latent_channels, latent_height, latent_width)
        
        # Initialize random noise
        latents = torch.randn(latent_shape, device=device)
        
        # Proper DDPM sampling, maybe change this to DDIM 
        timesteps = torch.linspace(self.noise_scheduler.num_train_timesteps - 1, 0, num_inference_steps).long()

        for t in timesteps:
            timesteps_batch = torch.full((batch_size,), t, device=device)
            # Efficient batch: concat unconditional and conditional
            context = torch.cat([uncond_embeddings, text_embeddings], dim=0)  # [2*B, seq, dim]
            latents_input = torch.cat([latents, latents], dim=0)  # [2*B, ...]
            timesteps_input = torch.cat([timesteps_batch, timesteps_batch], dim=0)

            with torch.no_grad():
                noise_pred = self.unet(latents_input, timesteps_input, context)  # [2*B, ...]
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                # CFG interpolation
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
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
    # Create model with SHARED time embedding (default)
    print("=== Testing SHARED Time Embedding ===")
    model_shared = UShapeMambaDiffusion(
        vae_model_name="stabilityai/sd-vae-ft-mse",
        clip_model_name="openai/clip-vit-base-patch32",  # Still uses Hugging Face CLIP
        use_shared_time_embedding=True  # Shared approach
    )
    
    # Create model with SEPARATE time embedding
    print("\n=== Testing SEPARATE Time Embedding ===")
    model_separate = UShapeMambaDiffusion(
        vae_model_name="stabilityai/sd-vae-ft-mse",
        clip_model_name="openai/clip-vit-base-patch32",  # Still uses Hugging Face CLIP
        use_shared_time_embedding=False  # Separate approach
    )
    
    # Example inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 512, 512)  # Input images
    timesteps = torch.randint(0, 1000, (batch_size,))
    text_prompts = ["A beautiful sunset", "A cat in a garden"]
    
    # Test both models
    for name, model in [("Shared", model_shared), ("Separate", model_separate)]:
        print(f"\n--- Testing {name} Time Embedding Model ---")
    # Forward pass
    with torch.no_grad():
        predicted_noise, noise, latents = model(images, timesteps, text_prompts)
        # Sample new images
        generated_images = model.sample(["A futuristic city", "A peaceful lake"], num_inference_steps=20)
    
    print(f"{name} Model:")
    print(f"  Input images shape: {images.shape}")
    print(f"  Original latents shape: {latents.shape}")
    print(f"  Predicted noise shape: {predicted_noise.shape}")
    print(f"  Generated images shape: {generated_images.shape}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")