import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from unet import UShapeMamba
from debug import print_forward_shapes
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

class NoiseScheduler:
    """
    Advanced noise scheduler with Zero Terminal SNR and v-parameterization
    Now includes add_noise() and step() methods for training and inference
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, 
                 prediction_type="v_prediction"):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        
        # Use cosine schedule for better distribution
        steps = num_train_timesteps + 1
        x = torch.linspace(0, num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # 🔥 CRITICAL: Enforce Zero Terminal SNR
        alphas_cumprod[-1] = 0.0  # Perfect noise at t=T
        
        self.alphas_cumprod = alphas_cumprod[:-1]
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-2]])
        
        # For v-parameterization and general use
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # SNR for Min-SNR weighting
        self.snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        
        # For DDPM sampling
        self.betas = 1.0 - self.alphas_cumprod / self.alphas_cumprod_prev
        self.betas[0] = self.betas[1]  # Prevent NaN
        self.alphas = 1.0 - self.betas
    
    def add_noise(self, original_samples, noise, timesteps):
        """
        Add noise to samples for training (forward process)
        """
        # Ensure timesteps are on the same device
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(self, model_output, timestep, sample, eta=0.0, generator=None):
        """
        Reverse diffusion step for inference (DDPM sampling)
        Supports both noise prediction and v-prediction
        """
        t = timestep
        prev_t = t - self.num_train_timesteps // self.num_inference_timesteps if hasattr(self, 'num_inference_timesteps') else t - 1
        
        # Get schedule values
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        
        # Convert v-prediction to x0 prediction if needed
        if self.prediction_type == "v_prediction":
            pred_original_sample = self.predict_start_from_v(sample, model_output, t)
        else:  # epsilon prediction
            pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        
        # Compute coefficients for prev_sample
        alpha_prod_t_prev_sqrt = torch.sqrt(alpha_prod_t_prev)
        beta_prod_t_prev_sqrt = torch.sqrt(1 - alpha_prod_t_prev)
        
        # Compute prev_sample mean
        pred_sample_direction = beta_prod_t_prev_sqrt * model_output if self.prediction_type != "v_prediction" else beta_prod_t_prev_sqrt * self.get_epsilon_from_v(sample, model_output, t)
        prev_sample = alpha_prod_t_prev_sqrt * pred_original_sample + pred_sample_direction
        
        # Add noise (DDPM sampling), change to DDIM later
        if t > 0:
            noise = torch.randn_like(sample, generator=generator)
            variance = torch.sqrt(self._get_variance(t, prev_t)) * noise
            prev_sample = prev_sample + variance
            
        return prev_sample
    
    def _get_variance(self, t, prev_t):
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
    
    def get_v_target(self, x_0, noise, timesteps):
        """
        🔥 V-parameterization: Predict velocity instead of noise
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        v = sqrt_alpha * noise - sqrt_one_minus_alpha * x_0
        return v
    
    def predict_start_from_v(self, x_t, v, timesteps):
        """Convert v-prediction back to x_0 prediction"""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        x_0 = sqrt_alpha * x_t - sqrt_one_minus_alpha * v
        return x_0
    
    def get_epsilon_from_v(self, x_t, v, timesteps):
        """Convert v-prediction to epsilon (noise) prediction"""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        epsilon = sqrt_alpha * v + sqrt_one_minus_alpha * x_t
        return epsilon
    
#------------------------------Diffusion Model-----------------------#
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