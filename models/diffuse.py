import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UShapeMamba
from models.debug import print_forward_shapes
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

import torch
import math

class NoiseScheduler:
    """
    🔁 Unified Diffusion Noise Scheduler (DDPM + DDIM)
    - Cosine schedule w/ Zero Terminal SNR
    - Supports v-parameterization
    - Training: add_noise()
    - Inference: step_ddpm(), step_ddim()
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 prediction_type="v_prediction"):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        # Cosine schedule with zero terminal SNR
        steps = num_train_timesteps + 1
        x = torch.linspace(0, num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas_cumprod[-1] = 0.0  # zero terminal SNR

        self.alphas_cumprod = alphas_cumprod[:-1]
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-2]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.snr = self.alphas_cumprod / (1 - self.alphas_cumprod)

        self.betas = 1.0 - self.alphas_cumprod / self.alphas_cumprod_prev
        self.betas[0] = self.betas[1]
        self.alphas = 1.0 - self.betas

    def add_noise(self, x_start, noise, timesteps):
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def step_ddpm(self, model_output, t, x_t, generator=None):
        """
        ⬅ DDPM stochastic sampling step
        """
        prev_t = t - 1 if t > 0 else 0
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_t]
        beta_t = 1 - alpha_t

        if self.prediction_type == "v_prediction":
            x_0 = self.predict_start_from_v(x_t, model_output, t)
            eps = self.get_epsilon_from_v(x_t, model_output, t)
        else:
            eps = model_output
            x_0 = (x_t - torch.sqrt(beta_t) * eps) / torch.sqrt(alpha_t)

        coef_x0 = torch.sqrt(alpha_prev)
        coef_eps = torch.sqrt(1 - alpha_prev)
        x_prev = coef_x0 * x_0 + coef_eps * eps

        if t > 0:
            noise = torch.randn_like(x_t, generator=generator)
            var = torch.sqrt(self._get_variance(t, prev_t)) * noise
            x_prev = x_prev + var

        return x_prev

    def step_ddim(self, model_output, t, x_t, eta=0.0, generator=None):
        """
        ⬅ DDIM deterministic sampling step
        """
        prev_t = t - 1 if t > 0 else 0
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_prev = torch.sqrt(alpha_prev)

        if self.prediction_type == "v_prediction":
            x_0 = self.predict_start_from_v(x_t, model_output, t)
            eps = self.get_epsilon_from_v(x_t, model_output, t)
        else:
            eps = model_output
            x_0 = (x_t - torch.sqrt(1 - alpha_t) * eps) / sqrt_alpha_t

        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * eps
        noise = sigma * torch.randn_like(x_t, generator=generator) if eta > 0 else 0

        x_prev = sqrt_alpha_prev * x_0 + dir_xt + noise
        return x_prev

    def _get_variance(self, t, prev_t):
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_t = 1 - alpha_t
        beta_prev = 1 - alpha_prev
        return (beta_prev / beta_t) * (1 - alpha_t / alpha_prev)

    def get_v_target(self, x_0, noise, timesteps):
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x_0

    def predict_start_from_v(self, x_t, v, timesteps):
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v

    def get_epsilon_from_v(self, x_t, v, timesteps):
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha * v + sqrt_one_minus_alpha * x_t

    def to(self, device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.snr = self.snr.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        return self

    
#------------------------------Diffusion Model-----------------------#
class UShapeMambaDiffusion(nn.Module):
    def __init__(self, 
                 config):  # Removed use_openai_clip parameter
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load pre-trained VAE
        self.vae = AutoencoderKL.from_pretrained(config.Diffuser.vae_model_name, torch_dtype=torch.float16).to(self.device)
        
        # Load Hugging Face CLIP encoder (only option now)
        print(f"Loading Hugging Face CLIP: {config.Model.Diffuser.clip_model_name}")
        self.clip_text_encoder = CLIPTextModel.from_pretrained(config.Model.Diffuser.clip_model_name)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(config.Model.Diffuser.clip_model_name)
        context_dim = self.clip_text_encoder.config.hidden_size
        #print("CLIP context dim", context_dim)
        # Get VAE latent channels
        vae_latent_channels = self.vae.config.latent_channels
        
        # U-Shape Mamba denoiser with configurable time embedding
        self.unet = UShapeMamba(
            config,
            in_channels=vae_latent_channels,
            model_channels=config.Model.Unet.model_channels,
            context_dim=context_dim,
            time_embed_dim=config.Model.Unet.time_dim)
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(config.Model.Diffuser.num_train_timesteps).to(self.device)
        
        # Freeze pre-trained components
        for param in self.vae.parameters():
            param.requires_grad = False
        
        for param in self.clip_text_encoder.parameters():
            param.requires_grad = False
    
    def encode_images(self, images):
        """Encode images to latent space using pre-trained VAE"""
        with torch.no_grad():
            dtype = next(self.vae.parameters()).dtype  # detect VAE's dtype (e.g., float16)
            images = images.to(dtype=dtype)            # cast images to match VAE
            posterior = self.vae.encode(images).latent_dist
            latents = posterior.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents):
        """Decode latents to images using pre-trained VAE"""
        with torch.no_grad():
            dtype = next(self.vae.parameters()).dtype
            latents = latents.to(dtype=dtype)
            latents = latents / self.vae.config.scaling_factor
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
    
    #@print_forward_shapes
    def forward(self, images, timesteps, text_prompts=None):
        """
        Forward pass for training - predicts noise
        """
        # Encode images to latent space
        latents = self.encode_images(images)
        #print(f"Latents shape: {latents.shape}")
        # Add noise according to timesteps
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode text prompts if provided
        context = None
        if text_prompts is not None:
            context = self.encode_text(text_prompts)
            #print(f"Context shape: {context.shape}")
        
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
        print(f"Text embeddings shape: {text_embeddings.shape}")

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