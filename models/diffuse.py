import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UShapeMamba
from tools.debug import print_forward_shapes
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
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained models
        print(f"Loading Hugging Face CLIP: {config.Model.Diffuser.clip_model_name}")
        print(f"Loading pre-trained VAE: {config.Model.Diffuser.vae_model_name}")
        
        try:
            self.vae = AutoencoderKL.from_pretrained(
                config.Model.Diffuser.vae_model_name, 
                torch_dtype=torch.float16
            ).to(self.device)
            
            self.clip_text_encoder = CLIPTextModel.from_pretrained(
                config.Model.Diffuser.clip_model_name
            ).to(self.device)  # Added .to(device)
            
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(
                config.Model.Diffuser.clip_model_name
            )
        except Exception as e:
            print(f"Error loading pretrained models: {e}")
            raise
        
        context_dim = self.clip_text_encoder.config.hidden_size
        vae_latent_channels = self.vae.config.latent_channels
        
        # U-Shape Mamba denoiser
        self.unet = UShapeMamba(
            config,
            in_channels=vae_latent_channels,
            model_channels=config.Model.Unet.model_channels,
            context_dim=context_dim,
            time_embed_dim=config.Model.Unet.time_dim
        ).to(self.device)  # Added .to(device)
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(
            config.Model.Diffuser.train_timesteps
        ).to(self.device)
        
        # Freeze pre-trained components
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.clip_text_encoder.parameters():
            param.requires_grad = False
    
    def encode_images(self, images):
        """Encode images to latent space using pre-trained VAE"""
        with torch.no_grad():
            # Ensure images are on correct device and dtype
            images = images.to(device=self.device, dtype=next(self.vae.parameters()).dtype)
            posterior = self.vae.encode(images).latent_dist
            latents = posterior.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents):
        """Decode latents to images using pre-trained VAE"""
        with torch.no_grad():
            latents = latents.to(device=self.device, dtype=next(self.vae.parameters()).dtype)
            latents = latents / self.vae.config.scaling_factor
            images = self.vae.decode(latents).sample
        return images
        
    def encode_text(self, text_prompts, max_length=None, padding=True):
        """Encode text prompts using Hugging Face CLIP"""
        with torch.no_grad():
            tokenizer_kwargs = {
                "truncation": True,
                "return_tensors": "pt"
            }

            if max_length is not None:
                tokenizer_kwargs["max_length"] = max_length
                tokenizer_kwargs["padding"] = "max_length"  # ✅ force fixed length padding
            else:
                tokenizer_kwargs["padding"] = padding  # fallback to whatever was passed (e.g., True)

            text_inputs = self.clip_tokenizer(text_prompts, **tokenizer_kwargs)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_features = self.clip_text_encoder(**text_inputs).last_hidden_state

        return text_features

    
    def forward(self, images, timesteps, text_prompts=None):
        """Forward pass for training - predicts noise"""
        # Ensure inputs are on correct device
        images = images.to(self.device)
        timesteps = timesteps.to(self.device)
        
        # Encode images to latent space
        latents = self.encode_images(images)
        
        # Add noise according to timesteps
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode text prompts if provided
        context = None
        if text_prompts is not None:
            context = self.encode_text(text_prompts)
        
        # Predict noise
        predicted_noise = self.unet(noisy_latents, timesteps, context)
        
        return predicted_noise, noise, latents

    def sample(self, prompt, negative_prompt="", height=256, width=256, 
           num_inference_steps=50, guidance_scale=7.5, eta=0.0, generator=None):
        """
        Sample images using CFG + DDIM sampling
        
        Args:
            prompt (str or list): Text prompt(s) for generation
            negative_prompt (str or list): Negative prompt(s) for CFG
            height (int): Image height in pixels
            width (int): Image width in pixels  
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): CFG guidance scale (1.0 = no guidance)
            eta (float): DDIM eta parameter (0.0 = deterministic)
            generator: Random generator for reproducibility
        
        Returns:
            torch.Tensor: Generated images in pixel space
        """
        # Handle batch prompts
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)
        
        batch_size = len(prompt)
        #print("negative prompt len: ", len(negative_prompt))
        # Calculate latent dimensions
        latent_height = height // 8  # VAE downsamples by 8x
        latent_width = width // 8
        latent_channels = self.vae.config.latent_channels
        
        # Create timestep schedule
        timesteps = self._create_inference_schedule(num_inference_steps)
        
        # Initialize random latents
        latents = torch.randn(
            (batch_size, latent_channels, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=next(self.unet.parameters()).dtype
        )
        
        # Encode prompts for CFG
        if guidance_scale > 1.0:
            # Tokenize both to get max length needed
            prompt_tokens = self.clip_tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
            print(f"len negative_prompt before tokenizer: {len(negative_prompt)}")
            neg_tokens = self.clip_tokenizer(negative_prompt, padding=True, truncation=True, return_tensors="pt")
            max_length = max(prompt_tokens.input_ids.shape[1], neg_tokens.input_ids.shape[1])
            print(f"len negative_prompt after tokenizer: {len(negative_prompt)}")
            print("max_length: ", max_length)
            # Encode both with same max_length
            text_embeddings = self.encode_text(prompt, max_length=max_length)
            uncond_embeddings = self.encode_text(negative_prompt, max_length=max_length)
            print(f"text_embedding shape: {text_embeddings.shape}, uncond_shape: {uncond_embeddings.shape}")
            # Concatenate for batch processing
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
            # Duplicate latents for CFG
            latents = torch.cat([latents] * 2)
        else:
            # No CFG - just conditional
            text_embeddings = self.encode_text(prompt)
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand timestep for batch
            t_batch = torch.full((latents.shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(latents, t_batch, text_embeddings)
            
            # Apply CFG
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # Use only conditional latents for next step
                latents = latents.chunk(2)[1]
            
            # DDIM step
            latents = self.noise_scheduler.step_ddim(
                noise_pred, t, latents, eta=eta, generator=generator
            )
            
            # Re-duplicate for next CFG step (except last iteration)
            if guidance_scale > 1.0 and i < len(timesteps) - 1:
                latents = torch.cat([latents] * 2)
        
        # Decode latents to images
        images = self.decode_latents(latents)
        
        # Convert to [0, 1] range
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
        
        return images

    def _create_inference_schedule(self, num_inference_steps):
        """Create timestep schedule for inference"""
        step_ratio = self.noise_scheduler.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round().long()
        timesteps = torch.flip(timesteps, dims=[0])  # Reverse for denoising
        return timesteps.to(self.device)