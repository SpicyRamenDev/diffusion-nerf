import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


class StableDiffusion(nn.Module):
    def __init__(self, device, repo_id="stabilityai/stable-diffusion-2-1-base"):
        super().__init__()

        self.device = device

        self.pipeline = StableDiffusionPipeline.from_pretrained(repo_id)  # , torch_dtype=torch.float16, revision="fp16")
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.enable_attention_slicing()
        self.pipeline.to(device)

        for component in self.pipeline.components:
            if isinstance(component, nn.Module):
                component.requires_grad_(False)
                component.eval()

        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.scheduler = self.pipeline.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.alphas_cumprod = self.scheduler.alphas_cumprod

    def get_text_embedding(self, prompt, negative_prompt=""):
        with torch.no_grad():
            text_embedding = self.pipeline._encode_prompt(prompt=prompt,
                                                          device=self.device,
                                                          num_images_per_prompt=1,
                                                          do_classifier_free_guidance=True,
                                                          negative_prompt=negative_prompt)
        return text_embedding

    def encode(self, image):
        posterior = self.vae.encode(image).latent_dist
        latent = posterior.mean * 0.18215
        return latent

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, text_embeddings, image, guidance_scale=100,
             min_ratio=0.02, max_ratio=0.98,
             weight_type='constant',
             use_decoder=False,
             scaler=None):
        image = image.detach()
        image.requires_grad = True

        transformed_image = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False)
        transformed_image = transformed_image * 2 - 1

        latent = self.encode(transformed_image)

        min_step = int(self.num_train_timesteps * min_ratio)
        max_step = int(self.num_train_timesteps * max_ratio)
        t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        noise = torch.randn_like(latent)

        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            noisy_latent = self.add_noise(latent, noise, t)
            noisy_latents = torch.cat([noisy_latent] * 2)
            noise_preds = self.unet(noisy_latents, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_preds.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if weight_type == 'linear':
            w = (1 - self.alphas_cumprod[t])
        elif weight_type == 'quadratic':
            w = (1 - self.alphas_cumprod[t]) ** 2
        else:
            w = 1

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if scaler is not None:
            scaler.scale(latent).backward(gradient=grad)
        else:
            latent.backward(gradient=grad)
        return image.grad / (64 * 64)
