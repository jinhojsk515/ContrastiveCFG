"""
This module includes LDM-based inverse problem solvers.
Forward operators follow DPS and DDRM/DDNM.
"""
from typing import Any, Callable, Dict, Optional

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm
import numpy as np

####### Factory #######
__SOLVER__ = {}


def register_solver(name: str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls

    return wrapper


def get_solver(name: str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)


#######################

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


########################

class StableDiffusion():
    def __init__(self,
                 solver_config: Dict,
                 model_key: str = "pt-sk/stable-diffusion-1.5",  # "runwayml/stable-diffusion-v1-5" "pt-sk/stable-diffusion-1.5"
                 device: Optional[torch.device] = None,
                 **kwargs):
        self.device = device

        self.dtype = kwargs.get("pipe_dtype", torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype).to(device)
        self.vae = pipe.vae.eval()
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.eval()
        self.unet = pipe.unet.eval()

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.total_alphas = self.scheduler.alphas_cumprod.clone()

        self.sigmas = (1 - self.total_alphas).sqrt() / self.total_alphas.sqrt()
        self.log_sigmas = self.sigmas.log()

        total_timesteps = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = total_timesteps // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.sample(*args, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Solver must implement sample() method.")

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def get_text_embed(self, null_prompt, prompt, negative_prompt=None):
        """
        Get text embedding.
        args:
            null_prompt (str): null text
            prompt (str): guidance text
        """
        # null text embedding (negation)
        null_text_input = self.tokenizer(null_prompt,
                                         padding='max_length',
                                         max_length=self.tokenizer.model_max_length,
                                         return_tensors="pt", )
        null_text_embed = self.text_encoder(null_text_input.input_ids.to(self.device))[0]

        # text embedding (guidance)
        text_input = self.tokenizer(prompt,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt",
                                    truncation=True)
        text_embed = self.text_encoder(text_input.input_ids.to(self.device))[0]

        if negative_prompt is not None:
            negative_text_input = self.tokenizer(negative_prompt,
                                                 padding='max_length',
                                                 max_length=self.tokenizer.model_max_length,
                                                 return_tensors="pt", )
            negative_text_embed = self.text_encoder(negative_text_input.input_ids.to(self.device))[0]
            return null_text_embed, text_embed, negative_text_embed

        return null_text_embed, text_embed

    def encode(self, x):
        """
        xt -> zt
        """
        return self.vae.encode(x).latent_dist.sample() * 0.18215

    def decode(self, zt):
        """
        zt -> xt
        """
        zt = 1 / 0.18215 * zt
        img = self.vae.decode(zt).sample.float()
        return img

    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor,
                      uc: torch.Tensor,
                      c: torch.Tensor,
                      nc: torch.Tensor = None):
        """
        compuate epsilon_theta for null and condition
        args:
            zt (torch.Tensor): latent features
            t (torch.Tensor): timestep
            uc (torch.Tensor): null-text embedding
            c (torch.Tensor): text embedding
        """
        t_in = t.unsqueeze(0)
        t_in, uc, c, nc = t_in.expand(zt.shape[0]), uc.expand(zt.shape[0], -1, -1), c.expand(zt.shape[0], -1, -1), nc.expand(zt.shape[0], -1, -1)
        if uc is None:
            noise_c = self.unet(zt, t_in, encoder_hidden_states=c)['sample']
            noise_uc = noise_c
        elif c is None:
            noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc)['sample']
            noise_c = noise_uc
        elif nc is not None:
            noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc)['sample']
            noise_c = self.unet(zt, t_in, encoder_hidden_states=c)['sample']
            noise_nc = self.unet(zt, t_in, encoder_hidden_states=nc)['sample']
            return noise_uc, noise_c, noise_nc

        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2)
            t_in = torch.cat([t_in] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)
        return noise_uc, noise_c

    @torch.no_grad()
    def inversion(self,
                  z0: torch.Tensor,
                  uc: torch.Tensor,
                  c: torch.Tensor,
                  cfg_guidance: float = 1.0):

        # initialize z_0
        zt = z0.clone().to(self.device)

        answers = torch.from_numpy(np.load('cfg_inversion_answer.npy')).to(self.device)
        # zt = answers[-1].clone()
        # print(((zt - answers[-1])**2).mean())

        # loop
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM Inversion')
        for i, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)
            # print('inversion', t, at, at_prev)

            noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1 - at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1 - at).sqrt() * noise_pred
            # print(((zt - answers[-(i+2)]) ** 2).mean())

        return zt

    def initialize_latent(self,
                          method: str = 'random',
                          src_img: Optional[torch.Tensor] = None,
                          **kwargs):
        if method == 'ddim':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('uc'),
                               kwargs.get('c'),
                               cfg_guidance=kwargs.get('cfg_guidance', 0.0))
        elif method == 'npi':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('c'),
                               kwargs.get('c'),
                               cfg_guidance=1.0)
        elif method == 'random':
            size = kwargs.get('latent_dim', (1, 4, 64, 64))
            z = torch.randn(size).to(self.device)
        elif method == 'random_kdiffusion':
            size = kwargs.get('latent_dim', (1, 4, 64, 64))
            sigmas = kwargs.get('sigmas', [14.6146])
            z = torch.randn(size).to(self.device)
            z = z * (sigmas[0] ** 2 + 1) ** 0.5
        else:
            raise NotImplementedError

        return z.requires_grad_()

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def to_d(self, x, sigma, denoised):
        '''converts a denoiser output to a Karras ODE derivative'''
        return (x - denoised) / sigma.item()

    def get_ancestral_step(self, sigma_from, sigma_to, eta=1.):
        """Calculates the noise level (sigma_down) to step down to and the amount
        of noise to add (sigma_up) when doing an ancestral sampling step."""
        if not eta:
            return sigma_to, 0.
        sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        return sigma_down, sigma_up

    def calculate_input(self, x, sigma):
        return x / (sigma ** 2 + 1) ** 0.5

    def calculate_denoised(self, x, model_pred, sigma):
        return x - model_pred * sigma

    def kdiffusion_x_to_denoised(self, x, sigma, uc, c, cfg_guidance, t):
        xc = self.calculate_input(x, sigma)
        noise_uc, noise_c = self.predict_noise(xc, t, uc, c)
        noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
        denoised = self.calculate_denoised(x, noise_pred, sigma)
        uncond_denoised = self.calculate_denoised(x, noise_uc, sigma)
        return denoised, uncond_denoised


###########################################
# Base version
###########################################

@register_solver("ddim")
class BaseDDIM(StableDiffusion):
    """
    Basic DDIM solver for SD.
    Useful for text-to-image generation
    """

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["", ""],
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent()
        zt = kwargs.get('zT', zt).requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # tweedie
            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_pred

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


@register_solver("ddim_np_naive")
@torch.no_grad()
class BaseDDIMnpNaive(StableDiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["", "", ""],
               **kwargs):
        # Text embedding
        uc, pc, nc = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1], negative_prompt=prompt[2])
        use_nc = prompt[2] != ""
        use_pc = prompt[1] != ""

        # Initialize zT
        zt = self.initialize_latent()
        zt = kwargs.get('zT', zt).requires_grad_()

        # Sampling
        pbar = self.scheduler.timesteps if kwargs.get('quiet', False) else tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_pc, noise_nc = self.predict_noise(zt, t, uc, pc, nc)
                noise_pred = noise_uc.clone().detach()
                if use_pc:  noise_pred += cfg_guidance * (noise_pc - noise_uc)
                if use_nc:  noise_pred -= cfg_guidance * (noise_nc - noise_uc)

            # tweedie
            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_pred

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


@register_solver("ddim_np_ccfg")
@torch.no_grad()
class BaseDDIMnpCCFG(StableDiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["", "", ""],
               **kwargs):
        # Text embedding
        uc, pc, nc = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1], negative_prompt=prompt[2])
        use_nc = prompt[2] != ""
        use_pc = prompt[1] != ""

        # Initialize zT
        zt = self.initialize_latent()
        zt = kwargs.get('zT', zt).requires_grad_()
        coeff = kwargs['coeff']

        # Sampling
        pbar = self.scheduler.timesteps if kwargs.get('quiet', False) else tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            tau = coeff['tau']
            with torch.no_grad():
                noise_uc, noise_pc, noise_nc = self.predict_noise(zt, t, uc, pc, nc)
                noise_pred = noise_uc.clone().detach()

                if use_pc:
                    if coeff['naive_positive_cfg']:     noise_pred += cfg_guidance * (noise_pc - noise_uc)
                    else:
                        l2norm = tau * ((noise_pc - noise_uc) ** 2).sum(dim=(1, 2, 3), keepdim=True)
                        noise_pred += cfg_guidance * (noise_pc - noise_uc) * 2 / (1 + torch.exp(-l2norm))

                l2norm = tau * ((noise_nc - noise_uc) ** 2).sum(dim=(1, 2, 3), keepdim=True)
                if use_nc:  noise_pred -= cfg_guidance * (noise_nc - noise_uc) * 2 * (torch.exp(-l2norm) / (1 + torch.exp(-l2norm)))

            # tweedie
            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_pred

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


if __name__ == "__main__":
    # print all list of solvers
    print(f"Possble solvers: {[x for x in __SOLVER__.keys()]}")
