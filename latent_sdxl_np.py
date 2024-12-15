from typing import Any, Optional, Tuple
import os
from safetensors.torch import load_file

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.models.attention_processor import (AttnProcessor2_0,
                                                  LoRAAttnProcessor2_0,
                                                  LoRAXFormersAttnProcessor,
                                                  XFormersAttnProcessor)
from tqdm import tqdm
import einops

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


########################

class SDXL():
    def __init__(self,
                 solver_config: dict,
                 model_key: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 dtype=torch.float16,
                 device='cuda'):

        self.device = device
        pipe = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=dtype).to(device)
        self.dtype = dtype

        # avoid overflow in float16
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.unet = pipe.unet

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        # sampling parameters
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.total_alphas = self.scheduler.alphas_cumprod.clone()
        N_ts = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = N_ts // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.sample(*args, **kwargs)

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def _text_embed(self, prompt, tokenizer, text_enc, clip_skip):
        text_inputs = tokenizer(
            prompt,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_enc(text_input_ids.to(self.device), output_hidden_states=True)

        pool_prompt_embeds = prompt_embeds[0]
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            # +2 because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
        return prompt_embeds, pool_prompt_embeds

    @torch.no_grad()
    def get_text_embed(self, null_prompt_1, prompt_1, neg_prompt_1, null_prompt_2=None, prompt_2=None, neg_prompt_2=None, clip_skip=None):
        '''
        At this time, assume that batch_size = 1.
        We should extend the code to batch_size > 1.
        '''
        # Encode the prompts
        # if prompt_2 is None, set same as prompt_1
        prompt_1 = [prompt_1] if isinstance(prompt_1, str) else prompt_1
        null_prompt_1 = [null_prompt_1] if isinstance(null_prompt_1, str) else null_prompt_1
        neg_prompt_1 = [neg_prompt_1] if isinstance(neg_prompt_1, str) else neg_prompt_1

        prompt_embed_1, pool_prompt_embed = self._text_embed(prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        if prompt_2 is None:
            prompt_embed = [prompt_embed_1]
        else:
            prompt_embed_2, pool_prompt_embed = self._text_embed(prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            prompt_embed = [prompt_embed_1, prompt_embed_2]

        null_embed_1, pool_null_embed = self._text_embed(null_prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        if null_prompt_2 is None:
            null_embed = [null_embed_1]
        else:
            null_embed_2, pool_null_embed = self._text_embed(null_prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            null_embed = [null_embed_1, null_embed_2]

        neg_embed_1, pool_neg_embed = self._text_embed(neg_prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        if neg_prompt_2 is None:
            neg_embed = [neg_embed_1]
        else:
            neg_embed_2, pool_neg_embed = self._text_embed(neg_prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            neg_embed = [neg_embed_1, neg_embed_2]

        # concat embeds from two encoders
        null_prompt_embeds = torch.concat(null_embed, dim=-1)
        prompt_embeds = torch.concat(prompt_embed, dim=-1)
        neg_prompt_embeds = torch.concat(neg_embed, dim=-1)

        return null_prompt_embeds, prompt_embeds, neg_prompt_embeds, pool_null_embed, pool_prompt_embed, pool_neg_embed

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae

    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor

        # @torch.no_grad()

    def decode(self, zt):
        # make sure the VAE is in float32 mode, as it overflows in float16
        # needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        # if needs_upcasting:
        #     self.upcast_vae()
        #     zt = zt.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        image = self.vae.decode(zt / self.vae.config.scaling_factor).sample.float()
        return image

    def predict_noise(self, zt, t, uc, c, nc, added_cond_kwargs):
        t_in = t.unsqueeze(0)
        t_in, uc, c, nc = t_in.expand(zt.shape[0]), uc.expand(zt.shape[0], -1, -1), c.expand(zt.shape[0], -1, -1), nc.expand(zt.shape[0], -1, -1)
        if t_in.shape[0]*3 != added_cond_kwargs['text_embeds'].shape[0]:
            added_cond_kwargs['text_embeds'] = einops.repeat(added_cond_kwargs['text_embeds'], 'p f -> (p b) f', b=zt.shape[0])
            added_cond_kwargs['time_ids'] = einops.repeat(added_cond_kwargs['time_ids'], 'p f -> (p b) f', b=zt.shape[0])
        # print(t_in.shape, uc.shape, c.shape, nc.shape, added_cond_kwargs['text_embeds'].shape, added_cond_kwargs['time_ids'].shape)

        if uc is None:
            noise_c = self.unet(zt, t_in, encoder_hidden_states=c,
                                added_cond_kwargs=added_cond_kwargs)['sample']
            noise_uc = noise_c
        elif c is None:
            noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc,
                                 added_cond_kwargs=added_cond_kwargs)['sample']
            noise_c = noise_uc
        else:
            c_embed = torch.cat([uc, c, nc], dim=0)
            z_in = torch.cat([zt] * 3)
            t_in = torch.cat([t_in] * 3)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed,
                                   added_cond_kwargs=added_cond_kwargs)['sample']
            noise_uc, noise_c, noise_nc = noise_pred.chunk(3)
        return noise_uc, noise_c, noise_nc

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        assert expected_add_embed_dim == passed_add_embed_dim, (
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               prompt1=["", "", ""],
               prompt2=["", "", ""],
               cfg_guidance: float = 5.0,
               original_size: Optional[Tuple[int, int]] = None,
               crops_coords_top_left: Tuple[int, int] = (0, 0),
               target_size: Optional[Tuple[int, int]] = None,
               negative_original_size: Optional[Tuple[int, int]] = None,
               negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
               negative_target_size: Optional[Tuple[int, int]] = None,
               clip_skip: Optional[int] = None,
               **kwargs):

        # 0. Default height and width to unet
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # embedding
        (null_prompt_embeds,
         prompt_embeds,
         neg_prompt_embeds,
         pool_null_embed,
         pool_prompt_embed,
         pool_neg_embed) = self.get_text_embed(prompt1[0], prompt1[1], prompt1[2], prompt2[0], prompt2[1], prompt2[2], clip_skip)

        # prepare kwargs for SDXL
        add_text_embeds = pool_prompt_embed
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
        )

        add_text_embeds_neg = pool_neg_embed
        add_time_ids_neg = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=int(pool_neg_embed.shape[-1]),
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
            )
        else:
            negative_add_time_ids = add_time_ids
        negative_text_embeds = pool_null_embed

        if cfg_guidance != 0.0 and cfg_guidance != 1.0:
            # do cfg
            add_text_embeds = torch.cat([negative_text_embeds, add_text_embeds, add_text_embeds_neg], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids, add_time_ids_neg], dim=0)

        add_cond_kwargs = {
            'text_embeds': add_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }

        # reverse sampling
        zt = self.reverse_process(null_prompt_embeds, prompt_embeds, neg_prompt_embeds, cfg_guidance, add_cond_kwargs, target_size, **kwargs)

        # decode
        with torch.no_grad():
            img = self.decode(zt)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

    def initialize_latent(self,
                          method: str = 'random',
                          src_img: Optional[torch.Tensor] = None,
                          add_cond_kwargs: Optional[dict] = None,
                          **kwargs):
        if method == 'ddim':
            assert src_img is not None, "src_img must be provided for inversion"
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('uc'),
                               kwargs.get('c'),
                               kwargs.get('cfg_guidance', 0.0),
                               add_cond_kwargs)
        elif method == 'npi':
            assert src_img is not None, "src_img must be provided for inversion"
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('c'),
                               kwargs.get('c'),
                               1.0,
                               add_cond_kwargs)
        elif method == 'random':
            size = kwargs.get('size', (1, 4, 128, 128))
            z = torch.randn(size).to(self.device)
        else:
            raise NotImplementedError

        return z.requires_grad_()

    def inversion(self, z0, uc, c, cfg_guidance, add_cond_kwargs):
        # if we use cfg_guidance=0.0 or 1.0 for inversion, add_cond_kwargs must be splitted.
        if cfg_guidance == 0.0 or cfg_guidance == 1.0:
            add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
            add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)

        zt = z0.clone().to(self.device)
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1 - at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1 - at).sqrt() * noise_pred

        return zt

    def reverse_process(self, *args, **kwargs):
        raise NotImplementedError


###########################################
# Base version
###########################################
@register_solver('ddim_np_naive')
class BaseDDIMnpNaive(SDXL):
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds, neg_prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        zt = kwargs.get('zT', zt).requires_grad_()

        # sampling
        pbar = self.scheduler.timesteps.int() if kwargs.get('quiet', False) else tqdm(self.scheduler.timesteps.int(), desc="SD")
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            with torch.no_grad():
                noise_uc, noise_c, noise_nc = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, neg_prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
                noise_pred -= cfg_guidance * (noise_nc - noise_uc)

            # tweedie
            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1 - at_next).sqrt() * noise_pred

        # for the last stpe, do not add noise
        return z0t


@register_solver('ddim_np_ccfg')
class BaseDDIMnpCCFG(SDXL):
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds, neg_prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        zt = kwargs.get('zT', zt).requires_grad_()
        coeff = kwargs['coeff']

        # sampling
        pbar = self.scheduler.timesteps.int() if kwargs.get('quiet', False) else tqdm(self.scheduler.timesteps.int(), desc="SD")
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            tau = coeff['tau']
            with torch.no_grad():
                noise_uc, noise_pc, noise_nc = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, neg_prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc.clone().detach()

                if coeff['naive_positive_cfg']:     noise_pred += cfg_guidance * (noise_pc - noise_uc)
                else:
                    l2norm = tau * ((noise_pc - noise_uc) ** 2).sum(dim=(1, 2, 3), keepdim=True)
                    noise_pred += cfg_guidance * (noise_pc - noise_uc) * 2 / (1 + torch.exp(-l2norm))

                l2norm = tau * ((noise_nc - noise_uc) ** 2).sum(dim=(1, 2, 3), keepdim=True)
                noise_pred -= cfg_guidance * (noise_nc - noise_uc) * 2 * (torch.exp(-l2norm) / (1 + torch.exp(-l2norm)))  # * tau

            # tweedie
            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1 - at_next).sqrt() * noise_pred

        # for the last stpe, do not add noise
        return z0t


@register_solver('ddim_np_dng')
class BaseDDIMnpDNG(SDXL):
    def update_lambda(self, x, x_next, predict_epsilon, predict_epsilon_uc, idx, lambda_, at, at_prev, d, tau):
        sigma = 1-at/at_prev
        mu_c = (x - predict_epsilon*sigma/torch.sqrt(1-at)) / torch.sqrt(at/at_prev)
        mu_null = (x - predict_epsilon_uc*sigma/torch.sqrt(1-at)) / torch.sqrt(at/at_prev)
        l2diff_c = (mu_c - x_next).pow(2).sum(dim=(-3 ,-2, -1))
        l2diff_null = (mu_null - x_next).pow(2).sum(dim=(-3, -2, -1))
        new_lambda = lambda_ * torch.exp(-(tau*(l2diff_c - l2diff_null)-d)/(2*sigma))
        new_lambda = new_lambda.clamp(0.0001, 0.5)
        return new_lambda

    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds, neg_prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        zt = kwargs.get('zT', zt).requires_grad_()
        coeff = kwargs['coeff']

        # sampling
        pbar = self.scheduler.timesteps.int() if kwargs.get('quiet', False) else tqdm(self.scheduler.timesteps.int(), desc="SD")
        lambda_ = torch.ones(zt.size(0)).to(self.device) * coeff['p_c_init']    # DNG

        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            w = cfg_guidance * lambda_ / (1 - lambda_)  # DNG
            zt_prev = zt.clone().detach()

            with torch.no_grad():
                noise_uc, noise_c, noise_nc = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, neg_prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
                noise_pred -= w[:, None, None, None] * (noise_nc - noise_uc)  # DNG

            # tweedie
            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1 - at_next).sqrt() * noise_pred

            lambda_ = self.update_lambda(zt_prev, zt, noise_nc, noise_uc, t, lambda_, at, at_next, d=coeff['delta'], tau=coeff['tau'])       # DNG
        # for the last stpe, do not add noise
        return z0t


#############################

if __name__ == "__main__":
    # print all list of solvers
    print(f"Possble solvers: {[x for x in __SOLVER__.keys()]}")