import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from math import expm1, log
from functools import partial
from einops import repeat, rearrange, reduce
from typing import Any, Callable, List, Optional, Tuple, Union
from prompt_tts.variation_net.utils import (default, exists, simple_linear_schedule, 
                                            cosine_schedule, sigmoid_schedule, 
                                            right_pad_dims_to, gamma_to_alpha_sigma, 
                                            safe_div, gamma_to_log_snr, prob_mask_like,
                                            pad_or_curtail_to_length, Sequential)
from prompt_tts.variation_net.utils import (LearnedSinusoidalPosEmb, FilM)
from prompt_tts.ref_encoder.utils import Attention, FeedForward, RMSNorm

class Model(nn.Module):
    def __init__(self, dim, *, depth, causal=False, dim_head=64, heads=8,
                 use_flash=False, dropout=0., ff_mult=4, final_norm=False,
                 dim_cond_mult=4):
        super().__init__()
        self.dim = dim
        dim_time = dim * dim_cond_mult
        
        self.to_time_cond = Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim + 1, dim_time),
            nn.SiLU()
        )
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RMSNorm(dim),
                FilM(dim, dim_time),
                Attention(dim, causal=causal, dim_head=dim_head, heads=heads,
                          dropout=dropout, use_flash=use_flash),
                RMSNorm(dim),
                FilM(dim, dim_time),
                FeedForward(dim, mult=ff_mult),
                nn.Dropout(dropout)]))
        self.norm = RMSNorm(dim) if final_norm else nn.Identity()
        self.film = FilM()
    
    '''
    def forward_with_cond_scale(self, *args, cond_scale=1., **kwargs):
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)
        if cond_scale == 1.:
            return logits
        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        
        return null_logits + (logits - null_logits) * cond_scale
    '''
        
    def forward(self, x, t, cond=None, cond_drop_prob=None):
        b = x.shape[0]
        t = self.to_time_cond(t)
        
        if exists(self.cond_to_model_dim):
            assert exists(cond)
            cond = self.cond_to_model_dim(cond)
            cond_drop_mask = prob_mask_like((b,), cond_drop_prob, self.device)
            cond = torch.where(
                rearrange(cond_drop_mask, 'b -> b 1 1'),
                self.null_cond,
                cond
            )
            cond = pad_or_curtail_to_length(cond, x.shape[-1])
            x += cond
            
        for attn_norm, film_1, attn, ff_norm, film_2, ff, dropout in self.layers:
            res = x
            x = attn(film_1(attn_norm(x), t)) + res
            res = x
            x = dropout(ff(film_2(ff_norm(x), t))) + res  
        
        return self.film(self.norm(x), t)
            
class Variation_Net(nn.Module):
    
    def __init__(self, model: Model, timesteps=1000, use_ddim=True, noise_schedule='sigmoid', 
                 objective='v', schedule_kwargs:dict = dict(), time_difference=0., scale=1.):
        self.model = model
        assert objective in {'x0', 'eps', 'v'}, 'objective must be either predict x0 or noise'
        self.objective = objective
        
        if noise_schedule == 'linear':
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == 'cosine':
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == 'sigmoid':
            self.gamma_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        
        assert scale <= 1, 'scale must be less than or equal to 1'
        self.scale = scale
        
        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)
        
        self.timesteps = timesteps
        self.use_ddim = use_ddim
        
        self.time_difference = time_difference
        
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device=device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times
    
    @torch.no_grad()
    def ddpm_sample(self, shape, time_difference=None, cond_scale=1., cond=None):
        batch, device = shape[0], self.device
        time_difference = default(time_difference, self.time_difference)
        time_pairs = self.get_sampling_timesteps(batch, device=device)
        
        audio = torch.randn(shape, device=device)
        
        x_start = None
        last_latents = None
        
        for time, time_next in tqdm(time_pairs, total=self.timesteps):
            time_next = (time_next - self.time_difference).clamp(min=0.)
            noise_cond = time
            
            model_output = self.model(audio, noise_cond, cond_scale=cond_scale, cond=cond)
            
            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)
            gamma, gamma_next = map(partial(right_pad_dims_to, audio), (gamma, gamma_next))
            
            alpha, sigma = gamma_to_alpha_sigma(gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next, self.scale)
            
            if self.objective == 'x0':
                x_start = model_output
            elif self.objective == 'eps':
                x_start = safe_div(audio - sigma * model_output, alpha)
            elif self.objective == 'v':
                x_start = alpha * audio - sigma * model_output
                
            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))
            
            c = -expm1(log_snr - log_snr_next)
            
            mean = alpha_next * (audio * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)
            
            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1 '),
                torch.randn_like(audio),
                torch.zeros_like(audio)
            )
            
            audio = mean + (0.5 * log_variance).exp() * noise
            
        return audio
    
    @torch.no_grad()
    def ddim_sample(self, shape, time_difference=None, cond_scale=1., cond=None):
        batch, device = shape[0], self.device
        time_difference = default(time_difference, self.time_difference)
        time_pairs = self.get_sampling_timesteps(batch, device=device)
        
        audio = torch.randn(shape, device=device)
        
        x_start = None
        last_latents = None
        
        for times, times_next in tqdm(time_pairs):
            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)
            
            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, audio),(gamma, gamma_next))
            
            alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next, self.scale)
            
            times_next = (times_next - time_difference).clamp(min = 0.)
            
            model_output = self.model(audio, times, cond_scale=cond_scale, cond=cond)
            
            if self.objective == 'x0':
                x_start = model_output
            elif self.objective == 'eps':
                x_start = safe_div(audio - sigma * model_output, alpha)
            elif self.objective == 'v':
                x_start = alpha * audio - sigma * model_output
            
            pred_noise = safe_div(audio - alpha * x_start, sigma)
            
            audio = x_start * alpha_next + pred_noise * sigma_next
            
        return audio
    
    @torch.no_grad()
    def sample(self, *, length, batch_size=1, cond_scale=1.,
               prompt_rep=None, prompt_lens=None):
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        ref_rep = sample_fn((batch_size, length, self.dim),
                            cond=prompt_rep, cond_scale=cond_scale)
        return ref_rep
    
    def forward(self, ref_rep, prompt_rep):
        batch, n, d, device = *ref_rep.shape, self.device
        
        times = torch.zeros((batch,), device=device).float().uniform_(0, 1.)
        
        noise = torch.randn_like(ref_rep)
        
        gamma = self.gamma_schedule(times)
        padded_gamma = right_pad_dims_to(ref_rep, gamma)
        alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)
        
        noised_ref = alpha * ref_rep + sigma * noise
        
        pred = self.model(noised_ref, times, cond=prompt_rep)
        
        if self.objective == 'eps':
            target = noise
        elif self.objective == 'x0':
            target = ref_rep
        elif self.objective == 'v':
            target = alpha * noise - sigma * ref_rep
        
        loss = torch.nn.L1Loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        
        return loss.mean()