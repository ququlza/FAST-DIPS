from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import setproctitle
import torch
import torch.func as torch_func
import torch.nn as nn
import tqdm
import wandb
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import get_dataset
from forward_operators import get_operator
from model import get_model
from misc import (
    # Presets extracted from upstream YAMLs (configs/*/*.yaml)
    DATA_PRESETS,
    MODEL_PRESETS,
    SAMPLER_PRESETS,
    TASK_PRESETS,

    # Scheduler registry + evaluation + utilities
    Trajectory,
    get_diffusion_scheduler,
    Evaluator,
    get_eval_fn,
    get_eval_fn_cmp,
    calculate_fid,
    resize,
    safe_dir,
    tensor_to_pils,
    save_mp4_video,
)



def _flatten_inner_prod(a: torch.Tensor, b: torch.Tensor):
    """Compute batchwise inner product <a,b> -> (B,1,1,1)."""
    return (a * b).flatten(1).sum(dim=1, keepdim=True).view(-1, 1, 1, 1)


def _l2_norm_per_sample(t: torch.Tensor):
    """Compute L2 norm per sample -> (B,1,1,1)."""
    return t.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1)


def x_update_autograd(A, x, x0, b, rho, gamma, steps=1, alpha_max=1.0, backtrack=True,
                      eps: float = 1e-3, use_exact_jvp=False):
    """
    Minimize F(x) = (1/(2γ))||x-x0||^2 + (ρ/2)||A(x)-b||^2 by a few steps:
      g  = (x-x0)/γ + ρ J^T (A(x)-b)
      α* = argmin along -g using linearized exact formula with JVP
         = ( <x-x0,g>/γ + ρ <r,Jg> ) / ( ||g||^2/γ + ρ ||Jg||^2 )
    """
    x = x.clone()
    for _ in range(max(1, steps)):
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            Ax = A(x_req)
            r = Ax - b
            v = r
            g_data = torch.autograd.grad(outputs=Ax, inputs=x_req, grad_outputs=v,
                                         create_graph=False, retain_graph=False, allow_unused=False)[0]
        g = (x - x0) / gamma + rho * g_data
        g = torch.nan_to_num(g)
        if use_exact_jvp:
            def f(u):
                return A(u)
        
            _, Jg = torch_func.jvp(f, (x.detach(),), (g.detach(),))

            num = _flatten_inner_prod(x - x0, g) / gamma + rho * _flatten_inner_prod(r.detach(), Jg)
            den = _flatten_inner_prod(g, g) / gamma + rho * _flatten_inner_prod(Jg, Jg) + 1e-12
            alpha = (num / den).clamp(0, alpha_max)
        else:
            A_step = A(x + eps * g)
            dA = A_step - Ax
            numer = _flatten_inner_prod(r, dA)
            denom = _flatten_inner_prod(dA, dA) + 1e-12
            alpha = (numer / denom) * eps
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            alpha = alpha.clamp(min=0, max=alpha_max)

        if backtrack:
            def F_energy(zcur):
                rz = A(zcur) - b
                return (_flatten_inner_prod(rz, rz) * (rho / 2.0) + _flatten_inner_prod(zcur - x0, zcur - x0) * (
                        1.0 / (2.0 * gamma)))

            f0 = F_energy(x)
            x1 = x - alpha * g
            f1 = F_energy(x1)
            mask = (f1 > f0)
            tries = 0
            while mask.any() and tries < 8:
                alpha[mask] *= 0.5
                x1 = x - alpha * g
                f1 = F_energy(x1)
                mask = (f1 > f0)
                tries += 1
        
        x = x - alpha * g
    return x.detach()


def z_update_composite_autograd(decode_fn, operator, z, z0, b, rho, gamma, steps=1, alpha_max=1.0,
                                backtrack=True):
    """
    Minimize H(z) = (1/(2γ))||z - z0||^2 + (ρ/2)||A(Dec(z)) - b||^2 by a few steps.

    Uses autodiff for VJP and JVP to compute an exact line-search step along -g under linearization.
    """
    z = z.clone()
    for _ in range(max(1, steps)):
        with torch.enable_grad():
            z_req = z.detach().requires_grad_(True)
            Dz = decode_fn(z_req)
            Fz = operator(Dz)
            r = Fz - b
            # VJP: J^T r where J = dF/dz
            g_data = torch.autograd.grad(outputs=Fz, inputs=z_req, grad_outputs=r,
                                         create_graph=False, retain_graph=False, allow_unused=False)[0]
        g = (z - z0) / gamma + rho * g_data
        g = torch.nan_to_num(g)

        def F(u):
            return operator(decode_fn(u))

        _, Jg = torch.func.jvp(F, (z.detach(),), (g.detach(),))

        num = _flatten_inner_prod(z - z0, g) / gamma + rho * _flatten_inner_prod(r.detach(), Jg)
        den = _flatten_inner_prod(g, g) / gamma + rho * _flatten_inner_prod(Jg, Jg) + 1e-12
        alpha = (num / den).clamp(0.0, alpha_max)

        if backtrack:
            def H_energy(zcur):
                Dzcur = decode_fn(zcur)
                rz = operator(Dzcur) - b
                return (_flatten_inner_prod(rz, rz) * (rho / 2.0) +
                        _flatten_inner_prod(zcur - z0, zcur - z0) * (1.0 / (2.0 * gamma)))

            f0 = H_energy(z)
            z1 = z - alpha * g
            f1 = H_energy(z1)
            mask = (f1 > f0)
            tries = 0
            while mask.any() and tries < 8:
                alpha[mask] *= 0.5
                z1 = z - alpha * g
                f1 = H_energy(z1)
                mask = (f1 > f0)
                tries += 1
        
        z = z - alpha * g
    return z.detach()


def pca_torch(X):
    """Compute PCA eigenvalues of input tensor."""
    X_centered = X - X.mean(dim=0)
    cov_matrix = torch.mm(X_centered.T, X_centered) / (X_centered.shape[0] - 1)
    eigenvalues, _ = torch.linalg.eigh(cov_matrix)
    return eigenvalues


def estimate_noise_level(eigenvalues):
    """Estimate noise level from PCA eigenvalues."""
    S = eigenvalues.tolist()
    while True:
        mean = sum(S) / len(S)
        median = statistics.median(S)
        if abs(mean - median) < 1e-6:
            break
        S.remove(max(S))
    return torch.sqrt(torch.tensor(mean))


def extract_patches(image, patch_size):
    """Extract patches from image tensor."""
    unfold = torch.nn.functional.unfold(image.unsqueeze(0), kernel_size=patch_size)
    patches = unfold.squeeze(0).transpose(0, 1)
    return patches


def estimate_noise_level_pca(batch, patch_size=8):
    """Estimate noise level using PCA for each sample in batch."""
    batch_size = batch.shape[0]
    noise_levels = torch.zeros(batch_size)

    for i in range(batch_size):
        patches = extract_patches(batch[i], patch_size)
        eigenvalues = pca_torch(patches)
        noise_level = estimate_noise_level(eigenvalues)
        noise_levels[i] = noise_level

    return noise_levels.to(batch.device)


def get_sampler(**kwargs):
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        return LatentFAST_DIPS(**kwargs)
    return FAST_DIPS(**kwargs)


class FAST_DIPS(nn.Module):
    def __init__(self, diffusion_scheduler_config,
                 # step size options
                 backtrack=True, alpha_max=1.0,
                 # ADMM parameters (pixel-space)
                 rho=200, K=3, S=1,
                 # constraint parameters
                 epsilon=0.05,
                 ):
        """
        Args:
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
            backtrack (bool): Enable backtracking line search.
            alpha_max (float): Maximum step size.
            rho (float): ADMM augmented Lagrangian penalty parameter.
            K (int): Number of ADMM iterations per noise level.
            S (int): Number of gradient steps in x-update per ADMM iteration.
            epsilon (float): Hard constraint tolerance.
        """
        super().__init__()
        self.diffusion_scheduler = get_diffusion_scheduler(**diffusion_scheduler_config)
        # step size options
        self.backtrack = backtrack
        self.alpha_max = alpha_max

        # ADMM params (pixel-space)
        self.rho = float(rho)
        self.K = int(K)
        self.S = int(S)
        self.epsilon = float(epsilon)

    @torch.no_grad()
    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Args:
            model (nn.Module): Diffusion model.
            x_start (torch.Tensor): Initial tensor/state.
            operator (nn.Module): Measurement operator.
            measurement (torch.Tensor): Observed measurement tensor.
            evaluator (Evaluator, optional): Evaluator for performance metrics.
            record (bool, optional): If True, records the sampling trajectory.
            verbose (bool, optional): Enables progress bar and logs.
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation.

        Returns:
            torch.Tensor: Final sampled tensor/state.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.diffusion_scheduler.num_steps - 1) if verbose else range(
            self.diffusion_scheduler.num_steps - 1)
        xt = x_start

        dcdtype = torch.float32
        measurement_dc = measurement.to(dcdtype)

        total_sampling_time = 0.0
        device = xt.device
        cuda_device = device if device.type == 'cuda' else None


        for step in pbar:
            sigma, sigma_next = self.diffusion_scheduler.sigma_steps[step], self.diffusion_scheduler.sigma_steps[
                step + 1]

            if cuda_device is not None:
                torch.cuda.reset_peak_memory_stats(cuda_device)
            timer_state = _start_timer(cuda_device)

            x0hat = model.tweedie(xt, sigma)
            denoised = torch.nan_to_num(x0hat)

            x0 = denoised.to(dcdtype)

            x = x0.clone()
            u = torch.zeros_like(measurement_dc)
            v = operator(x0)

            gamma_t = (sigma ** 2)

            for _ in range(max(1, self.K)):
                b = v - u
                x = x_update_autograd(
                    operator,
                    x,
                    x0,
                    b,
                    rho=self.rho,
                    gamma=gamma_t,
                    steps=self.S,
                    alpha_max=self.alpha_max,
                    backtrack=self.backtrack,
                )
                Ax = operator(x)

                w = Ax + u
                d = w - measurement_dc
                nrm = _l2_norm_per_sample(d).clamp_min(1e-12)
                scale = (self.epsilon / nrm).clamp_max(1.0)
                v = torch.where((nrm <= self.epsilon), w, measurement_dc + scale * d)

                u = u + Ax - v

            x0y = x.to(denoised.dtype)
            x0y = torch.nan_to_num(x0y)

            if step == self.diffusion_scheduler.num_steps - 2 and operator.name != "high_dynamic_range":
                delta = estimate_noise_level_pca(x0y, patch_size=4)
                delta = delta.to(x0y.dtype)

                if hasattr(operator, "mask"):

                    if (delta > 0.15).any():
                        x0y = model.tweedie(x0y, delta)
                    else:
                        sig = (0.15 ** 2 - delta ** 2).sqrt()
                        x0y = x0y + (0.15 - sig).view(-1, 1, 1, 1) * (1 - operator.mask) * torch.randn_like(x0y) \
                              + sig.view(-1, 1, 1, 1) * torch.randn_like(x0y)
                        x0y = model.tweedie(x0y, torch.full((x0y.size(0),), 0.15, device=x0y.device, dtype=x0y.dtype))
                else:
                    if (delta > 0.15).any():
                        x0y = model.tweedie(x0y, delta)
                    else:
                        sig = (0.15 ** 2 - delta ** 2).sqrt()
                        x0y = x0y + sig.view(-1, 1, 1, 1) * torch.randn_like(x0y)
                        x0y = model.tweedie(x0y, torch.full((x0y.size(0),), 0.15, device=x0y.device, dtype=x0y.dtype))

            xt = x0y + sigma_next * torch.randn_like(x0y)

            # if step == self.diffusion_scheduler.num_steps - 2:
            #     xt = x0hat

            elapsed = _stop_timer(timer_state)
            total_sampling_time += elapsed


            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)
                    torch.cuda.empty_cache()

                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)

        self.last_sampling_time = total_sampling_time
        return xt.clamp(-1, 1)

    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        """Records the intermediate states during sampling."""

        self.trajectory.add_tensor("xt", xt)
        self.trajectory.add_tensor("x0y", x0y)
        self.trajectory.add_tensor("x0hat", x0hat)
        self.trajectory.add_value("sigma", sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def get_start(self, batch_size, model):
        """
        Generates initial random state tensors from the Gaussian prior.

        Args:
            batch_size (int): Number of initial states to generate.
            model (nn.Module): Diffusion or latent diffusion model.

        Returns:
            torch.Tensor: Random initial tensor.
        """
        device = next(model.parameters()).device
        in_shape = model.get_in_shape()
        x_start = torch.randn(batch_size, *in_shape, device=device) * self.diffusion_scheduler.get_prior_sigma()
        return x_start


class LatentFAST_DIPS(FAST_DIPS):
    def __init__(self, diffusion_scheduler_config,
                 backtrack=True, alpha_max=1.0, sigma_switch=1.0,
                 rho_x=200, K_x=5, S_x=3,
                 rho_z=200, K_z=5, S_z=3,
                 epsilon=0.05):
        """
        Args:
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
            backtrack (bool): Enable backtracking line search.
            alpha_max (float): Maximum step size.
            sigma_switch (float): Hybrid switching parameter.
            rho_x (float): ADMM penalty for x-space branch.
            K_x (int): ADMM iterations for x-space branch.
            S_x (int): Gradient steps per x-update in x-space branch.
            rho_z (float): ADMM penalty for z-space branch.
            K_z (int): ADMM iterations for z-space branch.
            S_z (int): Gradient steps per z-update in z-space branch.
            epsilon (float): Hard-constraint tolerance for both branches.
        """
        super().__init__(diffusion_scheduler_config)

        # Step-size options
        self.backtrack = backtrack
        self.alpha_max = alpha_max

        # Hybrid branch switch
        self.sigma_switch = float(sigma_switch)

        # Credible-set radius
        self.epsilon = float(epsilon)

        # ADMM parameters for x-space branch
        self.rho_x = float(rho_x)
        self.K_x = int(K_x)
        self.S_x = int(S_x)

        # ADMM parameters for z-space branch
        self.rho_z = float(rho_z)
        self.K_z = int(K_z)
        self.S_z = int(S_z)

    @torch.no_grad()
    def sample(self, model, z_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Args:
            model (LatentDiffusionModel): Latent diffusion model.
            z_start (torch.Tensor): Initial latent state tensor.
            operator (nn.Module): Measurement operator applied in data space.
            measurement (torch.Tensor): Observed measurement tensor.
            evaluator (Evaluator, optional): Evaluator for monitoring performance.
            record (bool, optional): Whether to record intermediate states and metrics.
            verbose (bool, optional): Enables progress bar and evaluation metrics.
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation.

        Returns:
            torch.Tensor: Final sampled data decoded from latent space.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.diffusion_scheduler.num_steps - 1) if verbose else range(
            self.diffusion_scheduler.num_steps - 1)

        zt = z_start
        dcdtype = torch.float32
        measurement_dc = measurement.to(dcdtype)
        total_sampling_time = 0
        device = zt.device
        cuda_device = device if device.type == 'cuda' else None

        for step in pbar:
            if cuda_device is not None:
                torch.cuda.reset_peak_memory_stats(cuda_device)
            timer_state = _start_timer(cuda_device)

            sigma, sigma_next = self.diffusion_scheduler.sigma_steps[step], self.diffusion_scheduler.sigma_steps[
                step + 1]
            z0hat = model.tweedie(zt, sigma)

            is_pixel_branch = sigma > self.sigma_switch
            if is_pixel_branch:
                x0hat = model.decode(z0hat).to(dcdtype)
                x = x0hat.clone()

                gamma_t_x = (sigma ** 2)

                u = torch.zeros_like(measurement_dc)
                v = operator(x0hat)

                for _ in range(self.K_x):
                    b = v - u
                    x = x_update_autograd(
                        operator,
                        x,
                        x0hat,
                        b,
                        rho=self.rho_x,
                        gamma=gamma_t_x,
                        steps=self.S_x,
                        alpha_max=self.alpha_max,
                        backtrack=self.backtrack,
                        use_exact_jvp=True,
                    )

                    Ax = operator(x)
                    w = Ax + u
                    d = w - measurement_dc
                    nrm = _l2_norm_per_sample(d).clamp_min(1e-12)
                    scale = (self.epsilon / nrm).clamp_max(1.0)
                    v = torch.where((nrm <= self.epsilon), w, measurement_dc + scale * d)

                    u = u + Ax - v
                x = torch.clamp(x, -1., 1.)
                z0 = model.encode(x)

            else:
                z0 = z0hat.clone().detach()
                gamma_t_z = (sigma ** 2)
                
                with torch.enable_grad():
                    Decz0 = model.decode(z0.detach()).detach().to(dcdtype)
                v = operator(Decz0)

                u = torch.zeros_like(measurement_dc)

                for _ in range(self.K_z):
                    b = v - u
                    z0 = z_update_composite_autograd(
                        model.decode,
                        operator,
                        z0,
                        z0hat,
                        b,
                        rho=self.rho_z,
                        gamma=gamma_t_z,
                        steps=self.S_z,
                        alpha_max=self.alpha_max,
                        backtrack=self.backtrack,
                    )
                    with torch.no_grad():
                        Decz = model.decode(z0.detach()).detach().to(dcdtype)
                        Ax = operator(Decz)

                    w = Ax + u
                    d = w - measurement_dc
                    nrm = _l2_norm_per_sample(d).clamp_min(1e-12)
                    scale = (self.epsilon / nrm).clamp_max(1.0)
                    v = torch.where((nrm <= self.epsilon), w, measurement_dc + scale * d)
                
                    u = u + Ax - v

            z_star = z0

            if step != self.diffusion_scheduler.num_steps - 2:
                zt = z_star + sigma_next * torch.randn_like(z_star)
            else:
                zt = z_star

            elapsed = _stop_timer(timer_state)
            total_sampling_time += elapsed

            with torch.no_grad():
                x0hat = model.decode(z0hat)
                x0y = model.decode(z_star)
                xt = model.decode(zt)

            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)
                    torch.cuda.empty_cache()

            if verbose and evaluator and 'gt' in kwargs:
                main_eval_fn_name = evaluator.main_eval_fn_name
                pbar.set_postfix({
                    'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                    'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                })
            if record:
                with torch.no_grad():
                    x0hat_record = model.decode(z0)
                self._record(xt, x0y, x0hat_record, sigma, x0hat_results, x0y_results)

        self.last_sampling_time = total_sampling_time

        if 'xt' not in locals():
            with torch.no_grad():
                xt = model.decode(zt)

        return xt.clamp(-1, 1)
def _start_timer(device):
    """Start a CUDA (or CPU fallback) timer and return state."""
    if device is not None and getattr(device, 'type', None) == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
        return ('cuda', device, start_event, end_event)
    return ('cpu', time.perf_counter())


def _stop_timer(timer_state):
    """Stop a timer started by _start_timer and return elapsed seconds."""
    mode = timer_state[0]
    if mode == 'cuda':
        _, device, start_event, end_event = timer_state
        end_event.record()
        torch.cuda.synchronize(device)
        return start_event.elapsed_time(end_event) / 1000.0
    return time.perf_counter() - timer_state[1]

def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean.")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FAST-DIPS")

    # General
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", type=str, default="demo")
    p.add_argument("--save_dir", type=str, default="./results")
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--num_runs", type=int, default=1)

    p.add_argument("--wandb", type=_str2bool, default=False)
    p.add_argument("--project_name", type=str, default="FAST-DIPS")

    p.add_argument("--save_samples", type=_str2bool, default=True)
    p.add_argument("--save_traj", type=_str2bool, default=True)
    p.add_argument("--save_traj_video", type=_str2bool, default=False)
    p.add_argument("--save_traj_raw_data", type=_str2bool, default=False)

    p.add_argument("--eval_fid", type=_str2bool, default=False)
    p.add_argument("--eval_fn_list", type=str, default="psnr,ssim,lpips")

    # Preset selectors
    p.add_argument("--data", type=str, choices=sorted(DATA_PRESETS.keys()), default="demo-ffhq")
    p.add_argument("--model", type=str, choices=sorted(MODEL_PRESETS.keys()), default="ffhq256ddpm")
    p.add_argument("--sampler", type=str, choices=sorted(SAMPLER_PRESETS.keys()), default="edm_FAST_DIPS")
    p.add_argument("--task", type=str, choices=sorted(TASK_PRESETS.keys()), default="phase_retrieval")

    # Data overrides
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--data_resolution", type=int, default=None)
    p.add_argument("--data_start_id", type=int, default=None)
    p.add_argument("--data_end_id", type=int, default=None)

    # Sampler overrides
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--diffusion_sigma_max", type=float, default=None)
    p.add_argument("--diffusion_sigma_min", type=float, default=None)

    # FAST-DIPS (common) overrides
    p.add_argument("--backtrack", type=_str2bool, default=True)
    p.add_argument("--alpha_max", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.05)

    # FAST-DIPS overrides
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--S", type=int, default=None)

    # LatentFAST-DIPS overrides
    p.add_argument("--sigma_switch", type=float, default=None)
    p.add_argument("--rho_x", type=float, default=None)
    p.add_argument("--K_x", type=int, default=None)
    p.add_argument("--S_x", type=int, default=None)
    p.add_argument("--rho_z", type=float, default=None)
    p.add_argument("--K_z", type=int, default=None)
    p.add_argument("--S_z", type=int, default=None)

    # Operator overrides (most common)
    p.add_argument("--operator_sigma", type=float, default=None)
    p.add_argument("--phase_oversample", type=float, default=None)
    p.add_argument("--down_scale_factor", type=int, default=None)
    p.add_argument("--inpaint_mask_len", type=int, default=None)  # for box inpainting
    p.add_argument("--random_inpaint_prob", type=float, default=None)
    p.add_argument("--gaussian_kernel_size", type=int, default=None)
    p.add_argument("--gaussian_intensity", type=float, default=None)
    p.add_argument("--motion_kernel_size", type=int, default=None)
    p.add_argument("--motion_intensity", type=float, default=None)
    p.add_argument("--hdr_scale", type=float, default=None)
    p.add_argument("--bkse_opt_yml_path", type=str, default=None)

    # Model overrides (checkpoint paths)
    p.add_argument("--ddpm_model_path", type=str, default=None)
    p.add_argument("--ldm_diffusion_path", type=str, default=None)
    p.add_argument("--sd_model_id", type=str, default=None)
    p.add_argument("--sd_inner_resolution", type=int, default=None)
    p.add_argument("--sd_target_resolution", type=int, default=None)
    p.add_argument("--sd_guidance_scale", type=float, default=None)
    p.add_argument("--sd_prompt", type=str, default=None)
    p.add_argument("--sd_hf_home", type=str, default=None)

    return p.parse_args()


def _apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Optional[Any]]) -> Dict[str, Any]:
    cfg = dict(cfg)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def build_configs(args: argparse.Namespace) -> Dict[str, Any]:
    data_cfg = dict(DATA_PRESETS[args.data])
    data_cfg = _apply_overrides(data_cfg, {
        "root": args.data_root,
        "resolution": args.data_resolution,
        "start_id": args.data_start_id,
        "end_id": args.data_end_id,
    })

    operator_cfg = dict(TASK_PRESETS[args.task]["operator"])

    if args.operator_sigma is not None:
        operator_cfg["sigma"] = args.operator_sigma
    if args.phase_oversample is not None and operator_cfg.get("name") == "phase_retrieval":
        operator_cfg["oversample"] = args.phase_oversample
    if args.down_scale_factor is not None and operator_cfg.get("name") == "down_sampling":
        operator_cfg["scale_factor"] = args.down_scale_factor
    if args.inpaint_mask_len is not None and operator_cfg.get("name") == "inpainting":
        # upstream inpainting uses mask_len_range in YAML; for box inpainting we use a fixed len
        operator_cfg["mask_len_range"] = [args.inpaint_mask_len, args.inpaint_mask_len]
    if args.random_inpaint_prob is not None and operator_cfg.get("name") == "inpainting":
        operator_cfg["mask_prob_range"] = [args.random_inpaint_prob, args.random_inpaint_prob]
    if args.gaussian_kernel_size is not None and operator_cfg.get("name") == "gaussian_blur":
        operator_cfg["kernel_size"] = args.gaussian_kernel_size
    if args.gaussian_intensity is not None and operator_cfg.get("name") == "gaussian_blur":
        operator_cfg["intensity"] = args.gaussian_intensity
    if args.motion_kernel_size is not None and operator_cfg.get("name") == "motion_blur":
        operator_cfg["kernel_size"] = args.motion_kernel_size
    if args.motion_intensity is not None and operator_cfg.get("name") == "motion_blur":
        operator_cfg["intensity"] = args.motion_intensity
    if args.hdr_scale is not None and operator_cfg.get("name") == "high_dynamic_range":
        operator_cfg["scale"] = args.hdr_scale
    if args.bkse_opt_yml_path is not None and operator_cfg.get("name") == "nonlinear_blur":
        operator_cfg["opt_yml_path"] = args.bkse_opt_yml_path

    sampler_cfg = dict(SAMPLER_PRESETS[args.sampler])
    if args.T is not None:
        sampler_cfg["diffusion_scheduler_config"] = dict(sampler_cfg["diffusion_scheduler_config"])
        sampler_cfg["diffusion_scheduler_config"]["num_steps"] = args.T
    if args.diffusion_sigma_max is not None:
        sampler_cfg["diffusion_scheduler_config"] = dict(sampler_cfg["diffusion_scheduler_config"])
        sampler_cfg["diffusion_scheduler_config"]["sigma_max"] = args.diffusion_sigma_max
    if args.diffusion_sigma_min is not None:
        sampler_cfg["diffusion_scheduler_config"] = dict(sampler_cfg["diffusion_scheduler_config"])
        sampler_cfg["diffusion_scheduler_config"]["sigma_min"] = args.diffusion_sigma_min
    if args.backtrack is not None:
        sampler_cfg["backtrack"] = args.backtrack
    if args.alpha_max is not None:
        sampler_cfg["alpha_max"] = args.alpha_max
    if args.rho is not None:
        sampler_cfg["rho"] = args.rho
    if args.K is not None:
        sampler_cfg["K"] = args.K
    if args.S is not None:
        sampler_cfg["S"] = args.S
    if args.epsilon is not None:
        sampler_cfg["epsilon"] = args.epsilon
    if args.sigma_switch is not None:
        sampler_cfg["sigma_switch"] = args.sigma_switch
    if args.rho_x is not None:
        sampler_cfg["rho_x"] = args.rho_x
    if args.K_x is not None:
        sampler_cfg["K_x"] = args.K_x
    if args.S_x is not None:
        sampler_cfg["S_x"] = args.S_x
    if args.rho_z is not None:
        sampler_cfg["rho_z"] = args.rho_z
    if args.K_z is not None:
        sampler_cfg["K_z"] = args.K_z
    if args.S_z is not None:
        sampler_cfg["S_z"] = args.S_z

    preset_model_cfg = dict(MODEL_PRESETS[args.model])
    model_cfg = dict(preset_model_cfg)

    if model_cfg["name"] == "ddpm" and args.ddpm_model_path is not None:
        model_cfg["model_config"] = dict(model_cfg["model_config"])
        model_cfg["model_config"]["model_path"] = args.ddpm_model_path
    if model_cfg["name"] == "ldm" and args.ldm_diffusion_path is not None:
        model_cfg["diffusion_path"] = args.ldm_diffusion_path
    if model_cfg["name"] == "sdm":
        if args.sd_model_id is not None:
            model_cfg["model_id"] = args.sd_model_id
        if args.sd_inner_resolution is not None:
            model_cfg["inner_resolution"] = args.sd_inner_resolution
        if args.sd_target_resolution is not None:
            model_cfg["target_resolution"] = args.sd_target_resolution
        if args.sd_guidance_scale is not None:
            model_cfg["guidance_scale"] = args.sd_guidance_scale
        if args.sd_prompt is not None:
            model_cfg["prompt"] = args.sd_prompt
        if args.sd_hf_home is not None:
            model_cfg["hf_home"] = args.sd_hf_home

    return {
        "data_cfg": data_cfg,
        "operator_cfg": operator_cfg,
        "sampler_cfg": sampler_cfg,
        "model_cfg": model_cfg,
    }


def sample_in_batch(sampler, model, x_start, operator, y, evaluator, args, root, run_id, gt):
    samples = []
    trajs = []
    sample_times = []
    B = x_start.shape[0]
    for s in range(0, B, args.batch_size):
        cur_x_start = x_start[s:s + args.batch_size]
        cur_y = y[s:s + args.batch_size]
        cur_gt = gt[s:s + args.batch_size]
        cur_samples = sampler.sample(model, cur_x_start, operator, cur_y, evaluator, verbose=True, record=args.save_traj, gt=cur_gt)
        sample_times.append(float(getattr(sampler, "last_sampling_time", 0.0)))
        samples.append(cur_samples)
        if args.save_traj:
            cur_trajs = sampler.trajectory.compile()
            trajs.append(cur_trajs)

        # save individual samples
        if args.save_samples:
            pil_image_list = tensor_to_pils(cur_samples)
            image_dir = safe_dir(root / "samples")
            for idx in range(len(pil_image_list)):
                image_path = image_dir / "{:05d}_run{:04d}.png".format(idx + s, run_id)
                pil_image_list[idx].save(str(image_path))

        # save trajectory grids + optional mp4
        if args.save_traj:
            traj_dir = safe_dir(root / "trajectory")
            x0hat_traj = cur_trajs.tensor_data["x0hat"]
            x0y_traj = cur_trajs.tensor_data["x0y"]
            xt_traj = cur_trajs.tensor_data["xt"]
            cur_resized_y = resize(cur_y, cur_samples, operator.name)
            slices = np.linspace(0, len(x0hat_traj) - 1, 10).astype(int)
            slices = np.unique(slices)
            for idx in range(cur_samples.shape[0]):
                if args.save_traj_video:
                    video_path = str(traj_dir / "{:05d}_run{:04d}.mp4".format(idx + s, run_id))
                    save_mp4_video(cur_samples[idx], cur_resized_y[idx], x0hat_traj[:, idx], x0y_traj[:, idx], xt_traj[:, idx], video_path)
                selected_traj_grid = torch.cat([x0y_traj[slices, idx], x0hat_traj[slices, idx], xt_traj[slices, idx]], dim=0)
                traj_grid_path = str(traj_dir / "{:05d}_run{:04d}.png".format(idx + s, run_id))
                save_image(selected_traj_grid * 0.5 + 0.5, fp=traj_grid_path, nrow=len(slices))

    if args.save_traj:
        trajs = Trajectory.merge(trajs)
    if sample_times:
        avg_time_per_sample = sum(sample_times) / len(sample_times)
        total_sampling_time = sum(sample_times)
    else:
        avg_time_per_sample = 0.0
        total_sampling_time = 0.0
    print(f"Pure sampling time per sample: {avg_time_per_sample:.3f} s")
    return torch.cat(samples, dim=0), trajs, avg_time_per_sample


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(f"cuda:{args.gpu}")

    setproctitle.setproctitle(args.name)

    cfgs = build_configs(args)
    data_cfg = cfgs["data_cfg"]
    operator_cfg = cfgs["operator_cfg"]
    sampler_cfg = cfgs["sampler_cfg"]
    model_cfg = cfgs["model_cfg"]

    dataset = get_dataset(**data_cfg, device=f"cuda:{args.gpu}")
    total_number = len(dataset)
    images = dataset.get_data(total_number, 0)

    operator = get_operator(**operator_cfg)
    y = operator.measure(images)

    sampler = get_sampler(**sampler_cfg)

    model_kwargs = dict(model_cfg)
    name = model_kwargs.pop("name")
    model = get_model(name=name, **model_kwargs, device=f"cuda:{args.gpu}")
    eval_fn_list = [get_eval_fn(n.strip()) for n in args.eval_fn_list.split(",") if n.strip()]
    evaluator = Evaluator(eval_fn_list)

    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = safe_dir(Path(args.save_dir))
    root = safe_dir(save_dir / args.name)

    full_config_dump = {
        "args": vars(args),
        "data": data_cfg,
        "operator": operator_cfg,
        "sampler": sampler_cfg,
        "model": model_cfg,
    }
    with open(str(root / "config.yaml"), "w") as f:
        yaml.safe_dump(full_config_dump, f, default_flow_style=False, allow_unicode=True)

    if args.wandb:
        wandb.init(project=args.project_name, name=args.name, config=full_config_dump)

    full_samples = []
    full_trajs = []
    run_avg_times = []
    for r in range(args.num_runs):
        print(f"Run: {r}")
        x_start = sampler.get_start(images.shape[0], model)
        samples, trajs, avg_time = sample_in_batch(sampler, model, x_start, operator, y, evaluator, args, root, r, images)
        full_samples.append(samples)
        full_trajs.append(trajs)
        run_avg_times.append(avg_time)
    full_samples = torch.stack(full_samples, dim=0)  # [num_runs, B, C, H, W]

    results = evaluator.report(images, y, full_samples)
    if args.wandb:
        evaluator.log_wandb(results, args.batch_size)
    markdown_text = evaluator.display(results)
    with open(str(root / "eval.md"), "w") as f:
        f.write(markdown_text)
    json.dump(results, open(str(root / "metrics.json"), "w"), indent=4)
    print(markdown_text)

    resized_y = resize(y, images, operator.name)
    stack = torch.cat([images, resized_y, full_samples.flatten(0, 1)])
    save_image(stack * 0.5 + 0.5, fp=str(root / "grid_results.png"), nrow=total_number)

    if args.save_traj_raw_data:
        traj_dir = safe_dir(root / "trajectory")
        traj_raw_data = safe_dir(traj_dir / "raw")
        for run, sde_traj in enumerate(full_trajs):
            print(f"saving trajectory run {run}...")
            torch.save(sde_traj, str(traj_raw_data / "trajectory_run{:04d}.pth".format(run)))

    if args.eval_fid:
        print("Calculating FID...")
        fid_dir = safe_dir(root / "fid")

        eval_fn_cmp = get_eval_fn_cmp(evaluator.main_eval_fn_name)
        eval_values = np.array(results[evaluator.main_eval_fn_name]["sample"])  # [B, num_runs]
        if eval_fn_cmp == "min":
            best_idx = np.argmin(eval_values, axis=1)
        elif eval_fn_cmp == "max":
            best_idx = np.argmax(eval_values, axis=1)
        else:
            raise ValueError(f"Unknown cmp {eval_fn_cmp}")

        best_samples = full_samples[best_idx, np.arange(full_samples.shape[1])]
        best_sample_dir = safe_dir(fid_dir / "best_sample")
        pil_image_list = tensor_to_pils(best_samples)
        for idx in range(len(pil_image_list)):
            image_path = best_sample_dir / "{:05d}.png".format(idx)
            pil_image_list[idx].save(str(image_path))

        fake_dataset = get_dataset(data_cfg["name"], resolution=data_cfg["resolution"], root=str(best_sample_dir), device=f"cuda:{args.gpu}")
        real_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        fake_loader = DataLoader(fake_dataset, batch_size=100, shuffle=False)
        fid_score = calculate_fid(real_loader, fake_loader)
        print(f"FID Score: {fid_score.item():.4f}")
        with open(str(fid_dir / "fid.txt"), "w") as f:
            f.write(f"FID Score: {fid_score.item():.4f}")
        if args.wandb:
            wandb.log({"FID": fid_score.item()})

    if run_avg_times:
        overall_avg_time = sum(run_avg_times) / len(run_avg_times)
        print("\n=== Overall Statistics ===")
        print(
            f"Average time per sample across {args.num_runs} runs: {overall_avg_time:.3f} s"
        )

    print(f"finish {args.name}!")


if __name__ == "__main__":
    main()
