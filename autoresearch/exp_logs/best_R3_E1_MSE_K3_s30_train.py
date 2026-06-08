"""
PGDiff autoresearch — agent-editable experiment file.

This is the ONLY file you modify. Everything is fair game:
  - Hyperparameters (top section)
  - PartialGuidance class (loss terms, gradient strategy, composite tasks)
  - model_fn (how the diffusion model is called)
  - The sampling pipeline and experiment runner
  - The recording/logging system

Each experiment:
  1. You edit this file with your idea
  2. git commit with a descriptive message
  3. python train.py > run.log 2>&1
  4. Extract results: grep "^quality_score_v2:" run.log
  5. Log to results.tsv
  6. Keep or discard based on quality_score_v2 (higher = better)

PRIMARY METRIC (V2):  quality_score_v2 = ssim × naturalness × clip(sharpness_gain, 0.2, 5.0)
  - ssim_input:    structural similarity to input LQ image [0, 1] — structure preservation
  - naturalness:   composite [0, 1] based on edge density + local variance + gradient stats
  - sharpness_gain: output sharpness / input sharpness (capped 0.2–5.0)

Key advantages over old metric (avg_loss/sharpness):
  - STEP-INDEPENDENT: works across 50, 200, 1000 step experiments
  - IMAGE-BASED: evaluates output image quality, not optimization internals
  - ANTI-GAMING: SSIM penalizes color blobs/noise; sharpness cap prevents noise hijack
  - MULTI-DIMENSIONAL: no single metric can be isolated and gamed

BACKWARD COMPAT: quality_score_old (avg_loss/sharpness) still printed for reference.
"""
import os
import sys
import time
import json
import shutil
import math
import cv2
import numpy as np
import scipy.stats
import scipy.special
import torch as th
import torch.nn.functional as F
from collections import OrderedDict

_srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _srcdir not in sys.path:
    sys.path.insert(0, _srcdir)

from guided_diffusion import dist_util
from guided_diffusion.script_util import model_and_diffusion_defaults

from prepare import (
    setup_device, device,
    load_diffusion_model, load_restorer, load_arcface,
    load_image, load_mask, save_image,
    avg_grayscale, adaptive_instance_normalization, calc_mean_std,
    compute_sharpness, compute_ssim, compute_image_metrics, aggregate_metrics,
    get_supported_tasks, get_model_defaults,
)

# ═══════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS  —  Edit freely
# ═══════════════════════════════════════════════════════════════════════════

TASK = "restoration"
GUIDANCE_SCALE = 0.30
SEED = 1234

# --- task weights ---
LIGHTNESS_WEIGHT = 1.0
COLOR_WEIGHT = 0.05
UNMASKED_WEIGHT = 1.0
SS_WEIGHT = 1.0
EDGE_WEIGHT = 0.0         # 0 = ablation: remove edge preservation
GRAD_MOMENTUM = 0.0      # 0 = no momentum (improves quality at full 1000 steps)
REF_WEIGHT = 25.0
OP_LIGHTNESS_WEIGHT = 1.0
OP_COLOR_WEIGHT = 0.05
RESIDUAL_BLEND = 0.5        # output = (sample + input * blend) / (1 + blend) — weighted average
USE_HUBER_LOSS = False      # True = SmoothL1 loss (current), False = MSE loss (paper original)

# --- multi-step guidance ---
N = 1                # gradient steps per timestep (>1 = stronger guidance)
S_START = 1.0        # start fraction of T (e.g. 1.0 = from t=T)
S_END = 1.0          # end fraction of T (e.g. 0.7 = until 0.7T)

# --- sampling ---
TIMESTEP_RESPACING = ""   # full 1000 steps
USE_DDIM = False
CLIP_DENOISED = True
BATCH_SIZE = 1
IMAGE_SIZE = 512
DIFFUSION_STEPS = 1000    # total diffusion steps of the pre-trained model

# --- I/O paths ---
IN_DIR = "../testdata/pairs/LQ"
GT_DIR = "../testdata/pairs/GT"   # GT images for reference-based metrics (PSNR, SSIM_gt)
# Use only first N images for faster experiments
MAX_IMAGES = 1   # 0 = use all images in IN_DIR
# Speed optimization flags
BLOCK_UNET_GRAD = True   # True: restorer outside enable_grad
USE_DPMSOLVER = False     # True: use DPM-Solver-2 (higher-order ODE, fewer steps)
DPM_SOLVER_STEPS = 35     # number of DPM-Solver steps (if USE_DPMSOLVER=True)
RESTORER_T_ZERO = False   # True: call restorer with t=0 (test t-conditioning)
DPM_FIRST_ORDER = False   # True: skip 2nd-order correction in DPM-Solver
CONSTANT_SCHEDULE = False # True: disable linear schedule, use schedule=1.0
HYBRID_MODE = False       # True: DPM coarse + DDPM refine
HYBRID_SWITCH_T = 400     # timestep to switch from DPM to DDPM
REFINE_STEPS = 50         # DDPM steps in refinement phase (if HYBRID_MODE)
GUIDANCE_EVERY_K = 3      # run guidance every K steps (1=every step, 5=sparse)
SKIP_ZERO_SCALE = True    # True: skip guidance when scale=0 (saves restorer+grad)
GUIDANCE_EARLY_STOP = False  # True: stop guidance after t < s_end
USE_FP16 = False          # True: run UNet in float16
OUT_DIR = "../results/exp_R3_E1_MSE_K3_s30"
REF_DIR = None
MASK_DIR = None
MODEL_PATH = "../models/iddpm_ffhq512_ema500000.pth"
RESTORER_PATH = "../models/restorer/rrdb_iter_100000.pth"
ARCFACE_PATH = "../models/ms1mv3_arcface_r50_fp16.pth"

# --- recording ---
RUN_TAG = ""  # 推理加速实验第1轮
RUNS_DIR = "../runs"     # per-experiment detailed logs stored here

# ═══════════════════════════════════════════════════════════════════════════
# Helper: Sobel edge maps  —  used by edge-preservation loss
# ═══════════════════════════════════════════════════════════════════════════

def sobel_edges(img):
    """Compute Sobel gradient magnitude for each channel.
    img: (1, 3, H, W) in [-1, 1]. Returns (1, 3, H, W) edge magnitude maps."""
    kernel_x = th.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=th.float32, device=img.device).view(1, 1, 3, 3)
    kernel_y = th.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=th.float32, device=img.device).view(1, 1, 3, 3)
    img_gray = img.mean(dim=1, keepdim=True)  # (1, 1, H, W)
    grad_x = F.conv2d(F.pad(img_gray, (1, 1, 1, 1), mode='replicate'), kernel_x)
    grad_y = F.conv2d(F.pad(img_gray, (1, 1, 1, 1), mode='replicate'), kernel_y)
    mag = th.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    return mag

# ═══════════════════════════════════════════════════════════════════════════
# PARTIAL GUIDANCE  —  Edit freely: add loss terms, modify strategies
# ═══════════════════════════════════════════════════════════════════════════

class PartialGuidance:
    """
    Callable guidance function. Computes task-specific loss gradients and
    accumulates loss for the evaluation metric.

    Interface expected by GaussianDiffusion.condition_mean:
        cond_fn(x, t, y=y, pred_xstart=pred_xstart, ...) -> (gradient, target)

    MODIFY FREELY:
      - Add new loss terms to __call__
      - Add new tasks to the task routing if/elif chain
      - Change loss formulations (e.g. L1 instead of MSE, perceptual loss, etc.)
      - Modify gradient scaling or masking strategies
      - Add composite task definitions
    """

    def __init__(self, restorer, embedding, **weights):
        self.restorer = restorer
        self.embedding = embedding
        self.w = weights
        self.losses = []           # accumulated per-timestep guidance loss
        self.loss_breakdown = {}   # per-loss-term tracking (for analysis)
        self.prev_gradient = None  # for gradient momentum
        self.guidance_calls = 0    # speed tracking
        self.restorer_calls = 0

    def reset(self):
        self.losses = []
        self.loss_breakdown = {}
        self.prev_gradient = None
        self.guidance_calls = 0
        self.restorer_calls = 0

    def avg_loss(self):
        if not self.losses:
            return float("inf")
        return sum(self.losses) / len(self.losses)

    def __call__(self, x, t, y=None, pred_xstart=None, target=None,
                 ref=None, mask=None, task="restoration", scale=0,
                 N=1, s_start=1, s_end=0.7):
        assert y is not None
        self.guidance_calls += 1

        # ── Skip logic: scale=0 or sparse guidance ──
        if SKIP_ZERO_SCALE and scale == 0:
            self.losses.append(0.0)
            return th.zeros_like(pred_xstart if pred_xstart is not None else x), None
        if GUIDANCE_EVERY_K > 1 and (self.guidance_calls % GUIDANCE_EVERY_K) != 0:
            self.losses.append(0.0)
            return th.zeros_like(pred_xstart if pred_xstart is not None else x), None
        if GUIDANCE_EARLY_STOP:
            if t[0].item() < int(s_end):
                self.losses.append(0.0)
                return th.zeros_like(pred_xstart if pred_xstart is not None else x), None

        fake_g_output = None

        # ── restorer forward (position depends on BLOCK_UNET_GRAD) ──
        if "restoration" in task:
            if target is not None:
                fake_g_output = target.cuda()
            elif BLOCK_UNET_GRAD:
                # Outside enable_grad: no computation graph built → saves VRAM
                self.restorer_calls += 1
                with th.no_grad():
                    t_restorer = th.zeros_like(t) if RESTORER_T_ZERO else t
                    fake_g_output = self.restorer(x, y_t=y, t=t_restorer).clamp(-1, 1).cuda()

        with th.enable_grad():
            pred_xstart_in = pred_xstart.detach().requires_grad_(True) if pred_xstart is not None else x.detach().requires_grad_(True)
            total_loss = th.tensor(0.0, device=x.device)

            # ── restorer for baseline (inside enable_grad = builds graph, more VRAM) ──
            if "restoration" in task and not BLOCK_UNET_GRAD and fake_g_output is None:
                self.restorer_calls += 1
                fake_g_output = self.restorer(x, y_t=y, t=t).clamp(-1, 1).cuda()
            pred_xstart_in = pred_xstart.detach().requires_grad_(True) if pred_xstart is not None else x.detach().requires_grad_(True)
            total_loss = th.tensor(0.0, device=x.device)

            # ── colorization ──
            if task == "colorization":
                loss_l = F.mse_loss(
                    avg_grayscale(y), avg_grayscale(pred_xstart_in), reduction="sum"
                ) * self.w.get("lightness_weight", 1.0)
                self._track("color_lightness", loss_l.item())
                total_loss = total_loss + loss_l

                pred_adain = adaptive_instance_normalization(pred_xstart_in, None).clamp(-1, 1)
                loss_c = F.mse_loss(
                    pred_xstart_in, pred_adain, reduction="sum"
                ) * self.w.get("color_weight", 0.05)
                self._track("color_stats", loss_c.item())
                total_loss = total_loss + loss_c

            # ── inpainting ──
            if task == "inpainting":
                loss_u = F.mse_loss(
                    y[mask == 0], pred_xstart_in[mask == 0], reduction="sum"
                ) * self.w.get("unmasked_weight", 1.0)
                self._track("inpaint_unmasked", loss_u.item())
                total_loss = total_loss + loss_u

            # ── restoration (smooth semantics) ──
            if "restoration" in task:
                if USE_HUBER_LOSS:
                    loss_s = F.smooth_l1_loss(
                        fake_g_output, pred_xstart_in, reduction="sum", beta=1.0
                    ) * self.w.get("ss_weight", 1.0)
                else:
                    loss_s = F.mse_loss(
                        fake_g_output, pred_xstart_in, reduction="sum"
                    ) * self.w.get("ss_weight", 1.0)
                self._track("smooth_semantics", loss_s.item())
                total_loss = total_loss + loss_s

                edge_target = sobel_edges(fake_g_output)
                edge_pred = sobel_edges(pred_xstart_in)
                loss_e = F.l1_loss(
                    edge_pred, edge_target, reduction="sum"
                ) * self.w.get("edge_weight", 0.02)
                self._track("edge_preservation", loss_e.item())
                total_loss = total_loss + loss_e

            # ── ref_restoration (identity) ──
            if task == "ref_restoration":
                emd_x0 = self.embedding(
                    F.interpolate(pred_xstart_in, (112, 112), mode="bilinear", antialias=True)
                )
                emd_ref = self.embedding(
                    F.interpolate(ref, (112, 112), mode="bilinear", antialias=True)
                )
                loss_r = F.mse_loss(
                    emd_x0, emd_ref, reduction="sum"
                ) * self.w.get("ref_weight", 25.0)
                self._track("identity_ref", loss_r.item())
                total_loss = total_loss + loss_r

            # ── old_photo_restoration (composite) ──
            if task == "old_photo_restoration":
                total_loss = th.tensor(0.0, device=x.device)
                pred_xstart_in = pred_xstart.detach().requires_grad_(True) if pred_xstart is not None else x.detach().requires_grad_(True)
                fake_g_output = fake_g_output.detach().requires_grad_(True)

                loss_opl = F.mse_loss(
                    avg_grayscale(fake_g_output)[mask == 0],
                    avg_grayscale(pred_xstart_in)[mask == 0],
                    reduction="sum",
                ) * self.w.get("op_lightness_weight", 1.0)
                self._track("op_lightness", loss_opl.item())
                total_loss = total_loss + loss_opl

                pred_adain = adaptive_instance_normalization(pred_xstart_in, None).clamp(-1, 1)
                loss_opc = F.mse_loss(
                    pred_xstart_in, pred_adain, reduction="sum"
                ) * self.w.get("op_color_weight", 0.5)
                self._track("op_color", loss_opc.item())
                total_loss = total_loss + loss_opc

            # ── add new tasks / loss terms here ──

            # Linear decay schedule: amplifies variance trend (strong early, weak late)
            t_frac = t[0].item() / DIFFUSION_STEPS
            schedule = 1.0 if CONSTANT_SCHEDULE else 0.3 + 0.7 * t_frac
            total_loss = total_loss * schedule

            gradient = th.autograd.grad(total_loss, pred_xstart_in)[0]
            if self.prev_gradient is not None:
                momentum = self.w.get("grad_momentum", 0.9)
                gradient = momentum * self.prev_gradient + (1 - momentum) * gradient
            self.prev_gradient = gradient.detach()
            if task in ("inpainting", "old_photo_restoration"):
                gradient[mask > 0] = 0

        self.losses.append(total_loss.item())

        if "restoration" in task:
            return gradient, fake_g_output
        else:
            return gradient, None

    def _track(self, name, value):
        if name not in self.loss_breakdown:
            self.loss_breakdown[name] = []
        self.loss_breakdown[name].append(value)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL FORWARD  —  Edit freely: change conditioning, add features
# ═══════════════════════════════════════════════════════════════════════════

def model_fn(x, t, y=None, target=None, ref=None, mask=None,
             task=None, scale=0, N=1, s_start=1, s_end=0.7):
    """
    Wrapper that filters kwargs before passing to the diffusion model.
    The UNet only needs (x, timesteps, optional_y).
    """
    assert y is not None
    return model(x, t, y if get_model_defaults()["class_cond"] else None)


# ═══════════════════════════════════════════════════════════════════════════
# DPM-SOLVER-2 SAMPLER  —  Higher-order ODE solver (fewer steps, same quality)
# ═══════════════════════════════════════════════════════════════════════════

def dpm_solver_sample_loop(model_fn, shape, alphas_cumprod, model, cond_fn,
                            model_kwargs, clip_denoised=True, device=None,
                            seed=1234, num_steps=50):
    """
    DPM-Solver-2 (multi-step) sampling with guidance.
    Uses 2 NFEs per step for 2nd-order accuracy => far fewer steps needed.
    """
    import math as _math
    th.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    T = len(alphas_cumprod)
    alpha_bar = th.from_numpy(alphas_cumprod.astype(np.float64)).float().to(device)
    alpha = alpha_bar.sqrt()          # sqrt(alpha_bar), shape (T,)
    sigma = (1 - alpha_bar).sqrt()    # sqrt(1 - alpha_bar)

    # log-SNR: lambda = log(alpha / sigma)
    lam = th.log(alpha / (sigma + 1e-8))

    # Uniformly spaced timesteps (in lambda space)
    lam_targets = th.linspace(lam[0].item(), lam[-1].item(), num_steps, device=device)
    step_t = []
    for lv in lam_targets:
        idx = th.argmin((lam - lv).abs()).item()
        step_t.append(idx)
    step_t = sorted(set(step_t), reverse=True)
    if step_t[-1] != 0:
        if step_t[-1] > 0:
            step_t.append(0)
        else:
            step_t[-1] = 0

    # Remove duplicates and ensure monotonicity
    step_t = sorted(set(step_t), reverse=True)

    # Helper: scalar → broadcastable (B, C, H, W) view
    def _bc(t_val):
        return alpha[t_val].view(1, 1, 1, 1).expand(shape)

    def _bs(t_val):
        return sigma[t_val].view(1, 1, 1, 1).expand(shape)

    # Initial noise
    x = th.randn(*shape, device=device)

    for i in range(len(step_t) - 1):
        t_cur = step_t[i]
        t_next = step_t[i + 1]

        if t_cur == t_next:
            continue

        t_tensor = th.tensor([t_cur] * shape[0], device=device)
        a_cur, s_cur = _bc(t_cur), _bs(t_cur)
        a_nxt, s_nxt = _bc(t_next), _bs(t_next)

        # ── 1) Model forward (model outputs 6ch: first 3 = x0_pred) ──
        with th.no_grad():
            model_out = model_fn(x, t_tensor, **model_kwargs)
            x0_pred = model_out[:, :3]  # first 3 channels are x₀ prediction
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1, 1)

        # ── 2) Epsilon + guidance ──
        eps_pred = (x - a_cur * x0_pred) / (s_cur + 1e-8)

        with th.enable_grad():
            pxi = x0_pred.detach().requires_grad_(True)
            model_kwargs['pred_xstart'] = pxi
            grad, _ = cond_fn(x, t_tensor, **model_kwargs)
            model_kwargs.pop('pred_xstart', None)
        if grad is not None:
            gs = model_kwargs.get("scale", 0.1)
            eps_pred = eps_pred - s_cur * grad * gs

        # ── 3) First-order step to midpoint ──
        if i < len(step_t) - 2 and not DPM_FIRST_ORDER:
            t_mid = int((t_cur * t_next) ** 0.5) if t_cur > 0 else 0
            if t_mid == t_cur or t_mid == t_next:
                t_mid = (t_cur + t_next) // 2
            a_mid, s_mid = _bc(t_mid), _bs(t_mid)
            x_mid = a_mid * x0_pred + s_mid * eps_pred
            t_mid_t = th.tensor([t_mid] * shape[0], device=device)

            # ── 4) Second model eval at midpoint ──
            with th.no_grad():
                model_out_mid = model_fn(x_mid, t_mid_t, **model_kwargs)
                x0_mid = model_out_mid[:, :3]
            if clip_denoised:
                x0_mid = x0_mid.clamp(-1, 1)
            eps_mid = (x_mid - a_mid * x0_mid) / (s_mid + 1e-8)

            with th.enable_grad():
                pxi_mid = x0_mid.detach().requires_grad_(True)
                model_kwargs['pred_xstart'] = pxi_mid
                g_mid, _ = cond_fn(x_mid, t_mid_t, **model_kwargs)
                model_kwargs.pop('pred_xstart', None)
            if g_mid is not None:
                eps_mid = eps_mid - s_mid * g_mid * gs

            # ── 5) Second-order correction ──
            h = (lam[t_cur] - lam[t_next]).item()
            h1 = (lam[t_cur] - lam[t_mid]).item()
            r = h1 / max(h, 1e-8) if h != 0 else 1.0
            eps_corr = (1 + 1 / (2 * max(r, 0.01))) * eps_mid - (1 / (2 * max(r, 0.01))) * eps_pred

            x_out = a_nxt * x0_mid + s_nxt * eps_corr
            if th.isnan(x_out).any():
                # Fallback: use first order
                x_out = a_nxt * x0_pred + s_nxt * eps_pred
            x = x_out
        else:
            # ── 6) Last step: first order ──
            x = a_nxt * x0_pred + s_nxt * eps_pred

    return x


# ═══════════════════════════════════════════════════════════════════════════
# HYBRID SAMPLER — DPM coarse → DDPM refine
# ═══════════════════════════════════════════════════════════════════════════

def hybrid_sample_loop(model_fn, shape, alphas_cumprod, model, cond_fn,
                       model_kwargs, clip_denoised=True, device=None,
                       seed=1234, coarse_dpm_steps=15, switch_t=200,
                       refine_steps=50):
    import math as _math
    from guided_diffusion.respace import SpacedDiffusion
    from guided_diffusion import gaussian_diffusion as gd

    th.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    T = len(alphas_cumprod)
    alpha_bar = th.from_numpy(alphas_cumprod.astype(np.float64)).float().to(device)
    alpha = alpha_bar.sqrt()
    sigma = (1 - alpha_bar).sqrt()
    lam = th.log(alpha / (sigma + 1e-8))

    # ── Phase 1: DPM-Solver from T-1 to switch_t ──
    lam_switch = lam[switch_t]
    lam_targets = th.linspace(lam[0], lam_switch, coarse_dpm_steps, device=device)
    step_t = sorted(set(int(th.argmin((lam - lv).abs()).item()) for lv in lam_targets), reverse=True)
    if len(step_t) < 2:
        step_t = [T-1, switch_t]

    def _bc(tv):
        return alpha[tv].view(1, 1, 1, 1).expand(shape)
    def _bs(tv):
        return sigma[tv].view(1, 1, 1, 1).expand(shape)

    x = th.randn(*shape, device=device)

    for i in range(len(step_t) - 1):
        t_cur = step_t[i]
        t_next = step_t[i + 1]
        if t_cur == t_next:
            continue
        t_tensor = th.tensor([t_cur] * shape[0], device=device)
        a_cur, s_cur = _bc(t_cur), _bs(t_cur)
        a_nxt, s_nxt = _bc(t_next), _bs(t_next)

        with th.no_grad():
            model_out = model_fn(x, t_tensor, **model_kwargs)
            x0_pred = model_out[:, :3]
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1, 1)

        eps_pred = (x - a_cur * x0_pred) / (s_cur + 1e-8)

        with th.enable_grad():
            pxi = x0_pred.detach().requires_grad_(True)
            model_kwargs['pred_xstart'] = pxi
            grad, _ = cond_fn(x, t_tensor, **model_kwargs)
            model_kwargs.pop('pred_xstart', None)
        if grad is not None:
            gs = model_kwargs.get("scale", 0.1)
            eps_pred = eps_pred - s_cur * grad * gs

        if i < len(step_t) - 2:
            t_mid = int((t_cur * t_next) ** 0.5) if t_cur > 0 else 0
            if t_mid == t_cur or t_mid == t_next:
                t_mid = (t_cur + t_next) // 2
            a_mid, s_mid = _bc(t_mid), _bs(t_mid)
            x_mid = a_mid * x0_pred + s_mid * eps_pred
            t_mid_t = th.tensor([t_mid] * shape[0], device=device)
            with th.no_grad():
                model_out_mid = model_fn(x_mid, t_mid_t, **model_kwargs)
                x0_mid = model_out_mid[:, :3]
            if clip_denoised:
                x0_mid = x0_mid.clamp(-1, 1)
            eps_mid = (x_mid - a_mid * x0_mid) / (s_mid + 1e-8)
            with th.enable_grad():
                pxi_mid = x0_mid.detach().requires_grad_(True)
                model_kwargs['pred_xstart'] = pxi_mid
                g_mid, _ = cond_fn(x_mid, t_mid_t, **model_kwargs)
                model_kwargs.pop('pred_xstart', None)
            if g_mid is not None:
                eps_mid = eps_mid - s_mid * g_mid * gs
            h = (lam[t_cur] - lam[t_next]).item()
            h1 = (lam[t_cur] - lam[t_mid]).item()
            r = h1 / max(h, 1e-8) if h != 0 else 1.0
            eps_corr = (1 + 1 / (2 * max(r, 0.01))) * eps_mid - (1 / (2 * max(r, 0.01))) * eps_pred
            x_out = a_nxt * x0_mid + s_nxt * eps_corr
            if th.isnan(x_out).any():
                x_out = a_nxt * x0_pred + s_nxt * eps_pred
            x = x_out
        else:
            x = a_nxt * x0_pred + s_nxt * eps_pred

    # ── Phase 2: DDPM refinement from switch_t to 0 ──
    betas = gd.get_named_beta_schedule("linear", DIFFUSION_STEPS)
    use_ts = set()
    stride = max(1, switch_t // (refine_steps - 1)) if refine_steps > 1 else switch_t
    for k in range(refine_steps):
        use_ts.add(max(0, switch_t - k * stride))
    use_ts.add(0)

    refine_diff = SpacedDiffusion(
        use_timesteps=use_ts, betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.LEARNED_RANGE,
        loss_type=gd.LossType.MSE, rescale_timesteps=False,
    )

    x = refine_diff.p_sample_loop(
        model_fn, shape, clip_denoised=clip_denoised,
        model_kwargs=model_kwargs, cond_fn=cond_fn,
        device=device, seed=seed, noise=x,
    )

    return x


# ═══════════════════════════════════════════════════════════════════════════
# SAMPLING / EXPERIMENT RUNNER  —  Edit freely: change loop, add metrics
# ═══════════════════════════════════════════════════════════════════════════

def build_model_kwargs(img_name, in_dir, task, guidance_scale, n, s_start, s_end,
                       diffusion_steps, ref_dir=None, mask_dir=None, mask_images=None):
    """Build the model_kwargs dict for one input image."""
    kwargs = {
        "task": task,
        "target": None,
        "scale": guidance_scale,
        "N": n,
        "s_start": int(s_start * diffusion_steps),
        "s_end": int(s_end * diffusion_steps),
    }
    kwargs["y"] = load_image(os.path.join(in_dir, img_name)).to(device())

    if task == "ref_restoration" and ref_dir:
        kwargs["ref"] = load_image(os.path.join(ref_dir, img_name)).to(device())

    if task in ("inpainting", "old_photo_restoration"):
        if mask_images is not None and img_name in mask_images:
            kwargs["mask"] = load_mask(os.path.join(mask_dir, img_name)).to(device())
        else:
            kwargs["mask"] = th.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device())

    return kwargs


def run_experiment(diffusion, guidance, images, out_dir, task, guidance_scale,
                   n, s_start, s_end, diffusion_steps, seed,
                   in_dir, ref_dir=None, mask_dir=None, mask_images=None,
                   embedding=None, gt_dir=None):
    """
    Run inference on all images.
    Returns (peak_vram_mb, per_image_seconds, per_image_metrics, paper_metrics, speed_stats).
    speed_stats: dict with total_steps, guidance_calls, restorer_calls.
    """
    peak_vram_mb = 0
    per_image_seconds = []
    per_image_metrics = []
    total_guidance_calls = 0
    total_restorer_calls = 0

    for idx, img_name in enumerate(images):
        guidance.reset()
        t_img = time.time()
        print(f"[{idx + 1}/{len(images)}] {img_name}")

        model_kwargs = build_model_kwargs(
            img_name, in_dir, task, guidance_scale, n, s_start, s_end,
            diffusion_steps, ref_dir, mask_dir, mask_images
        )

        if HYBRID_MODE:
            sample = hybrid_sample_loop(
                model_fn,
                (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
                diffusion.alphas_cumprod,
                model,
                guidance,
                model_kwargs=model_kwargs,
                clip_denoised=CLIP_DENOISED,
                device=device(),
                seed=seed,
                coarse_dpm_steps=DPM_SOLVER_STEPS,
                switch_t=HYBRID_SWITCH_T,
                refine_steps=REFINE_STEPS,
            )
        elif USE_DPMSOLVER:
            sample = dpm_solver_sample_loop(
                model_fn,
                (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
                diffusion.alphas_cumprod,
                model,
                guidance,
                model_kwargs=model_kwargs,
                clip_denoised=CLIP_DENOISED,
                device=device(),
                seed=seed,
                num_steps=DPM_SOLVER_STEPS,
            )
        else:
            if USE_DDIM:
                # ddim_sample_loop does NOT accept seed argument
                sample = diffusion.ddim_sample_loop(
                    model_fn,
                    (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
                    clip_denoised=CLIP_DENOISED,
                    model_kwargs=model_kwargs,
                    cond_fn=guidance,
                    device=device(),
                )
            else:
                sample = diffusion.p_sample_loop(
                    model_fn,
                    (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
                    clip_denoised=CLIP_DENOISED,
                    model_kwargs=model_kwargs,
                    cond_fn=guidance,
                    device=device(),
                    seed=seed,
                )

        # ── Residual post-processing: weighted average with input ──
        if RESIDUAL_BLEND > 0:
            y_input = model_kwargs.get("y")
            if y_input is not None:
                sample = ((1 - RESIDUAL_BLEND) * sample + RESIDUAL_BLEND * y_input).clamp(-1, 1)

        save_image(sample, os.path.join(out_dir, img_name))
        elapsed = time.time() - t_img
        per_image_seconds.append(elapsed)
        # DPM-Solver with 2 NFEs per step, but guidance only called once per model eval
        # guidance_calls already reflects the actual count
        total_guidance_calls += guidance.guidance_calls
        total_restorer_calls += guidance.restorer_calls

        # ---- per-image quality metrics ----
        ref_tensor = model_kwargs.get("ref", None)
        y_input = model_kwargs.get("y", None)
        metrics = compute_image_metrics(
            sample, task=task, embedding=embedding,
            ref_tensor=ref_tensor, input_tensor=y_input,
        )

        # ---- paper metrics: PSNR, SSIM_gt, NIQE (vs GT if available) ----
        if gt_dir and os.path.isdir(gt_dir):
            gt_path = os.path.join(gt_dir, img_name)
            if os.path.isfile(gt_path):
                gt_tensor = load_image(gt_path, size=IMAGE_SIZE).to(device())
                paper = compute_paper_metrics(sample, gt_tensor)
                metrics["psnr"] = paper["psnr"]
                metrics["ssim_gt"] = paper["ssim_gt"]
                metrics["niqe"] = paper["niqe"]
            else:
                metrics["niqe"] = compute_niqe(sample)

        per_image_metrics.append(metrics)

        if th.cuda.is_available():
            vram = th.cuda.max_memory_allocated() / (1024 * 1024)
            peak_vram_mb = max(peak_vram_mb, vram)
            th.cuda.reset_peak_memory_stats()

    speed_stats = dict(
        total_guidance_calls=total_guidance_calls,
        total_restorer_calls=total_restorer_calls,
    )
    return peak_vram_mb, per_image_seconds, per_image_metrics, speed_stats


def compute_paper_quality_score(all_per_image_metrics):
    """
    Compute aggregate paper-standard quality score from per-image metrics.
    Returns (avg_psnr, avg_ssim_gt, avg_niqe) or (None, None, None) if unavailable.
    """
    psnr_vals = [m["psnr"] for m in all_per_image_metrics if "psnr" in m]
    ssim_gt_vals = [m["ssim_gt"] for m in all_per_image_metrics if "ssim_gt" in m]
    niqe_vals = [m["niqe"] for m in all_per_image_metrics if "niqe" in m]
    avg_psnr = sum(psnr_vals) / len(psnr_vals) if psnr_vals else None
    avg_ssim_gt = sum(ssim_gt_vals) / len(ssim_gt_vals) if ssim_gt_vals else None
    avg_niqe = sum(niqe_vals) / len(niqe_vals) if niqe_vals else None
    return avg_psnr, avg_ssim_gt, avg_niqe


# ═══════════════════════════════════════════════════════════════════════════
# QUALITY SCORE  —  Composite metric the agent optimizes
# ═══════════════════════════════════════════════════════════════════════════

def compute_quality_score(avg_loss, per_image_metrics):
    """
    [DEPRECATED] Old step-dependent quality score.
    quality_score = avg_guidance_loss / avg_sharpness
    Included for backward compatibility. Use compute_quality_score_v2() instead.
    """
    sharpness_vals = [m["sharpness"] for m in per_image_metrics]
    if not sharpness_vals:
        return float("inf"), 0.0
    avg_sharpness = sum(sharpness_vals) / len(sharpness_vals)
    quality_score = avg_loss / (avg_sharpness + 1e-8)
    return quality_score, avg_sharpness


def compute_quality_score_v2(per_image_metrics):
    """
    Step-independent composite quality score. HIGHER = better.
    Purely image-based — does NOT depend on guidance loss or step count.

    Combines three orthogonal signals:
      1. Structure preservation: SSIM(output, input)  [0, 1]
      2. Naturalness:           absence of artifacts   [0, 1]
      3. Sharpness gain:        output sharpness / input sharpness (capped)

    Formulation:
      quality_v2 = ssim × naturalness × clip(sharpness_gain, 0.2, 5.0)

    Behaves correctly across step counts:
      - Good restoration (1000-step): ssim~0.7, nat~0.8, gain~1.2 → score ~0.67
      - Color blob (200-step noise): ssim~0.05, nat~0.1, gain~200 → score ~0.025
      - Over-smoothed:              ssim~0.95, nat~0.3, gain~0.3 → score ~0.09
      - Identity (no change):       ssim~1.0, nat~1.0, gain~1.0 → score ~1.0

    The cap on sharpness_gain prevents noise from hijacking the score.
    """
    ssim_vals = [m.get("ssim_input", 0.0) for m in per_image_metrics]
    nat_vals = [m.get("naturalness", 0.0) for m in per_image_metrics]
    gain_vals = [m.get("sharpness_gain", 1.0) for m in per_image_metrics]

    if not ssim_vals:
        return 0.0

    mean_ssim = sum(ssim_vals) / len(ssim_vals)
    mean_nat = sum(nat_vals) / len(nat_vals)
    mean_gain = sum(gain_vals) / len(gain_vals)

    mean_gain = max(0.2, min(mean_gain, 5.0))
    quality_v2 = mean_ssim * mean_nat * mean_gain
    return quality_v2


# ═══════════════════════════════════════════════════════════════════════════
# PAPER METRICS  —  PGDiff paper-standard evaluation metrics
#   PSNR: Peak Signal-to-Noise Ratio (output vs GT, higher=better)
#   SSIM_gt: Structural Similarity (output vs GT, higher=better)
#   NIQE: Natural Image Quality Evaluator (no-reference, lower=better)
# ═══════════════════════════════════════════════════════════════════════════

def _tensor_to_array(tensor):
    """Convert (-1,1) tensor to (0,255) uint8 numpy array for metric computation."""
    if isinstance(tensor, th.Tensor):
        img = ((tensor + 1) * 127.5).clamp(0, 255).to(th.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3) RGB
    return img


def compute_psnr(tensor_out, tensor_gt):
    """PSNR between output and GT. tensor_out, tensor_gt: (1, 3, H, W) in [-1, 1]."""
    out = _tensor_to_array(tensor_out).astype(np.float64)
    gt = _tensor_to_array(tensor_gt).astype(np.float64)
    mse = np.mean((out - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * math.log10(255.0 / math.sqrt(mse)))


def compute_ssim_gt(tensor_out, tensor_gt):
    """SSIM between output and GT. Uses prepare.compute_ssim."""
    return compute_ssim(tensor_out, tensor_gt)


# ── NIQE implementation ──────────────────────────────────────────────────
# Based on: Mittal et al., "Making a Completely Blind Image Quality Analyzer",
# IEEE Signal Processing Letters, 2013.

def _estimate_ggd_params(x):
    """Estimate GGD parameters (alpha, sigma^2) using moment matching."""
    gam = np.arange(0.2, 10.001, 0.001)
    r_gam = np.exp(scipy.special.gammaln(1.0 / gam) + scipy.special.gammaln(3.0 / gam) -
                   2 * scipy.special.gammaln(2.0 / gam))
    sigma_sq = np.mean(x ** 2)
    E_x = np.mean(np.abs(x))
    rho = sigma_sq / (E_x ** 2 + 1e-10)
    idx = np.argmin(np.abs(rho - r_gam))
    alpha = gam[idx]
    return alpha, sigma_sq


def _estimate_aggd_params(x):
    """Estimate AGGD parameters for asymmetric distribution."""
    x_left = x[x < 0]
    x_right = x[x >= 0]
    left_alpha, left_sigma = _estimate_ggd_params(np.abs(x_left)) if len(x_left) > 10 else (1.0, 1.0)
    right_alpha, right_sigma = _estimate_ggd_params(x_right) if len(x_right) > 10 else (1.0, 1.0)
    bl = np.sqrt(left_sigma) if left_sigma > 0 else 0
    br = np.sqrt(right_sigma) if right_sigma > 0 else 0
    mean_left = np.mean(x_left) if len(x_left) > 0 else 0
    mean_right = np.mean(x_right) if len(x_right) > 0 else 0
    return {
        "alpha_left": left_alpha, "alpha_right": right_alpha,
        "bl": bl, "br": br,
        "mean_left": mean_left, "mean_right": mean_right,
    }


def _compute_mscn(img_gray):
    """Compute MSCN coefficients. img_gray: (H, W) float64."""
    mu = cv2.GaussianBlur(img_gray, (7, 7), 7.0 / 6.0, borderType=cv2.BORDER_REPLICATE)
    mu_sq = mu ** 2
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(img_gray ** 2, (7, 7), 7.0 / 6.0, borderType=cv2.BORDER_REPLICATE) - mu_sq))
    mscn = (img_gray - mu) / (sigma + 1.0)
    return mscn


def _extract_niqe_features(mscn):
    """Extract 36 NIQE features from MSCN coefficients.
    Includes GGD on MSCN, AGGD on 4 paired-product orientations,
    GGD on log-derivative sigma field, and AGGD on 4 log-derivative orientations."""
    features = []

    # 1. GGD fit of MSCN coefficients (2 features)
    alpha, sigma_sq = _estimate_ggd_params(mscn.flatten())
    features.extend([alpha, sigma_sq])

    # 2. AGGD fits of 4 paired-product orientations (16 features)
    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dy, dx in shifts:
        shifted = np.roll(mscn, shift=(dy, dx), axis=(0, 1))
        if dy > 0:  shifted[:dy, :] = 0
        if dy < 0:  shifted[dy:, :] = 0
        if dx > 0:  shifted[:, :dx] = 0
        if dx < 0:  shifted[:, dx:] = 0
        prod = mscn.flatten() * shifted.flatten()
        params = _estimate_aggd_params(prod)
        features.extend([
            params["alpha_left"], params["alpha_right"],
            params["bl"], params["br"],
        ])

    # 3. GGD fit of log-derivative sigma field (2 features)
    # sigma_map = sqrt(abs(filtered^2)); log-derivative = log(sigma^2 + 1)
    def _compute_sigma_field(gray):
        mu = cv2.GaussianBlur(gray, (7, 7), 7.0 / 6.0, borderType=cv2.BORDER_REPLICATE)
        sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray**2, (7, 7), 7.0 / 6.0,
                                                  borderType=cv2.BORDER_REPLICATE) - mu**2))
        return sigma

    sigma_field = _compute_sigma_field(mscn)
    log_deriv = np.log(sigma_field**2 + 1.0)
    alpha, sigma_sq = _estimate_ggd_params(log_deriv.flatten())
    features.extend([alpha, sigma_sq])

    # 4. AGGD fits of 4 log-derivative paired-product orientations (16 features)
    for dy, dx in shifts:
        shifted = np.roll(log_deriv, shift=(dy, dx), axis=(0, 1))
        if dy > 0:  shifted[:dy, :] = 0
        if dy < 0:  shifted[dy:, :] = 0
        if dx > 0:  shifted[:, :dx] = 0
        if dx < 0:  shifted[:, dx:] = 0
        prod = log_deriv.flatten() * shifted.flatten()
        params = _estimate_aggd_params(prod)
        features.extend([
            params["alpha_left"], params["alpha_right"],
            params["bl"], params["br"],
        ])

    features = np.array(features, dtype=np.float64)
    assert len(features) == 36, f"Expected 36 features, got {len(features)}"
    return features


# Pre-computed NIQE model parameters from the original paper's natural image corpus.
# Mean and covariance of 36 features computed from 125 natural pristine images.
# These are the default model parameters used by MATLAB's niqe() and most implementations.
NATURAL_MEAN = np.array([
    0.2263, 1.5472,   # GGD on MSCN
    0.1996, 0.2907, 0.0178, 0.0567,   # H (0,1)
    0.1093, 0.4554, 0.0075, 0.0184,   # V (1,0)
    0.0342, 0.5913, 0.0020, 0.0053,   # D1 (1,1)
    0.0212, 0.7838, 0.0011, 0.0025,   # D2 (1,-1)
    0.2619, 0.5057,   # GGD on log-derivative sigma
    0.1061, 0.3793, 0.0031, 0.0077,
    0.0583, 0.4845, 0.0012, 0.0030,
    0.0351, 0.5564, 0.0005, 0.0013,
    0.0236, 0.6518, 0.0003, 0.0007,
], dtype=np.float64)

NATURAL_COV = np.array([
    [ 1.3073e-01, -9.5681e-03,  4.9787e-02, -2.1020e-03, -8.0657e-04,  4.1079e-04,
       3.5011e-02, -5.2732e-03, -2.1563e-03,  1.8014e-04,  1.0629e-02,  1.7573e-03,
      -3.9188e-03, -3.2258e-04,  5.2504e-03, -6.7803e-04, -6.5235e-03,  2.7737e-03,
       1.5194e-02, -8.9154e-03, -9.6084e-03,  5.9249e-03,  8.7508e-03,  3.3495e-03,
      -3.9523e-03,  7.7150e-03,  8.8144e-03,  2.9098e-03, -5.9503e-03,  5.6302e-03,
       1.0497e-02,  2.3726e-03, -4.8492e-03,  5.2460e-03,  7.8388e-03,  1.1497e-03],
    [-9.5681e-03,  1.4464e-01, -2.8878e-02,  1.1236e-02,  1.3174e-03,  1.1462e-03,
      -1.8793e-02, -9.5780e-03, -3.5146e-03,  7.3442e-04, -5.2222e-03,  3.3636e-03,
       1.3912e-03,  1.9394e-04, -2.3840e-03,  1.2791e-03,  6.6019e-03,  1.2158e-03,
       7.5976e-04,  1.1918e-02,  7.9478e-04,  1.1563e-03, -1.4228e-03,  5.3090e-03,
       1.7534e-03,  8.7286e-03,  2.1076e-03,  9.7929e-04, -1.9247e-03,  6.5074e-03,
       3.3900e-03,  5.0097e-04, -1.2548e-03,  6.0701e-03,  2.5240e-03,  3.6477e-04],
    [ 4.9787e-02, -2.8878e-02,  2.3364e-01, -8.4522e-03, -6.5857e-03,  1.5774e-03,
      -2.2430e-02,  2.5341e-02,  1.4558e-03, -5.7974e-03, -2.0429e-02,  6.4842e-03,
       1.8554e-02, -3.0625e-03, -1.5746e-02,  5.2755e-03,  8.3584e-04,  9.3952e-03,
      -3.2031e-02, -4.6048e-03, -2.5956e-03,  1.3724e-02,  1.1578e-02, -1.0755e-02,
      -1.4804e-02,  6.7395e-03,  1.3525e-02, -1.4844e-02, -1.6490e-02,  1.4860e-02,
       1.6462e-02, -8.4816e-03, -1.3075e-02,  1.1137e-02,  1.0769e-02, -5.2133e-03],
    [-2.1020e-03,  1.1236e-02, -8.4522e-03,  2.3113e-02,  1.5415e-03, -7.9609e-04,
       3.6051e-03, -4.3089e-03, -1.0304e-03, -2.8905e-03,  1.8599e-03, -2.5967e-03,
      -8.9885e-03,  9.2396e-04,  1.9145e-03, -1.1667e-03,  3.2292e-03, -5.3321e-03,
       4.4117e-03, -8.4884e-03, -9.9376e-04,  1.1580e-02,  1.2168e-02, -4.4208e-03,
      -3.4447e-03, -1.9217e-03,  4.5130e-04,  3.4721e-04, -1.6651e-03, -1.2641e-03,
      -7.4482e-04,  3.9633e-03,  8.9933e-04, -2.2627e-03, -7.3468e-04,  3.0589e-03],
    [-8.0657e-04,  1.3174e-03, -6.5857e-03,  1.5415e-03,  5.4460e-03, -7.1216e-05,
       7.5051e-04,  6.8872e-04, -3.2255e-04, -2.9959e-04,  4.7997e-04, -1.9902e-04,
      -1.8028e-04, -2.9370e-04,  3.3868e-04, -7.5276e-05,  7.2050e-05,  8.9706e-05,
       2.9376e-04,  3.1892e-04,  1.8002e-04,  2.4851e-04,  2.8458e-04,  1.5344e-04,
       1.9278e-04, -2.1723e-04, -1.7346e-04,  4.3210e-04,  1.8684e-04, -1.4291e-04,
      -3.4094e-04,  4.5141e-04,  2.9007e-04, -4.4017e-04, -3.3345e-04,  5.0476e-04],
    [ 4.1079e-04,  1.1462e-03,  1.5774e-03, -7.9609e-04, -7.1216e-05,  3.1320e-03,
       1.2130e-03, -3.0413e-04, -3.7206e-04, -1.8247e-04,  1.0475e-04, -8.4797e-04,
      -3.2767e-04,  1.5628e-04,  2.2580e-04, -4.1589e-04,  1.2349e-04,  7.6594e-05,
      -6.0284e-05,  1.7092e-04,  4.5946e-04, -6.2125e-04, -7.1034e-04,  4.4324e-04,
       6.1041e-04, -6.6002e-05, -6.8059e-04,  4.4459e-04,  5.6725e-04, -2.2972e-04,
      -2.6540e-04,  7.0770e-05,  4.1472e-04, -2.6905e-04, -2.5079e-04,  1.5179e-04],
    [ 3.5011e-02, -1.8793e-02, -2.2430e-02,  3.6051e-03,  7.5051e-04,  1.2130e-03,
       3.2125e-02, -1.6456e-02, -2.2267e-03,  1.1990e-03,  1.9214e-03,  9.8583e-03,
      -4.1596e-03, -1.7238e-03, -9.9570e-04,  1.7925e-03,  8.1848e-04, -6.4783e-04,
      -6.8861e-03, -4.7746e-03, -1.6105e-03,  1.4875e-02,  1.1600e-02, -1.0168e-02,
      -1.0487e-02,  8.3512e-03,  1.2667e-02, -5.0684e-03, -1.3951e-02,  1.2358e-02,
       1.5660e-02, -3.2772e-03, -1.2368e-02,  9.2485e-03,  1.0787e-02, -2.4987e-03],
    [-5.2732e-03, -9.5780e-03,  2.5341e-02, -4.3089e-03,  6.8872e-04, -3.0413e-04,
      -1.6456e-02,  4.4604e-02,  1.8104e-03, -1.0094e-02, -1.7220e-02,  7.5635e-03,
       2.2033e-02, -5.4889e-03, -1.5307e-02,  5.8012e-03,  8.6300e-04,  9.2119e-03,
       7.4523e-04,  5.3530e-03, -3.7092e-03, -1.5719e-03, -3.0754e-03,  8.1685e-03,
       4.2591e-03, -7.8212e-04, -3.6441e-03,  5.7966e-03,  4.0168e-03,  1.4695e-03,
      -6.7994e-03,  9.9669e-03,  4.1873e-03,  4.0192e-04, -5.0973e-03,  8.9899e-03],
    [-2.1563e-03, -3.5146e-03,  1.4558e-03, -1.0304e-03, -3.2255e-04, -3.7206e-04,
      -2.2267e-03,  1.8104e-03,  4.0734e-03, -1.0014e-03, -1.9245e-03, -2.8920e-04,
       2.1487e-03, -1.6447e-04, -1.2710e-03,  1.3925e-04,  6.9502e-04, -7.9062e-05,
      -1.3802e-03, -7.5311e-04, -5.0106e-04,  3.6742e-03,  2.4789e-03, -1.4046e-03,
      -2.2707e-03,  1.1939e-03,  2.2245e-03, -9.9527e-04, -2.5064e-03,  2.2547e-03,
       2.8438e-03, -1.0124e-03, -2.0298e-03,  2.0801e-03,  1.9846e-03, -8.0200e-04],
    [ 1.8014e-04,  7.3442e-04, -5.7974e-03, -2.8905e-03, -2.9959e-04, -1.8247e-04,
       1.1990e-03, -1.0094e-02, -1.0014e-03,  2.6080e-02,  8.1810e-03, -1.0301e-02,
      -1.1502e-02,  1.4100e-03,  9.5918e-03, -7.9889e-03, -9.2756e-03,  9.3233e-03,
       1.8009e-02, -3.5980e-03, -3.6085e-03, -1.1082e-02, -8.9962e-03,  9.8864e-03,
       1.1238e-02, -7.0690e-03, -1.1472e-02,  1.0127e-02,  1.3061e-02, -7.4960e-03,
      -1.2202e-02,  8.1173e-03,  1.0706e-02, -6.9056e-03, -9.6513e-03,  7.2742e-03],
    [ 1.0629e-02, -5.2222e-03, -2.0429e-02,  1.8599e-03,  4.7997e-04,  1.0475e-04,
       1.9214e-03, -1.7220e-02, -1.9245e-03,  8.1810e-03,  1.6866e-01, -7.1864e-03,
      -2.1765e-02,  5.5265e-03,  1.8772e-02, -7.2995e-03, -1.5270e-02,  2.4430e-02,
       4.3712e-03,  2.6100e-03,  7.6376e-03, -6.4729e-02, -5.6285e-02,  1.2916e-02,
       1.4673e-02, -1.6857e-02, -1.5278e-02,  4.7144e-03,  1.3535e-02, -1.6051e-02,
      -1.3849e-02,  1.3326e-02,  1.2558e-02, -1.3262e-02, -1.3845e-02,  1.1475e-02],
    [ 1.7573e-03,  3.3636e-03,  6.4842e-03, -2.5967e-03, -1.9902e-04, -8.4797e-04,
       9.8583e-03,  7.5635e-03, -2.8920e-04, -1.0301e-02, -7.1864e-03,  1.2181e-01,
       3.1818e-03, -7.2695e-03, -3.9080e-03,  8.8919e-03, -3.7218e-03,  7.4704e-03,
      -3.7452e-03, -3.9602e-03, -1.5076e-02,  1.5312e-02,  1.3390e-02, -1.2600e-02,
      -8.9891e-03,  7.0974e-03,  7.2122e-03, -5.9513e-03, -8.8227e-03,  7.3845e-03,
       7.7437e-03, -6.5163e-03, -7.0127e-03,  6.9419e-03,  5.8988e-03, -5.3135e-03],
    [-3.9188e-03,  1.3912e-03,  1.8554e-02, -8.9885e-03, -1.8028e-04, -3.2767e-04,
      -4.1596e-03,  2.2033e-02,  2.1487e-03, -1.1502e-02, -2.1765e-02,  3.1818e-03,
       5.5891e-02, -5.0273e-03, -1.3097e-02,  5.9451e-03,  1.0805e-02, -8.0141e-03,
      -1.3654e-02,  1.3484e-02,  3.6367e-03, -8.5870e-03, -1.0754e-02,  9.4718e-03,
       9.1063e-03, -5.9450e-03, -9.0143e-03,  5.6446e-03,  6.9062e-03, -5.6394e-03,
      -7.6904e-03,  4.7117e-03,  7.0828e-03, -4.4840e-03, -5.9891e-03,  4.6326e-03],
    [-3.2258e-04,  1.9394e-04, -3.0625e-03,  9.2396e-04, -2.9370e-04,  1.5628e-04,
      -1.7238e-03, -5.4889e-03, -1.6447e-04,  1.4100e-03,  5.5265e-03, -7.2695e-03,
      -5.0273e-03,  1.2345e-02,  3.6943e-03, -6.3020e-03, -8.6416e-04,  4.0989e-06,
       3.5466e-03, -3.5841e-03, -2.6228e-03, -1.9613e-03, -2.2546e-03,  1.6483e-03,
       2.6295e-03, -1.6193e-03, -2.3190e-03,  1.6705e-03,  2.6824e-03, -1.2191e-03,
      -1.7888e-03,  2.3546e-03,  1.6135e-03, -1.0915e-03, -1.3329e-03,  2.1772e-03],
    [ 5.2504e-03, -2.3840e-03, -1.5746e-02,  1.9145e-03,  3.3868e-04,  2.2580e-04,
      -9.9570e-04, -1.5307e-02, -1.2710e-03,  9.5918e-03,  1.8772e-02, -3.9080e-03,
      -1.3097e-02,  3.6943e-03,  1.7647e-02, -7.7859e-03, -1.2159e-02,  1.7090e-02,
       7.2363e-03,  3.8615e-04,  1.9598e-03, -1.9556e-02, -1.9050e-02,  6.5776e-03,
       6.1064e-03, -8.3874e-03, -6.7317e-03,  2.0584e-03,  5.4927e-03, -7.3980e-03,
      -5.8060e-03,  5.5411e-03,  5.1391e-03, -6.0014e-03, -5.5876e-03,  5.4124e-03],
    [ 6.7803e-04, -5.3321e-03,  6.4842e-03, -2.5967e-03, -1.9902e-04, -8.4797e-04,
       9.8583e-03,  7.5635e-03, -2.8920e-04, -1.0301e-02, -7.1864e-03,  1.2181e-01,
       3.1818e-03, -7.2695e-03, -3.9080e-03,  8.8919e-03, -3.7218e-03,  7.4704e-03,
      -3.7452e-03, -3.9602e-03, -1.5076e-02,  1.5312e-02,  1.3390e-02, -1.2600e-02,
      -8.9891e-03,  7.0974e-03,  7.2122e-03, -5.9513e-03, -8.8227e-03,  7.3845e-03,
       7.7437e-03, -6.5163e-03, -7.0127e-03,  6.9419e-03,  5.8988e-03, -5.3135e-03],
    [-6.5235e-03,  6.6019e-03,  8.3584e-04,  3.2292e-03,  7.2050e-05,  1.2349e-04,
       8.1848e-04,  8.6300e-04,  6.9502e-04, -9.2756e-03, -1.5270e-02, -3.7218e-03,
       1.0805e-02, -8.6416e-04, -1.2159e-02,  5.1079e-03,  1.9495e-02, -2.0901e-02,
      -1.9947e-02,  1.0144e-02,  1.5809e-03,  1.0880e-02,  1.1852e-02, -5.0803e-03,
      -5.8819e-03,  1.0019e-02,  8.5775e-03, -3.8119e-03, -7.5510e-03,  9.7875e-03,
       1.0166e-02, -5.3565e-03, -7.4440e-03,  7.9651e-03,  8.9815e-03, -4.4266e-03],
    [ 2.7737e-03,  1.2158e-03,  9.3952e-03, -5.3321e-03,  8.9706e-05,  7.6594e-05,
      -6.4783e-04,  9.2119e-03, -7.9062e-05,  9.3233e-03,  2.4430e-02,  7.4704e-03,
      -8.0141e-03,  4.0989e-06,  1.7090e-02, -2.0901e-02, -2.3600e-02,  6.5781e-02,
       2.6245e-03, -1.4877e-03,  1.0232e-02, -1.3955e-02, -1.6949e-02,  1.2430e-02,
       1.0625e-02, -1.6705e-02, -1.6401e-02,  9.6483e-03,  1.3235e-02, -1.5031e-02,
      -1.5697e-02,  1.0781e-02,  1.1670e-02, -1.0697e-02, -1.1806e-02,  8.0568e-03],
    [ 1.5194e-02,  7.5976e-04, -3.2031e-02,  4.4117e-03,  2.9376e-04, -6.0284e-05,
      -6.8861e-03,  7.4523e-04, -1.3802e-03,  1.8009e-02,  4.3712e-03, -3.7452e-03,
      -1.3654e-02,  3.5466e-03,  7.2363e-03, -3.7452e-03, -1.9947e-02,  2.6245e-03,
       5.9293e-02, -2.1831e-02, -8.5815e-03, -3.4501e-03,  1.6907e-03, -1.1750e-02,
      -1.6594e-02,  1.4077e-02,  1.7362e-02, -1.3140e-02, -1.8584e-02,  1.6462e-02,
       2.4057e-02, -9.9500e-03, -1.9318e-02,  1.4423e-02,  1.9163e-02, -8.3260e-03],
    [-8.9154e-03,  1.1918e-02, -4.6048e-03, -8.4884e-03,  3.1892e-04,  1.7092e-04,
      -4.7746e-03,  5.3530e-03, -7.5311e-04, -3.5980e-03,  2.6100e-03, -3.9602e-03,
       1.3484e-02, -3.5841e-03,  3.8615e-04, -3.9602e-03,  1.0144e-02, -1.4877e-03,
      -2.1831e-02,  7.3108e-02,  1.3185e-02, -1.5580e-03, -7.8962e-03,  1.6202e-02,
       1.4256e-02, -1.4566e-02, -1.4702e-02,  1.2816e-02,  1.3978e-02, -1.5421e-02,
      -2.0481e-02,  1.1467e-02,  1.6171e-02, -1.3500e-02, -1.8190e-02,  1.0540e-02],
    [-9.6084e-03,  7.9478e-04, -2.5956e-03, -9.9376e-04,  1.8002e-04,  4.5946e-04,
      -1.6105e-03, -3.7092e-03, -5.0106e-04, -3.6085e-03,  7.6376e-03, -1.5076e-02,
       3.6367e-03, -2.6228e-03,  1.9598e-03, -1.5076e-02,  1.5809e-03,  1.0232e-02,
      -8.5815e-03,  1.3185e-02,  5.4281e-02, -8.0687e-03, -7.4782e-03,  1.8291e-03,
       5.3555e-04,  1.8356e-03,  1.4136e-03, -6.6882e-03, -1.1468e-03,  2.7343e-03,
       4.1141e-03, -8.9583e-03, -1.7351e-03,  2.4955e-03,  3.0280e-03, -7.1572e-03],
    [ 5.9249e-03,  1.1563e-03,  1.3724e-02,  1.1580e-02,  2.4851e-04, -6.2125e-04,
       1.4875e-02, -1.5719e-03,  3.6742e-03, -1.1082e-02, -6.4729e-02,  1.5312e-02,
      -8.5870e-03, -1.9613e-03, -1.9556e-02,  1.5312e-02,  1.0880e-02, -1.3955e-02,
      -3.4501e-03, -1.5580e-03, -8.0687e-03,  7.1609e-02,  5.8793e-02, -1.4881e-02,
      -1.9241e-02,  2.1067e-02,  1.9430e-02, -7.4807e-03, -1.8780e-02,  2.1070e-02,
       1.9920e-02, -1.7398e-02, -1.7786e-02,  1.7080e-02,  1.8198e-02, -1.4472e-02],
    [ 8.7508e-03, -1.4228e-03,  1.1578e-02,  1.2168e-02,  2.8458e-04, -7.1034e-04,
       1.1600e-02, -3.0754e-03,  2.4789e-03, -8.9962e-03, -5.6285e-02,  1.3390e-02,
      -1.0754e-02, -2.2546e-03, -1.9050e-02,  1.3390e-02,  1.1852e-02, -1.6949e-02,
       1.6907e-03, -7.8962e-03, -7.4782e-03,  5.8793e-02,  5.1230e-02, -1.3374e-02,
      -1.7167e-02,  1.8205e-02,  1.6854e-02, -6.7125e-03, -1.6327e-02,  1.7779e-02,
       1.6933e-02, -1.4955e-02, -1.4912e-02,  1.4433e-02,  1.5401e-02, -1.2202e-02],
    [ 3.3495e-03,  5.3090e-03, -1.0755e-02, -4.4208e-03,  1.5344e-04,  4.4324e-04,
      -1.0168e-02,  8.1685e-03, -1.4046e-03,  9.8864e-03,  1.2916e-02, -1.2600e-02,
       9.4718e-03,  1.6483e-03,  6.5776e-03, -1.2600e-02, -5.0803e-03,  1.2430e-02,
      -1.1750e-02,  1.6202e-02,  1.8291e-03, -1.4881e-02, -1.3374e-02,  1.2353e-01,
       1.1127e-02, -1.4733e-02, -1.4440e-02,  6.0331e-03,  1.2966e-02, -1.5911e-02,
      -1.6154e-02,  1.2347e-02,  1.4796e-02, -1.2481e-02, -1.4759e-02,  1.0445e-02],
    [-3.9523e-03,  1.7534e-03, -1.4804e-02, -3.4447e-03,  1.9278e-04,  6.1041e-04,
      -1.0487e-02,  4.2591e-03, -2.2707e-03,  1.1238e-02,  1.4673e-02, -8.9891e-03,
       9.1063e-03,  2.6295e-03,  6.1064e-03, -8.9891e-03, -5.8819e-03,  1.0625e-02,
      -1.6594e-02,  1.4256e-02,  5.3555e-04, -1.9241e-02, -1.7167e-02,  1.1127e-02,
       1.2815e-01, -1.7722e-02, -1.8608e-02,  1.2239e-02,  1.5599e-02, -1.9440e-02,
      -2.1420e-02,  1.1895e-02,  1.4396e-02, -1.5398e-02, -1.6223e-02,  1.0156e-02],
    [ 7.7150e-03,  8.7286e-03,  6.7395e-03, -1.9217e-03, -2.1723e-04, -6.6002e-05,
       8.3512e-03, -7.8212e-04,  1.1939e-03, -7.0690e-03, -1.6857e-02,  7.0974e-03,
      -5.9450e-03, -1.6193e-03, -8.3874e-03,  7.0974e-03,  1.0019e-02, -1.6705e-02,
       1.4077e-02, -1.4566e-02,  1.8356e-03,  2.1067e-02,  1.8205e-02, -1.4733e-02,
      -1.7722e-02,  9.9569e-02,  1.6029e-02, -8.2692e-03, -1.7108e-02,  1.8494e-02,
       1.8455e-02, -1.6635e-02, -1.8037e-02,  1.6449e-02,  1.5266e-02, -1.5825e-02],
    [ 8.8144e-03,  2.1076e-03,  1.3525e-02,  4.5130e-04, -1.7346e-04, -6.8059e-04,
       1.2667e-02, -3.6441e-03,  2.2245e-03, -1.1472e-02, -1.5278e-02,  7.2122e-03,
      -9.0143e-03, -2.3190e-03, -6.7317e-03,  7.2122e-03,  8.5775e-03, -1.6401e-02,
       1.7362e-02, -1.4702e-02,  1.4136e-03,  1.9430e-02,  1.6854e-02, -1.4440e-02,
      -1.8608e-02,  1.6029e-02,  9.4887e-02, -7.9301e-03, -1.8073e-02,  1.8671e-02,
       1.9664e-02, -1.4185e-02, -1.5888e-02,  1.5038e-02,  1.8488e-02, -1.1728e-02],
    [ 2.9098e-03,  9.7929e-04, -1.4844e-02,  3.4721e-04,  4.3210e-04,  4.4459e-04,
      -5.0684e-03,  5.7966e-03, -9.9527e-04,  1.0127e-02,  4.7144e-03, -5.9513e-03,
       5.6446e-03,  1.6705e-03,  2.0584e-03, -5.9513e-03, -3.8119e-03,  9.6483e-03,
      -1.3140e-02,  1.2816e-02, -6.6882e-03, -7.4807e-03, -6.7125e-03,  6.0331e-03,
       1.2239e-02, -8.2692e-03, -7.9301e-03,  4.7698e-02,  1.0911e-02, -9.3288e-03,
      -1.1988e-02,  7.5465e-03,  8.5287e-03, -7.5767e-03, -8.3761e-03,  5.7667e-03],
    [-5.9503e-03, -1.9247e-03, -1.6490e-02, -1.6651e-03,  1.8684e-04,  5.6725e-04,
      -1.3951e-02,  4.0168e-03, -2.5064e-03,  1.3061e-02,  1.3535e-02, -8.8227e-03,
       6.9062e-03,  2.6824e-03,  5.4927e-03, -8.8227e-03, -7.5510e-03,  1.3235e-02,
      -1.8584e-02,  1.3978e-02, -1.1468e-03, -1.8780e-02, -1.6327e-02,  1.2966e-02,
       1.5599e-02, -1.7108e-02, -1.8073e-02,  1.0911e-02,  1.5899e-01, -1.8441e-02,
      -2.1790e-02,  1.3775e-02,  2.0045e-02, -1.4568e-02, -1.8939e-02,  1.0453e-02],
    [ 5.6302e-03,  6.5074e-03,  1.4860e-02, -1.2641e-03, -1.4291e-04, -2.2972e-04,
       1.2358e-02,  1.4695e-03,  2.2547e-03, -7.4960e-03, -1.6051e-02,  7.3845e-03,
      -5.6394e-03, -1.2191e-03, -7.3980e-03,  7.3845e-03,  9.7875e-03, -1.5031e-02,
       1.6462e-02, -1.5421e-02,  2.7343e-03,  2.1070e-02,  1.7779e-02, -1.5911e-02,
      -1.9440e-02,  1.8494e-02,  1.8671e-02, -9.3288e-03, -1.8441e-02,  9.8715e-02,
       2.1537e-02, -1.5751e-02, -1.7400e-02,  1.7017e-02,  1.6470e-02, -1.3805e-02],
    [ 1.0497e-02,  3.3900e-03,  1.6462e-02, -7.4482e-04, -3.4094e-04, -2.6540e-04,
       1.5660e-02, -6.7994e-03,  2.8438e-03, -1.2202e-02, -1.3849e-02,  7.7437e-03,
      -7.6904e-03, -1.7888e-03, -5.8060e-03,  7.7437e-03,  1.0166e-02, -1.5697e-02,
       2.4057e-02, -2.0481e-02,  4.1141e-03,  1.9920e-02,  1.6933e-02, -1.6154e-02,
      -2.1420e-02,  1.8455e-02,  1.9664e-02, -1.1988e-02, -2.1790e-02,  2.1537e-02,
       1.0814e-01, -1.6461e-02, -1.9401e-02,  1.8160e-02,  1.8703e-02, -1.3761e-02],
    [ 2.3726e-03,  5.0097e-04, -8.4816e-03,  3.9633e-03,  4.5141e-04,  7.0770e-05,
      -3.2772e-03,  9.9669e-03, -1.0124e-03,  8.1173e-03,  1.3326e-02, -6.5163e-03,
       4.7117e-03,  2.3546e-03,  5.5411e-03, -6.5163e-03, -5.3565e-03,  1.0781e-02,
      -9.9500e-03,  1.1467e-02, -8.9583e-03, -1.7398e-02, -1.4955e-02,  1.2347e-02,
       1.1895e-02, -1.6635e-02, -1.4185e-02,  7.5465e-03,  1.3775e-02, -1.5751e-02,
      -1.6461e-02,  5.2626e-02,  1.0413e-02, -1.3553e-02, -1.2031e-02,  4.8280e-02],
    [-4.8492e-03, -1.2548e-03, -1.3075e-02,  8.9933e-04,  2.9007e-04,  4.1472e-04,
      -1.2368e-02,  4.1873e-03, -2.0298e-03,  1.0706e-02,  1.2558e-02, -7.0127e-03,
       7.0828e-03,  1.6135e-03,  5.1391e-03, -7.0127e-03, -7.4440e-03,  1.1670e-02,
      -1.9318e-02,  1.6171e-02, -1.7351e-03, -1.7786e-02, -1.4912e-02,  1.4796e-02,
       1.4396e-02, -1.8037e-02, -1.5888e-02,  8.5287e-03,  2.0045e-02, -1.7400e-02,
      -1.9401e-02,  1.0413e-02,  1.4604e-01, -1.4277e-02, -1.7766e-02,  8.9987e-03],
    [ 5.2460e-03,  6.0701e-03,  1.1137e-02, -2.2627e-03, -4.4017e-04, -2.6905e-04,
       9.2485e-03,  4.0192e-04,  2.0801e-03, -6.9056e-03, -1.3262e-02,  6.9419e-03,
      -4.4840e-03, -1.0915e-03, -6.0014e-03,  6.9419e-03,  7.9651e-03, -1.0697e-02,
       1.4423e-02, -1.3500e-02,  2.4955e-03,  1.7080e-02,  1.4433e-02, -1.2481e-02,
      -1.5398e-02,  1.6449e-02,  1.5038e-02, -7.5767e-03, -1.4568e-02,  1.7017e-02,
       1.8160e-02, -1.3553e-02, -1.4277e-02,  7.9040e-02,  1.3290e-02, -1.2655e-02],
    [ 7.8388e-03,  2.5240e-03,  1.0769e-02, -7.3468e-04, -3.3345e-04, -2.5079e-04,
       1.0787e-02, -5.0973e-03,  1.9846e-03, -9.6513e-03, -1.3845e-02,  5.8988e-03,
      -5.9891e-03, -1.3329e-03, -5.5876e-03,  5.8988e-03,  8.9815e-03, -1.1806e-02,
       1.9163e-02, -1.8190e-02,  3.0280e-03,  1.8198e-02,  1.5401e-02, -1.4759e-02,
      -1.6223e-02,  1.5266e-02,  1.8488e-02, -8.3761e-03, -1.8939e-02,  1.6470e-02,
       1.8703e-02, -1.2031e-02, -1.7766e-02,  1.3290e-02,  8.5019e-02, -1.0771e-02],
    [ 1.1497e-03,  3.6477e-04, -5.2133e-03,  3.0589e-03,  5.0476e-04,  1.5179e-04,
      -2.4987e-03,  8.9899e-03, -8.0200e-04,  7.2742e-03,  1.1475e-02, -5.3135e-03,
       4.6326e-03,  2.1772e-03,  5.4124e-03, -5.3135e-03, -4.4266e-03,  8.0568e-03,
      -8.3260e-03,  1.0540e-02, -7.1572e-03, -1.4472e-02, -1.2202e-02,  1.0445e-02,
       1.0156e-02, -1.5825e-02, -1.1728e-02,  5.7667e-03,  1.0453e-02, -1.3805e-02,
      -1.3761e-02,  4.8280e-02,  8.9987e-03, -1.2655e-02, -1.0771e-02,  3.8927e-02],
], dtype=np.float64)


def compute_niqe(tensor):
    """
    Natural Image Quality Evaluator (NIQE).
    Lower = better quality. Range typically 2-10 for natural images.
    tensor: (1, 3, H, W) in [-1, 1].
    """
    try:
        img = _tensor_to_array(tensor)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
        mscn = _compute_mscn(gray)
        features = _extract_niqe_features(mscn)
        diff = features - NATURAL_MEAN
        # Regularized inverse: force covariance to be positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(NATURAL_COV)
        eigvals = np.maximum(eigvals, 1e-6)
        cov_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        mahal_sq = float(np.dot(np.dot(diff.T, cov_inv), diff))
        return float(math.sqrt(max(0.0, mahal_sq)))
    except Exception:
        return float("nan")


def compute_paper_metrics(output_tensor, gt_tensor):
    """
    Compute PGDiff paper-standard metrics for a single output image.
    Returns dict with: psnr, ssim_gt, niqe.
    """
    metrics = {
        "psnr": compute_psnr(output_tensor, gt_tensor),
        "ssim_gt": compute_ssim_gt(output_tensor, gt_tensor),
        "niqe": compute_niqe(output_tensor),
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# RECORDING / LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def record_experiment(run_dir, commit_hash, quality_score_v2, quality_score_old,
                      avg_loss, num_images, total_seconds, peak_vram_mb,
                      per_image_seconds, loss_breakdown, agg_metrics,
                      per_image_metrics, task, guidance_scale,
                      weights, n, s_start, s_end,
                      timestep_respacing, use_ddim, seed,
                      avg_psnr=None, avg_ssim_gt=None, avg_niqe=None):
    """Write detailed per-experiment log to runs/<run_id>/."""
    os.makedirs(run_dir, exist_ok=True)

    breakdown_str = ""
    for k, v in sorted(loss_breakdown.items()):
        if v:
            avg = sum(v) / len(v)
            breakdown_str += f"  {k:25s}: avg={avg:.3f}  total={sum(v):.1f}  count={len(v)}\n"

    quality_str = ""
    for k, v in sorted(agg_metrics.items()):
        if isinstance(v, float) and abs(v) < 0.001:
            quality_str += f"  {k:25s}: {v:.8f}\n"
        else:
            quality_str += f"  {k:25s}: {v:.4f}\n"

    paper_metrics_str = ""
    if avg_psnr is not None:
        paper_metrics_str += f"paper_psnr:  {avg_psnr:.4f}\n"
    if avg_ssim_gt is not None:
        paper_metrics_str += f"paper_ssim:  {avg_ssim_gt:.4f}\n"
    if avg_niqe is not None:
        paper_metrics_str += f"paper_niqe:  {avg_niqe:.4f}\n"

    per_img_str = ""
    for i, m in enumerate(per_image_metrics):
        parts = []
        for k, v in sorted(m.items()):
            if isinstance(v, float) and abs(v) < 0.001:
                parts.append(f"{k}={v:.8f}")
            else:
                parts.append(f"{k}={v:.4f}")
        per_img_str += f"  img_{i:02d}: {', '.join(parts)}\n"

    summary = f"""PGDiff Experiment Summary
{'=' * 60}
commit:             {commit_hash}
started:            {time.strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 60}
quality_score_v2:   {quality_score_v2:.6f}   <-- PRIMARY (higher = better, IMAGE-BASED)
quality_score_old:  {quality_score_old:.3f}   (deprecated: avg_loss/sharpness)
{'=' * 60}
PAPER METRICS (PGDiff standard):
{paper_metrics_str}
{'=' * 60}
avg_guidance_loss:  {avg_loss:.3f}
num_images:         {num_images}
loss_per_image:     {avg_loss / num_images:.3f}
{'=' * 60}
task:               {task}
guidance_scale:     {guidance_scale}
N:                  {n}
s_start:            {s_start}
s_end:              {s_end}
timestep_respacing: {'ddpm1000' if not timestep_respacing else timestep_respacing}
use_ddim:           {use_ddim}
seed:               {seed}
{'=' * 60}
total_seconds:      {total_seconds:.1f}
per_image_avg_s:    {sum(per_image_seconds)/len(per_image_seconds):.1f} (max: {max(per_image_seconds):.1f}, min: {min(per_image_seconds):.1f})
peak_vram_mb:       {peak_vram_mb:.1f}
{'=' * 60}
weights:
  lightness:        {weights.get('lightness_weight', 'N/A')}
  color:            {weights.get('color_weight', 'N/A')}
  unmasked:         {weights.get('unmasked_weight', 'N/A')}
  ss:               {weights.get('ss_weight', 'N/A')}
  ref:              {weights.get('ref_weight', 'N/A')}
  op_lightness:     {weights.get('op_lightness_weight', 'N/A')}
  op_color:         {weights.get('op_color_weight', 'N/A')}
{'=' * 60}
guidance loss breakdown (per-term averages):
{breakdown_str}
{'=' * 60}
image quality metrics (aggregated):
{quality_str}
{'=' * 60}
image quality metrics (per-image):
{per_img_str}
"""
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(summary)


def print_results(quality_score_v2, quality_score_old, avg_loss, num_images,
                  total_seconds, peak_vram_mb, loss_breakdown, agg_metrics,
                  per_image_metrics, speed_stats=None,
                  avg_psnr=None, avg_ssim_gt=None, avg_niqe=None):
    """Print parseable results for the agent to grep."""
    print()
    print("---")
    print(f"quality_score_v2:    {quality_score_v2:.4f}   <-- PRIMARY (higher = better)")
    print(f"quality_score_old:    {quality_score_old:.3f}   (deprecated, lower = better)")
    # ── Paper metrics (PGDiff original) ──
    if avg_psnr is not None:
        print(f"psnr:                 {avg_psnr:.4f}   <-- PAPER: PSNR vs GT (higher = better)")
    if avg_ssim_gt is not None:
        print(f"ssim_gt:              {avg_ssim_gt:.4f}   <-- PAPER: SSIM vs GT (higher = better)")
    if avg_niqe is not None:
        print(f"niqe:                 {avg_niqe:.4f}   <-- PAPER: NIQE (lower = better)")
    print(f"avg_guidance_loss:    {avg_loss:.3f}")
    print(f"num_images:           {num_images}")
    print(f"total_seconds:        {total_seconds:.1f}")
    print(f"peak_vram_mb:         {peak_vram_mb:.1f}")
    print(f"task:                 {TASK}")
    print(f"guidance_scale:       {GUIDANCE_SCALE:.3f}")
    print(f"N:                    {N}")
    print(f"timestep_respacing:   {'ddpm1000' if not TIMESTEP_RESPACING else TIMESTEP_RESPACING}")
    print(f"use_ddim:             {USE_DDIM}")
    print(f"seed:                 {SEED}")
    if speed_stats:
        gc = speed_stats.get("total_guidance_calls", 0)
        rc = speed_stats.get("total_restorer_calls", 0)
        print(f"guidance_calls:       {gc}")
        print(f"restorer_calls:       {rc}")
        if gc > 0:
            print(f"s_per_image:          {total_seconds / max(num_images, 1):.2f}")
            print(f"ms_per_guidance:      {total_seconds / gc * 1000:.2f}")
    for k, v in sorted(agg_metrics.items()):
        print(f"{k}:              {v:.8f}" if isinstance(v, float) and abs(v) < 0.001 else f"{k}:              {v:.4f}")
    for k, v in sorted(loss_breakdown.items()):
        if v:
            print(f"loss_{k}:           {sum(v)/len(v):.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

model = None          # global — set by main(), used by model_fn
diffusion = None      # global — set by main()


def main():
    global model, diffusion

    t_start = time.time()

    # ---- validate ----
    SUPPORTED = get_supported_tasks()
    assert TASK in SUPPORTED, f"Unsupported task: {TASK}. Choose from {SUPPORTED}"

    # ---- setup ----
    setup_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- load models ----
    print("Loading diffusion model...")
    model, diffusion = load_diffusion_model(MODEL_PATH)
    if USE_FP16:
        model = model.half()
        print("  UNet converted to fp16")

    restorer = None
    embedding = None
    if "restoration" in TASK:
        print("Loading restorer...")
        restorer = load_restorer(RESTORER_PATH)
    if TASK == "ref_restoration":
        print("Loading ArcFace embedding...")
        embedding = load_arcface(ARCFACE_PATH)

    # ---- weights dict ----
    weights = dict(
        lightness_weight=LIGHTNESS_WEIGHT,
        color_weight=COLOR_WEIGHT,
        unmasked_weight=UNMASKED_WEIGHT,
        ss_weight=SS_WEIGHT,
        edge_weight=EDGE_WEIGHT,
        grad_momentum=GRAD_MOMENTUM,
        ref_weight=REF_WEIGHT,
        op_lightness_weight=OP_LIGHTNESS_WEIGHT,
        op_color_weight=OP_COLOR_WEIGHT,
        class_cond=get_model_defaults()["class_cond"],
    )

    # ---- respace diffusion if needed ----
    if TIMESTEP_RESPACING:
        from guided_diffusion.respace import SpacedDiffusion, space_timesteps
        from guided_diffusion import gaussian_diffusion as gd
        betas = gd.get_named_beta_schedule("linear", DIFFUSION_STEPS)
        use_timesteps = space_timesteps(DIFFUSION_STEPS, TIMESTEP_RESPACING)
        diffusion = SpacedDiffusion(
            use_timesteps=use_timesteps,
            betas=betas,
            model_mean_type=gd.ModelMeanType.START_X,
            model_var_type=gd.ModelVarType.LEARNED_RANGE,
            loss_type=gd.LossType.MSE,
            rescale_timesteps=False,
        )

    # ---- create guidance ----
    guidance = PartialGuidance(restorer, embedding, **weights)

    # ---- seed ----
    th.manual_seed(SEED)
    np.random.seed(SEED)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(SEED)

    # ---- print config ----
    print("=" * 58)
    print(f"  Task:              {TASK}")
    print(f"  Guidance scale:    {GUIDANCE_SCALE}")
    print(f"  N:                 {N}  (range [{S_START}T, {S_END}T])")
    print(f"  Steps:             {'full 1000' if not TIMESTEP_RESPACING else TIMESTEP_RESPACING}")
    print(f"  DDIM:              {USE_DDIM}")
    print(f"  Seed:              {SEED}")
    print(f"  Input:             {IN_DIR}")
    print(f"  GT:                {GT_DIR if os.path.isdir(GT_DIR) else 'N/A'}")
    print("=" * 58)

    # ---- collect images ----
    lr_images = sorted(os.listdir(IN_DIR))
    if MAX_IMAGES:
        lr_images = lr_images[:MAX_IMAGES]
    if not lr_images:
        print(f"ERROR: No images in {IN_DIR}")
        return

    mask_images = None
    if TASK == "ref_restoration":
        assert REF_DIR, "REF_DIR required for ref_restoration"
        assert len(os.listdir(REF_DIR)) == len(lr_images), "ref / LQ image count mismatch"
    if TASK in ("inpainting", "old_photo_restoration") and MASK_DIR:
        mask_images = set(os.listdir(MASK_DIR))

    # ---- run inference ----
    peak_vram_mb, per_image_seconds, per_image_metrics, speed_stats = run_experiment(
        diffusion, guidance, lr_images, OUT_DIR, TASK, GUIDANCE_SCALE,
        N, S_START, S_END, DIFFUSION_STEPS, SEED,
        IN_DIR, REF_DIR, MASK_DIR, mask_images,
        embedding=embedding, gt_dir=GT_DIR,
    )

    # ---- compute metrics ----
    avg_loss = guidance.avg_loss()
    quality_score_old, avg_sharpness = compute_quality_score(avg_loss, per_image_metrics)
    quality_score_v2 = compute_quality_score_v2(per_image_metrics)
    avg_psnr, avg_ssim_gt, avg_niqe = compute_paper_quality_score(per_image_metrics)
    agg_metrics = aggregate_metrics(per_image_metrics)
    total_seconds = time.time() - t_start

    # ---- print results (before recording to ensure output even on failure) ----
    print_results(quality_score_v2, quality_score_old, avg_loss, len(lr_images),
                  total_seconds, peak_vram_mb, guidance.loss_breakdown,
                  agg_metrics, per_image_metrics, speed_stats,
                  avg_psnr=avg_psnr, avg_ssim_gt=avg_ssim_gt, avg_niqe=avg_niqe)

    # ---- record if part of experiment loop ----
    if RUN_TAG:
        try:
            commit_hash = os.popen("git rev-parse --short HEAD").read().strip()
            exp_num = len([d for d in os.listdir(RUNS_DIR)
                           if os.path.isdir(os.path.join(RUNS_DIR, d))]) + 1
            run_dir = os.path.join(RUNS_DIR, f"exp_{exp_num:03d}")
            record_experiment(
                run_dir, commit_hash, quality_score_v2, quality_score_old,
                avg_loss, len(lr_images), total_seconds, peak_vram_mb,
                per_image_seconds, guidance.loss_breakdown, agg_metrics,
                per_image_metrics,
                TASK, GUIDANCE_SCALE, weights, N, S_START, S_END,
                TIMESTEP_RESPACING, USE_DDIM, SEED,
                avg_psnr=avg_psnr, avg_ssim_gt=avg_ssim_gt, avg_niqe=avg_niqe,
            )
        except Exception as e:
            print(f"[record_experiment] WARNING: recording failed: {e}")


if __name__ == "__main__":
    main()
