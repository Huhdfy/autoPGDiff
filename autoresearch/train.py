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
import cv2
import numpy as np
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
    compute_sharpness, compute_image_metrics, aggregate_metrics,
    get_supported_tasks, get_model_defaults,
)

# ═══════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS  —  Edit freely
# ═══════════════════════════════════════════════════════════════════════════

TASK = "restoration"
GUIDANCE_SCALE = 0.1
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
OP_COLOR_WEIGHT = 0.5
RESIDUAL_BLEND = 0.5        # output = (sample + input * blend) / (1 + blend) — weighted average

# --- multi-step guidance ---
N = 2                # gradient steps per timestep (>1 = stronger guidance)
S_START = 1.0        # start fraction of T (e.g. 1.0 = from t=T)
S_END = 0.5          # end fraction of T (e.g. 0.7 = until 0.7T)

# --- sampling ---
TIMESTEP_RESPACING = ""   # full 1000 steps
USE_DDIM = False
CLIP_DENOISED = True
BATCH_SIZE = 1
IMAGE_SIZE = 512
DIFFUSION_STEPS = 1000    # total diffusion steps of the pre-trained model

# --- I/O paths ---
IN_DIR = "../testdata/cropped_faces"
# Use only first N images for faster experiments
MAX_IMAGES = 1
# Speed optimization flags
BLOCK_UNET_GRAD = True   # True: restorer outside enable_grad
USE_DPMSOLVER = True     # True: use DPM-Solver-2 (higher-order ODE, fewer steps)
DPM_SOLVER_STEPS = 35     # number of DPM-Solver steps (if USE_DPMSOLVER=True)
OUT_DIR = "../results/experiment"
REF_DIR = None
MASK_DIR = None
MODEL_PATH = "../models/iddpm_ffhq512_ema500000.pth"
RESTORER_PATH = "../models/restorer/rrdb_iter_100000.pth"
ARCFACE_PATH = "../models/ms1mv3_arcface_r50_fp16.pth"

# --- recording ---
RUN_TAG = "accelerate_v1"  # 推理加速实验第1轮
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
        fake_g_output = None
        self.guidance_calls += 1

        # ── restorer forward (position depends on BLOCK_UNET_GRAD) ──
        if "restoration" in task:
            if target is not None:
                fake_g_output = target.cuda()
            elif BLOCK_UNET_GRAD:
                # Outside enable_grad: no computation graph built → saves VRAM
                self.restorer_calls += 1
                with th.no_grad():
                    fake_g_output = self.restorer(x, y_t=y, t=t).clamp(-1, 1).cuda()

        with th.enable_grad():
            pred_xstart_in = pred_xstart.detach().requires_grad_(True)
            total_loss = th.tensor(0.0, device=x.device)

            # ── restorer for baseline (inside enable_grad = builds graph, more VRAM) ──
            if "restoration" in task and not BLOCK_UNET_GRAD and fake_g_output is None:
                self.restorer_calls += 1
                fake_g_output = self.restorer(x, y_t=y, t=t).clamp(-1, 1).cuda()
            pred_xstart_in = pred_xstart.detach().requires_grad_(True)
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
                loss_s = F.smooth_l1_loss(
                    fake_g_output, pred_xstart_in, reduction="sum", beta=1.0
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
                pred_xstart_in = pred_xstart.detach().requires_grad_(True)
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
            schedule = 0.3 + 0.7 * t_frac
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
        if i < len(step_t) - 2:
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
                   embedding=None):
    """
    Run inference on all images.
    Returns (peak_vram_mb, per_image_seconds, per_image_metrics, speed_stats).
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

        if USE_DPMSOLVER:
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
# RECORDING / LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def record_experiment(run_dir, commit_hash, quality_score_v2, quality_score_old,
                      avg_loss, num_images, total_seconds, peak_vram_mb,
                      per_image_seconds, loss_breakdown, agg_metrics,
                      per_image_metrics, task, guidance_scale,
                      weights, n, s_start, s_end,
                      timestep_respacing, use_ddim, seed):
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
                  per_image_metrics, speed_stats=None):
    """Print parseable results for the agent to grep."""
    print()
    print("---")
    print(f"quality_score_v2:    {quality_score_v2:.4f}   <-- PRIMARY (higher = better)")
    print(f"quality_score_old:    {quality_score_old:.3f}   (deprecated, lower = better)")
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
        embedding=embedding,
    )

    # ---- compute metrics ----
    avg_loss = guidance.avg_loss()
    quality_score_old, avg_sharpness = compute_quality_score(avg_loss, per_image_metrics)
    quality_score_v2 = compute_quality_score_v2(per_image_metrics)
    agg_metrics = aggregate_metrics(per_image_metrics)
    total_seconds = time.time() - t_start

    # ---- print results (before recording to ensure output even on failure) ----
    print_results(quality_score_v2, quality_score_old, avg_loss, len(lr_images),
                  total_seconds, peak_vram_mb, guidance.loss_breakdown,
                  agg_metrics, per_image_metrics, speed_stats)

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
            )
        except Exception as e:
            print(f"[record_experiment] WARNING: recording failed: {e}")


if __name__ == "__main__":
    main()
