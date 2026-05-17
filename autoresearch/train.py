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
  4. Extract results: grep "^quality_score:" run.log
  5. Log to results.tsv
  6. Keep or discard based on quality_score (lower = better)

PRIMARY METRIC:  quality_score = avg_guidance_loss / avg_sharpness
  - Lower loss   → better constraint satisfaction
  - Higher sharp → less blur, more detail
  - quality_score balances both → the agent minimizes this
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
EDGE_WEIGHT = 0.02       # edge preservation loss weight
GRAD_MOMENTUM = 0.9      # momentum for gradient smoothing across timesteps
REF_WEIGHT = 25.0
OP_LIGHTNESS_WEIGHT = 1.0
OP_COLOR_WEIGHT = 0.5

# --- multi-step guidance ---
N = 1                # gradient steps per timestep (>1 = stronger guidance)
S_START = 1.0        # start fraction of T (e.g. 1.0 = from t=T)
S_END = 0.7          # end fraction of T (e.g. 0.7 = until 0.7T)

# --- sampling ---
TIMESTEP_RESPACING = "ddpm200"   # "" = full 1000 steps, "ddim25" = 25 DDIM steps
USE_DDIM = False
CLIP_DENOISED = True
BATCH_SIZE = 1
IMAGE_SIZE = 512
DIFFUSION_STEPS = 1000    # total diffusion steps of the pre-trained model

# --- I/O paths ---
IN_DIR = "testdata/cropped_faces"
OUT_DIR = "results/experiment"
REF_DIR = None
MASK_DIR = None
MODEL_PATH = "models/iddpm_ffhq512_ema500000.pth"
RESTORER_PATH = "models/restorer/rrdb_iter_100000.pth"
ARCFACE_PATH = "models/ms1mv3_arcface_r50_fp16.pth"

# --- recording ---
RUN_TAG = "may17_v2"  # set by the experiment loop; leave empty for manual runs
RUNS_DIR = "runs"     # per-experiment detailed logs stored here

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

    def reset(self):
        self.losses = []
        self.loss_breakdown = {}
        self.prev_gradient = None

    def avg_loss(self):
        if not self.losses:
            return float("inf")
        return sum(self.losses) / len(self.losses)

    def __call__(self, x, t, y=None, pred_xstart=None, target=None,
                 ref=None, mask=None, task="restoration", scale=0,
                 N=1, s_start=1, s_end=0.7):
        assert y is not None
        fake_g_output = None

        with th.enable_grad():
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
                if target is None:
                    fake_g_output = self.restorer(x, y_t=y, t=t).clamp(-1, 1)
                    fake_g_output = fake_g_output.detach().requires_grad_(True).cuda()
                else:
                    fake_g_output = target.detach().requires_grad_(True).cuda()
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

            # ── Timestep-dependent guidance scheduling ──
            # Linear decay: strong guidance at high t (structure), weaker at low t (details)
            t_frac = t[0].item() / DIFFUSION_STEPS
            schedule = 0.3 + 0.7 * t_frac  # ranges from ~1.0 (t=T) to ~0.3 (t=0)
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
            return gradient, fake_g_output.detach()
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
    Returns (peak_vram_mb, per_image_seconds, per_image_metrics).
    per_image_metrics: list of dicts with 'sharpness', 'colorfulness', 'identity_sim'.
    """
    peak_vram_mb = 0
    per_image_seconds = []
    per_image_metrics = []

    for idx, img_name in enumerate(images):
        guidance.reset()
        t_img = time.time()
        print(f"[{idx + 1}/{len(images)}] {img_name}")

        model_kwargs = build_model_kwargs(
            img_name, in_dir, task, guidance_scale, n, s_start, s_end,
            diffusion_steps, ref_dir, mask_dir, mask_images
        )

        sample_fn = diffusion.ddim_sample_loop if USE_DDIM else diffusion.p_sample_loop
        sample = sample_fn(
            model_fn,
            (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
            clip_denoised=CLIP_DENOISED,
            model_kwargs=model_kwargs,
            cond_fn=guidance,
            device=device(),
            seed=seed,
        )

        save_image(sample, os.path.join(out_dir, img_name))
        elapsed = time.time() - t_img
        per_image_seconds.append(elapsed)

        # ---- per-image quality metrics ----
        ref_tensor = model_kwargs.get("ref", None)
        metrics = compute_image_metrics(
            sample, task=task, embedding=embedding,
            ref_tensor=ref_tensor, input_tensor=model_kwargs.get("y", None),
        )
        per_image_metrics.append(metrics)

        if th.cuda.is_available():
            vram = th.cuda.max_memory_allocated() / (1024 * 1024)
            peak_vram_mb = max(peak_vram_mb, vram)
            th.cuda.reset_peak_memory_stats()

    return peak_vram_mb, per_image_seconds, per_image_metrics


# ═══════════════════════════════════════════════════════════════════════════
# QUALITY SCORE  —  Composite metric the agent optimizes
# ═══════════════════════════════════════════════════════════════════════════

def compute_quality_score(avg_loss, per_image_metrics):
    """
    Composite quality score — the PRIMARY metric to minimize.

    quality_score = avg_guidance_loss / avg_sharpness

    Lower is better:
      - Lower guidance loss  → better constraint satisfaction
      - Higher sharpness      → less blur, more detail
      - Loss / sharpness      → balances both in one number

    For ref_restoration, identity_sim is also reported separately.
    """
    sharpness_vals = [m["sharpness"] for m in per_image_metrics]
    if not sharpness_vals:
        return float("inf"), 0.0
    avg_sharpness = sum(sharpness_vals) / len(sharpness_vals)
    quality_score = avg_loss / (avg_sharpness + 1e-8)
    return quality_score, avg_sharpness


# ═══════════════════════════════════════════════════════════════════════════
# RECORDING / LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def record_experiment(run_dir, commit_hash, quality_score, avg_loss,
                      num_images, total_seconds, peak_vram_mb,
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
        quality_str += f"  {k:25s}: {v:.3f}\n"

    per_img_str = ""
    for i, m in enumerate(per_image_metrics):
        parts = ", ".join(f"{k}={v:.1f}" for k, v in sorted(m.items()))
        per_img_str += f"  img_{i:02d}: {parts}\n"

    summary = f"""PGDiff Experiment Summary
{'=' * 60}
commit:             {commit_hash}
started:            {time.strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 60}
quality_score:      {quality_score:.3f}   <-- PRIMARY METRIC (lower = better)
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


def print_results(quality_score, avg_loss, num_images, total_seconds,
                  peak_vram_mb, loss_breakdown, agg_metrics, per_image_metrics):
    """Print parseable results for the agent to grep."""
    print()
    print("---")
    print(f"quality_score:       {quality_score:.3f}   <-- PRIMARY (lower = better)")
    print(f"avg_guidance_loss:   {avg_loss:.3f}")
    print(f"num_images:          {num_images}")
    print(f"loss_per_image:      {avg_loss / num_images:.3f}")
    print(f"total_seconds:       {total_seconds:.1f}")
    print(f"peak_vram_mb:        {peak_vram_mb:.1f}")
    print(f"task:                {TASK}")
    print(f"guidance_scale:      {GUIDANCE_SCALE:.3f}")
    print(f"N:                   {N}")
    print(f"timestep_respacing:  {'ddpm1000' if not TIMESTEP_RESPACING else TIMESTEP_RESPACING}")
    print(f"use_ddim:            {USE_DDIM}")
    print(f"seed:                {SEED}")
    for k, v in sorted(agg_metrics.items()):
        print(f"{k}:             {v:.3f}")
    for k, v in sorted(loss_breakdown.items()):
        if v:
            print(f"loss_{k}:          {sum(v)/len(v):.3f}")


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
    peak_vram_mb, per_image_seconds, per_image_metrics = run_experiment(
        diffusion, guidance, lr_images, OUT_DIR, TASK, GUIDANCE_SCALE,
        N, S_START, S_END, DIFFUSION_STEPS, SEED,
        IN_DIR, REF_DIR, MASK_DIR, mask_images,
        embedding=embedding,
    )

    # ---- compute metrics ----
    avg_loss = guidance.avg_loss()
    quality_score, avg_sharpness = compute_quality_score(avg_loss, per_image_metrics)
    agg_metrics = aggregate_metrics(per_image_metrics)
    total_seconds = time.time() - t_start

    # ---- record if part of experiment loop ----
    if RUN_TAG:
        commit_hash = os.popen("git rev-parse --short HEAD").read().strip()
        exp_num = len([d for d in os.listdir(RUNS_DIR)
                       if os.path.isdir(os.path.join(RUNS_DIR, d))]) + 1
        run_dir = os.path.join(RUNS_DIR, f"exp_{exp_num:03d}")
        record_experiment(
            run_dir, commit_hash, quality_score, avg_loss,
            len(lr_images), total_seconds, peak_vram_mb,
            per_image_seconds, guidance.loss_breakdown, agg_metrics,
            per_image_metrics,
            TASK, GUIDANCE_SCALE, weights, N, S_START, S_END,
            TIMESTEP_RESPACING, USE_DDIM, SEED,
        )

    # ---- print results ----
    print_results(quality_score, avg_loss, len(lr_images), total_seconds,
                  peak_vram_mb, guidance.loss_breakdown,
                  agg_metrics, per_image_metrics)


if __name__ == "__main__":
    main()
