"""
Immutable infrastructure for PGDiff autoresearch.
Provides model loading, image I/O, and utility functions only.

The agent does NOT modify this file. All experiment logic lives in train.py.

Usage:
    from prepare import setup_device, load_diffusion_model, load_restorer, load_arcface
    from prepare import load_image, load_mask, save_image
    from prepare import avg_grayscale, adaptive_instance_normalization, calc_mean_std
"""

import os
import sys
import cv2
import numpy as np
import torch as th
import torch.nn.functional as F
from collections import OrderedDict

_srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _srcdir not in sys.path:
    sys.path.insert(0, _srcdir)

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    SUPPORTED_TASKS,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_restorer,
    create_arcface_embedding,
)

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def setup_device():
    dist_util.setup_dist()

def device():
    return dist_util.dev()

# ---------------------------------------------------------------------------
# Model loading (immutable — models are frozen during experiments)
# ---------------------------------------------------------------------------

def load_diffusion_model(model_path):
    """Load the pre-trained face diffusion UNet. Returns (model, diffusion)."""
    defaults = model_and_diffusion_defaults()
    model, diffusion = create_model_and_diffusion(**defaults)
    state_dict = dist_util.load_state_dict(model_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_state_dict[key[7:]] = value          # strip "module." prefix
    model.load_state_dict(new_state_dict)
    model.to(dist_util.dev())
    model.eval()
    return model, diffusion


def load_restorer(restorer_path):
    """Load the RRDBNet restorer for smooth semantics prediction. Returns restorer."""
    restorer = create_restorer()
    ckpt = dist_util.load_state_dict(restorer_path, map_location="cpu")
    restorer.load_state_dict(ckpt["state_dict"], strict=False)
    restorer.to(dist_util.dev())
    restorer.eval()
    return restorer


def load_arcface(arcface_path):
    """Load the ArcFace (iresnet50) for identity feature extraction. Returns embedding."""
    embedding = create_arcface_embedding()
    embedding.load_state_dict(dist_util.load_state_dict(arcface_path))
    embedding.to(dist_util.dev())
    embedding.eval()
    return embedding

# ---------------------------------------------------------------------------
# Image I/O (immutable preprocessing/postprocessing)
# ---------------------------------------------------------------------------

def load_image(path, size=512):
    """Load image from path, resize to (size,size), convert BGR->RGB, normalize to [-1,1].
    Returns tensor of shape (1, 3, size, size)."""
    img = cv2.resize(
        cv2.imread(path), (size, size)
    ).astype(np.float32)[:, :, [2, 1, 0]] / 127.5 - 1
    return th.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def load_mask(path, size=512):
    """Load binary mask from path, resize to (size,size), normalize to [0,1].
    Returns tensor of shape (1, 3, size, size)."""
    img = cv2.resize(
        cv2.imread(path), (size, size)
    ).astype(np.float32) / 255.0
    return th.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def save_image(tensor, path):
    """Save tensor (1,3,H,W) in [-1,1] to image file (BGR PNG)."""
    img = ((tensor + 1) * 127.5).clamp(0, 255).to(th.uint8)
    img = img.permute(0, 2, 3, 1).cpu().numpy()
    cv2.imwrite(path, img[0][..., [2, 1, 0]])

# ---------------------------------------------------------------------------
# Utility functions for partial guidance (immutable helpers)
# ---------------------------------------------------------------------------

def avg_grayscale(img):
    """Compute per-pixel average across RGB channels, expanded back to 3 channels."""
    rgb_mean = th.mean(img, [1], keepdim=True).expand(-1, 3, -1, -1)
    return rgb_mean


def calc_mean_std(feat, eps=1e-5):
    """Compute channel-wise mean and std of a feature map."""
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat=None):
    """
    Apply AdaIN using pre-computed CelebA-HQ color statistics.
    If style_feat is None, uses style 0 (default average statistics).
    The agent can add new styles by editing the style tensors below.
    """
    # style 0 (default): celebA-HQ avg
    style_mean = th.tensor([0.03202754, -0.16308397, -0.26475719]).reshape(1, 3, 1, 1).cuda()
    style_std  = th.tensor([0.53549316,  0.47539538,  0.46814889]).reshape(1, 3, 1, 1).cuda()

    if style_feat is not None:
        style_mean, style_std = calc_mean_std(style_feat)

    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


# ---------------------------------------------------------------------------
# Image quality metrics (immutable — reliable no-reference evaluation)
# ---------------------------------------------------------------------------

def compute_sharpness(tensor):
    """
    Laplacian variance — standard no-reference sharpness metric.
    Higher = sharper image (fewer artifacts from over-smoothing).
    tensor: (1, 3, H, W) in [-1, 1].
    """
    img = ((tensor + 1) * 127.5).clamp(0, 255).to(th.uint8)
    img_np = img[0].permute(1, 2, 0).cpu().numpy()          # (H, W, 3) RGB
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_colorfulness(tensor):
    """
    Colorfulness metric (Hasler & Suesstrunk 2003).
    Higher = more colorful. Good for colorization/old-photo tasks.
    tensor: (1, 3, H, W) in [-1, 1].
    """
    img = ((tensor + 1) * 127.5).clamp(0, 255).to(th.uint8)
    img_np = img[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (H, W, 3) RGB
    R, G, B = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    return np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)


def compute_identity_similarity(embedding, img1, img2):
    """
    ArcFace cosine similarity between two face images.
    1.0 = same identity, 0.0 = different.
    img1, img2: (1, 3, H, W) tensors in [-1, 1].
    """
    with th.no_grad():
        e1 = embedding(F.interpolate(img1, (112, 112), mode="bilinear", antialias=True))
        e2 = embedding(F.interpolate(img2, (112, 112), mode="bilinear", antialias=True))
        sim = F.cosine_similarity(e1, e2, dim=1).item()
    return sim


def compute_image_metrics(output_tensor, task="restoration",
                          embedding=None, ref_tensor=None, input_tensor=None):
    """
    Compute quality metrics for a single output image.
    Returns a dict of metric_name -> value.

    Metrics computed:
      - sharpness:       always (no-reference)
      - colorfulness:    for colorization / old_photo tasks
      - identity_sim:    for ref_restoration (needs embedding + ref_tensor)
    """
    metrics = {"sharpness": compute_sharpness(output_tensor)}

    if task in ("colorization", "old_photo_restoration"):
        metrics["colorfulness"] = compute_colorfulness(output_tensor)

    if task == "ref_restoration" and embedding is not None and ref_tensor is not None:
        metrics["identity_sim"] = compute_identity_similarity(
            embedding, output_tensor, ref_tensor
        )

    return metrics


def aggregate_metrics(all_per_image_metrics):
    """
    Aggregate per-image metric dicts into averages.
    Input: [{"sharpness": 100, "colorfulness": 45}, {"sharpness": 120, ...}, ...]
    Returns: {"sharpness_avg": 110.0, "sharpness_min": 100.0, ...}
    """
    if not all_per_image_metrics:
        return {}
    keys = set()
    for m in all_per_image_metrics:
        keys.update(m.keys())
    agg = {}
    for k in sorted(keys):
        values = [m[k] for m in all_per_image_metrics if k in m]
        if values:
            agg[f"{k}_avg"] = sum(values) / len(values)
            agg[f"{k}_min"] = min(values)
            agg[f"{k}_max"] = max(values)
    return agg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_supported_tasks():
    return SUPPORTED_TASKS


def get_model_defaults():
    return model_and_diffusion_defaults()
