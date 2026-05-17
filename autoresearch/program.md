# PGDiff autoresearch

Autonomous AI-driven research on the PGDiff face restoration pipeline.
The agent iterates on guidance strategies, loss formulations, task compositions, and
sampling approaches. **Innovation is required — don't just tune hyperparameters.**

## Evaluation metric

**Primary metric: `quality_score`** (lower = better)

```
quality_score = avg_guidance_loss / avg_sharpness
```

| Component | Meaning | Direction |
|-----------|---------|-----------|
| `avg_guidance_loss` | How well the output satisfies task constraints (MSE to restorer, color stats, etc.) | Lower = better |
| `avg_sharpness` | Laplacian variance of output images — proxy for detail preservation | Higher = better |
| `quality_score` | Composite: balances constraint satisfaction with sharpness | **Lower = better** |

Additional metrics also reported:
- `colorfulness` — for colorization/old-photo tasks (higher = more vivid)
- `identity_sim` — for ref_restoration (ArcFace cosine similarity, 0-1, higher = same person)
- Per-term loss breakdown (smooth_semantics, color_lightness, etc.)

**Why this works**: Pure guidance loss can be gamed (extreme scales produce artifacts that still have low loss). Sharpness penalizes blurry/artifact-ridden outputs. The composite score rewards both fidelity to constraints AND natural image quality.

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may17`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files** for full context:
   - `README.md` — PGDiff project overview.
   - `prepare.py` — immutable: model loading, I/O, AdaIN, quality metrics. Do NOT modify.
   - `train.py` — the file you modify. **Everything** in it is fair game.
   - `inference_pgdiff.py` — original PGDiff inference script (reference).
4. **Verify models and data exist**. If not, tell the human.
5. **Initialize results.tsv**: Create with header row. Baseline after first run.
6. **Confirm and go**.

## What you CAN modify (in train.py) — aim for INNOVATION

**Hyperparameters** (for warmup — but don't stop here):
- `TASK`, `GUIDANCE_SCALE`, all task weights
- `N`, `S_START`, `S_END`, `TIMESTEP_RESPACING`, `USE_DDIM`

**PartialGuidance class** — Where the real innovation happens:
- **Add new loss terms**: perceptual loss (LPIPS), adversarial loss, edge preservation (Sobel), total variation regularization, Laplacian pyramid loss, frequency-domain losses, contrastive losses
- **Change loss formulations**: L1 vs MSE, Huber loss, cosine vs MSE for identity, learnable loss weights, loss scheduling (different weights at different timesteps)
- **Add entirely new tasks**: face super-resolution, deblurring, denoising, face editing, expression transfer, makeup transfer, style mixing
- **Change the guidance strategy**: classifier-free guidance style instead of gradient-based, alternating guidance, adaptive guidance scale per timestep, guidance annealing
- **Modify how gradients are applied**: per-channel scaling, gradient clipping, momentum in gradient steps, learned gradient correction
- **Change the restorer interaction**: feed different inputs to restorer, ensemble multiple restorers, skip restorer at certain timesteps

**model_fn** — How the diffusion model is called:
- Multi-model ensembles, conditional injection strategies

**Sampling pipeline** (`run_experiment`):
- Progressive refinement, iterative sampling, different noise schedules
- Post-processing chains (sharpening, color correction)

**Quality metrics** (add your own):
- Add new metrics to `compute_quality_score` or `compute_image_metrics`
- Change the composite formula to better capture your goals

## What you CANNOT do

- Modify `prepare.py`
- Install packages beyond `requirements.txt`
- Modify files outside `autoresearch/`

## Every change MUST be recorded

### 1. Git commit message (detailed, required)
```
<concise summary>

- What: [specific code change]
- Why: [hypothesis — why this improves quality_score]
- Expected: [what you expect to happen to quality_score and sub-metrics]
```

### 2. Per-run log (automatic)
When `RUN_TAG` is set, each experiment writes `runs/exp_NNN/summary.txt` with:
- Full hyperparameter snapshot, loss breakdown per term, per-image quality metrics

### 3. results.tsv (tab-separated, 10 columns)
```
commit	quality_score	avg_loss	avg_sharpness	loss_per_image	task	status	description	key_params	loss_breakdown
```

## The experiment loop

LOOP FOREVER:

1. **Survey state**: current git commit. Read last few lines of `results.tsv`. Check `runs/` for detailed logs. Look for patterns — what worked, what didn't.
2. **Form hypothesis**: what change might reduce quality_score? Why?
3. **Implement**: edit `train.py`.
4. **Commit**: `git commit` with detailed message.
5. **Run**: `python train.py > run.log 2>&1`
6. **Extract**: `grep "^quality_score:\|^num_images:" run.log`
7. **Handle crashes**: if grep empty → `tail -50 run.log` → fix or log crash.
8. **Log to results.tsv**: append the 10-column row. NEVER commit results.tsv.
9. **Decide**:
   - **quality_score improved** (lower) → keep commit, advance branch
   - **Same or worse** → `git reset --soft HEAD~1`
   - **Crash** → `git reset --hard HEAD~1`

## Innovation ideas — USE THESE as starting points

### Architecture-level changes
- Replace the restorer network with a different backbone (swinIR, NAFNet)
- Add a learned guidance network that predicts optimal gradient direction
- Implement classifier-free guidance by training a conditional score estimator
- Add attention-based feature matching between input and output

### Loss function innovation
- Add LPIPS perceptual loss between restorer output and pred_xstart
- Add facial landmark consistency loss (detect landmarks, penalize drift)
- Add GAN-style adversarial loss (train a small discriminator on the fly)
- Add frequency-domain loss (FFT-based, penalize high-freq artifacts)
- Add self-supervised consistency loss (different seeds → same identity)

### Guidance strategy innovation
- Schedule guidance scale s(t) = s0 * f(t/T) with different decay functions
- Apply guidance only on early/middle/late timesteps and compare
- Use different guidance scales per channel (R, G, B)
- Add momentum to gradient steps across timesteps
- Implement trust-region guidance (clip gradient if step too large)

### Task innovation
- Define a "best of both" task: restoration + colorization simultaneously
- Define a "face hallucination" task: 64×64 → 512×512 super-resolution
- Define a "de-aging" task by guiding toward a younger reference embedding

### Sampling innovation
- Try stochastic DDIM (eta > 0) vs deterministic
- Implement restarted sampling (sample N times, pick best by sharpness)
- Try different noise schedules (cosine instead of linear)
- Implement coarse-to-fine: first sample at low res, then refine at high res

## Timeouts and crashes
- Max ~15 min per experiment. Kill if >20 min.
- Trivial bugs → fix and re-run. Broken ideas → log crash, move on.
- NaN → git reset --hard, log crash.

## NEVER STOP
Run indefinitely. ~12 experiments/hour, ~100 per sleep cycle.
If stuck, try BIGGER changes — architectural innovations, not just parameter sweeps.
