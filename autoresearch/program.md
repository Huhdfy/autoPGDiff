# PGDiff autoresearch

Autonomous AI-driven research on the PGDiff face restoration pipeline. The agent
iterates on guidance strategies, loss formulations, task compositions, and
sampling approaches — not just hyperparameters.

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may17`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files** for full context:
   - `README.md` — PGDiff project overview.
   - `prepare.py` — immutable infrastructure: model loading, image I/O, AdaIN helpers. Do NOT modify.
   - `train.py` — the file you modify. Contains: hyperparameters, PartialGuidance class, model_fn, sampling pipeline, recording system.
   - `inference_pgdiff.py` — original PGDiff inference script (reference).
4. **Verify models and data exist**. If not, tell the human.
5. **Initialize results.tsv**: Create with header row (see format below). Baseline recorded after first run.
6. **Confirm and go**.

Once confirmed, begin the experiment loop.

## What you CAN modify (in train.py)

**Hyperparameters** (top of file):
- `TASK`, `GUIDANCE_SCALE`, all task weights
- `N`, `S_START`, `S_END` — multi-step guidance
- `TIMESTEP_RESPACING`, `USE_DDIM` — sampling strategy
- `DIFFUSION_STEPS`, `SEED`, paths

**PartialGuidance class** — This is the CORE of PGDiff. Modify freely:
- Add new loss terms (perceptual loss, adversarial loss, TV regularization, etc.)
- Change loss formulations (L1 vs MSE, cosine vs MSE for identity, etc.)
- Add entirely new tasks (e.g., face super-resolution, deblurring, style transfer)
- Modify gradient scaling (e.g., per-channel coefficient, adaptive scaling)
- Add task routing logic (composite tasks that combine multiple losses)
- Change how `fake_g_output` (restorer target) is computed
- Experiment with different normalization or feature spaces

**model_fn** — How the diffusion model is called:
- Change conditioning strategy (pass features instead of raw images)
- Add ensemble strategies (multiple model calls)
- Modify input preprocessing

**Sampling pipeline** (`run_experiment`):
- Change the sampling loop (e.g., progressive refinement)
- Add post-processing steps
- Add additional metrics computation
- Modify how images are batched or ordered

**Recording system** (`record_experiment`, `print_results`):
- Add new tracked metrics
- Change output format

## What you CANNOT do

- Modify `prepare.py`. It is read-only.
- Install new packages beyond `requirements.txt`.
- Modify files outside `autoresearch/`.
- Modify the base diffusion model weights.

## The goal

**Minimize `avg_guidance_loss`** — the average partial guidance loss across all
timesteps and images. Lower = output better satisfies task constraints.

The loss is task-specific and directly measures quality:
- Restoration: distance to smooth semantics (restorer output)
- Colorization: lightness fidelity + color naturalness
- Inpainting: unmasked region preservation
- Ref-restoration: smooth semantics + identity embedding proximity
- Old-photo: composite (lightness + color on valid pixels)

## Every change MUST be recorded

This is critical. Every experiment leaves a complete paper trail:

### 1. Git commit message (REQUIRED, detailed)
```
<concise summary of what was changed and why>

- What: [specific code change, not vague]
- Why: [hypothesis — why this should improve loss]
- Expected: [what you expect to happen to avg_guidance_loss]
```

Example GOOD commit:
```
Add perceptual LPIPS loss term to restoration guidance

- What: Added LPIPS-based perceptual loss between restorer output
  and pred_xstart, weighted at 0.1 relative to MSE
- Why: MSE alone favors blurry outputs; perceptual loss should
  preserve texture details better
- Expected: avg_guidance_loss may stay similar but output quality
  should improve (if we had visual eval); loss breakdown will
  show new "perceptual" term
```

Example BAD commit: `try something` or `tune weights`

### 2. Per-run log (automatic)
When `RUN_TAG` is set in train.py, each experiment writes a detailed log to
`runs/exp_NNN/summary.txt` containing:
- Full hyperparameter snapshot
- Loss breakdown per term
- Timing and VRAM stats
- Git commit hash

### 3. results.tsv entry (REQUIRED)
Tab-separated, 8 columns:

```
commit	avg_loss	loss_per_image	task	status	description	key_params	loss_breakdown
```

| Column | Description |
|--------|-------------|
| `commit` | 7-char git hash |
| `avg_loss` | Average guidance loss (0.0 for crashes) |
| `loss_per_image` | avg_loss / num_images (0.0 for crashes) |
| `task` | Task name |
| `status` | `keep` / `discard` / `crash` |
| `description` | Brief description of what was tried |
| `key_params` | Key hyperparams snapshot (e.g. "s=0.15,ss=1.0,N=3") |
| `loss_breakdown` | Per-term losses (e.g. "smooth_semantics:1234.5") |

Example:
```
a1b2c3d	1245.6	103.8	restoration	keep	baseline (s=0.1, ss=1.0, N=1)	s=0.1,ss=1.0,N=1	smooth_semantics:1245.6
b2c3d4e	1102.3	91.9	restoration	keep	add LPIPS perceptual loss	w=0.1	s=0.1,ss=1.0,lpips=0.1	smooth_semantics:892.1,lpips:210.2
```

## The experiment loop

LOOP FOREVER:

1. **Survey state**: current git branch/commit. Read the last few lines of `results.tsv` to see what's been tried recently. Check `runs/` for detailed logs of past experiments.
2. **Form hypothesis**: what change might reduce avg_guidance_loss? Why?
3. **Implement**: edit `train.py` (hyperparameters, PartialGuidance, model_fn, or pipeline).
4. **Commit**: `git commit` with a DETAILED message (what, why, expected).
5. **Run**: `python train.py > run.log 2>&1` (no tee, no flooding).
6. **Extract**: `grep "^avg_guidance_loss:" run.log`
7. **Handle crashes**: If grep output empty → `tail -50 run.log` → fix trivial bugs or log as "crash".
8. **Log to results.tsv**: append the 8-column row. Do NOT commit results.tsv.
9. **Decide**:
   - **Improved** (lower avg_loss) → advance branch (keep commit)
   - **Same or worse** → `git reset --soft HEAD~1` (discard commit)
   - **Crash** → `git reset --hard HEAD~1` (discard broken code)
10. **Record detail**: ensure `runs/exp_NNN/summary.txt` was written.

## Experiment ideas — start here when stuck

### Hyperparameter sweeps (baseline exploration)
- Sweep guidance_scale: 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5 for current task
- Vary N=1,2,3,5 with different S_START/S_END ranges
- Try DDIM with 25, 50, 100, 200 steps vs full DDPM
- Test weight ratios (e.g., colorization: vary lightness_weight/color_weight ratio)

### Loss function experiments
- Replace MSE with L1 loss for restoration (smoother gradients?)
- Add Total Variation regularization to penalize artifacts
- Add Laplacian pyramid loss for multi-scale supervision
- Use cosine similarity instead of MSE for identity (ref_restoration)
- Add edge-preservation loss (Sobel gradient matching)
- Experiment with loss normalization (divide by spatial dims vs reduction='sum')

### Architecture of guidance
- Try classifier-free guidance style instead of gradient-based guidance
- Add a learned guidance network that predicts the gradient
- Use different guidance scales at different timesteps (schedule)
- Try alternating guidance (apply guidance only every K steps)
- Experiment with gradient clipping or normalization

### New tasks
- Define a "face super-resolution" task (upsample + restore)
- Define a "deblurring" task by adding blur-aware losses
- Define a "face editing" task (guidance toward specific attributes)
- Create a "best of all" composite task that combines multiple constraints

### Sampling innovations
- Try different noise schedules during sampling
- Experiment with warm-up (no guidance for first K steps)
- Try iterative refinement: sample → guide → re-sample
- Use ensemble of different guidance strategies and average

## Timeouts and crashes

- Each experiment should complete within ~15 min. Kill if >20 min.
- Crashes: trivial bugs (typo, None ref) → fix and re-run. Broken ideas → log crash, move on.
- NaN loss: `git reset --hard`, log as crash, move on.

## NEVER STOP

Once the experiment loop begins, do NOT ask the human if you should continue.
Work indefinitely until manually interrupted. Run ~12 experiments/hour, ~100 per
sleep cycle. If stuck, try bigger changes — modify the loss functions, add new
tasks, experiment with the guidance architecture itself.
