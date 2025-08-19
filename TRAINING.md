## Practical training guide (CPU-friendly)

### Quick recipe (CPU laptop)

Run 50 fast iterations with a small network:

```bash
FAST_SMALL_MODEL=1 \
FAST_NUM_WORKERS=0 \
FAST_TRAIN_BATCH=1000 \
FAST_ROLLOUT_LEN=100 \
FAST_STOP_ITERS=50 \
make train
```

Then plot and record:

```bash
make plot
make video   # if you trained first, this will use the newest checkpoint
```

Aggregate across all runs (cumulative plots):

```bash
PYTHONPATH=$PWD python -m eval.plot_training --all
```

### How long should I run?

- With the small settings above, expect useful behaviors after ~30–100 iterations on CPU.
- For stronger results, aim for 1–3M env steps (e.g., 60–200 iterations at 16k/iter, or proportionally more with smaller batches). On the new RLlib API, track total steps via `env_runners/num_env_steps_sampled_lifetime` (not `timesteps_total`).

### Make iterations faster (on CPU)

- Reduce per-iteration work using environment variables:
  - `FAST_TRAIN_BATCH` (total steps per training iteration)
  - `FAST_ROLLOUT_LEN` (steps per fragment)
  - `FAST_SMALL_MODEL=1` (use `[64, 64]` network instead of `[256, 256]`)
  - `FAST_NUM_WORKERS=0` (single local runner)
  - `FAST_STOP_ITERS` (how many iterations to run)

Examples:

```bash
# Very fast smoke
FAST_SMALL_MODEL=1 FAST_NUM_WORKERS=0 FAST_TRAIN_BATCH=200 FAST_ROLLOUT_LEN=50 FAST_STOP_ITERS=10 make train

# Moderate speed
FAST_SMALL_MODEL=1 FAST_NUM_WORKERS=0 FAST_TRAIN_BATCH=2000 FAST_ROLLOUT_LEN=200 FAST_STOP_ITERS=50 make train
```

### Resume training from a checkpoint

Get the latest checkpoint path (preferred: pointer file written by training):

```bash
# Preferred (written automatically by train/train_rllib_ppo.py)
export CKPT=$(cat runs/latest_checkpoint_path.txt)

# Fallback (searches directories)
if [ -z "$CKPT" ]; then CKPT=$(ls -dt runs/**/checkpoint_* 2>/dev/null | head -n1); fi
echo "Checkpoint: $CKPT"
```

Continue training (example: 50 more iterations) and save a new checkpoint under `runs/`:

```bash
PYTHONPATH=$PWD python - <<'PY'
from ray.rllib.algorithms.algorithm import Algorithm
import os

def get_any(d: dict, keys):
    for k in keys:
        if k in d:
            return d[k]
        if '/' in k:
            first, rest = k.split('/', 1)
            sub = d.get(first)
            if isinstance(sub, dict) and rest in sub:
                return sub.get(rest)
    return None

ckpt = os.environ.get('CKPT')
print('Resuming from:', ckpt)
algo = Algorithm.from_checkpoint(ckpt)

# Local cumulative counter to avoid None/negative metrics
cum_steps = 0
fallback_bs = int(os.getenv('FAST_TRAIN_BATCH', '1000'))

for i in range(50):
    r = algo.train()
    mean_r = get_any(r, [
        'env_runners/episode_return_mean',
        'env_runners/episode_reward_mean',
        'episode_return_mean',
        'episode_reward_mean',
    ])
    iter_steps = get_any(r, [
        'env_runners/num_env_steps_sampled_this_iter',
        'num_env_steps_sampled_this_iter',
    ])
    if isinstance(iter_steps, (int, float)) and iter_steps > 0:
        cum_steps += int(iter_steps)
        iter_disp = int(iter_steps)
    else:
        cum_steps += fallback_bs
        iter_disp = f'est:{fallback_bs}'

    life_steps = get_any(r, [
        'env_runners/num_env_steps_sampled_lifetime',
        'num_env_steps_sampled_lifetime',
    ])
    life_disp = int(life_steps) if isinstance(life_steps, (int, float)) and life_steps >= 0 else cum_steps

    print(f'Iter {i+1} mean_reward={mean_r} env_steps_iter={iter_disp} env_steps_total={life_disp}')

new_ckpt = algo.save(checkpoint_dir='runs').checkpoint.path
print('Saved new checkpoint:', new_ckpt)
open(os.path.join('runs', 'latest_checkpoint_path.txt'), 'w').write(new_ckpt+'\n')
PY
```

How to tell it actually resumed vs started from scratch:
- You should see a console line like “Resuming from: <checkpoint>”.
- The first `result` printed after resuming will often be close to where you left off (reward won’t reset to a brand‑new policy’s level).
- If you run with verbose logs, RLlib typically logs a “Restored from checkpoint …” message.
- Programmatically, you can compare the new checkpoint path timestamp to ensure it was created after resuming, not from a fresh run.

### Record a trained rollout

```bash
CKPT=$(cat runs/latest_checkpoint_path.txt)
if [ -z "$CKPT" ]; then CKPT=$(ls -dt runs/**/checkpoint_* 2>/dev/null | head -n1); fi
PYTHONPATH=$PWD python -m eval.record_video --ckpt "$CKPT" --out trained.mp4 --max_steps 500
```

### Notes & tips

- If `make video` runs without a checkpoint, it will generate a quick side‑by‑side using random policy for both panels (fast fallback).
- If plotting shows only one point, you likely trained for 1 iteration. Increase `FAST_STOP_ITERS`.
- Install Torch (`pip install torch`) for the torch backend; GPU is optional.


