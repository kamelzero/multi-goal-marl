## Multi-goal MARL: MPE Simple Spread

Multi-agent PPO on PettingZoo MPE `simple_spread` with a team reward that balances three goals: landmark coverage, collision avoidance, and control efficiency. The project is reproducible, Dockerizable, and produces MP4s and training curves.

### Real-world relevance

While simple_spread is a minimal 2-D benchmark, it directly mirrors coordination challenges faced in real systems. Think of fleets of drones or autonomous vehicles tasked with covering multiple surveillance points, delivery zones, or patrol areas. Each unit must spread out to cover all targets (landmark coverage), avoid collisions for safety, and move efficiently to conserve limited battery or fuel. These are the same principles that guide coordination in commercial logistics, agricultural monitoring, disaster-response robotics, and even military UAV patrols. This project demonstrates how deep reinforcement learning can discover decentralized strategies for such multi-goal, multi-agent tasks.

### Quickstart

```bash
make setup
make train
make video
make plot
```

Artifacts:
- `side_by_side.mp4` (random vs trained rollout)
- `training_curve.png` (from RLlib `progress.csv`)

CPU-only works; GPU is optional if available.

### Repo layout

```
multi-goal-marl/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ config.yaml
├─ Makefile
├─ docker/
│  └─ Dockerfile
├─ envs/
│  └─ spread_wrapper.py
├─ train/
│  ├─ rllib_env.py
│  └─ train_rllib_ppo.py
└─ eval/
   ├─ record_video.py
   └─ plot_training.py
```

### Deliverables (what each file does)
- `envs/spread_wrapper.py`: PettingZoo `simple_spread` with reward shaping and `render_mode="rgb_array"` support.
- `train/rllib_env.py`: RLlib wrapper around the PettingZoo env (shared policy mapping).
- `train/train_rllib_ppo.py`: trains PPO in RLlib; saves checkpoints to `runs/`.
- `eval/record_video.py`: writes `random.mp4` and `trained.mp4` and can be used to record short smoke videos.
- `eval/plot_training.py`: reads latest `runs/**/progress.csv` and saves `training_curve.png`.
- `docker/Dockerfile`: headless container with `ffmpeg` and Python deps.
- `Makefile`: convenience targets.
- `config.yaml`: environment and training hyperparameters.

### Reward shaping (team reward)
- Coverage: count landmarks within radius `cover_radius` covered by at least one agent.
- Collisions: penalty for pairs of agents closer than a threshold (0.05 world units by default).
- Effort: L2 penalty on action magnitudes.
- Team reward: shaped value added equally to all agents each step.

Parameters (see `config.yaml`):
- `cover_radius`: 0.1
- `collision_penalty`: 1.0
- `action_penalty`: 0.01

### Make targets
- `make setup`: create venv and install requirements.
- `make train`: run PPO training with RLlib; checkpoints under `runs/`.
- `make video`: produce `random.mp4`, `trained.mp4`, and `side_by_side.mp4`.
- `make plot`: write `training_curve.png` from latest `progress.csv`.
- `make docker-build` / `make docker-run`: build and run container; default command runs a demo.
- `make clean`: remove `runs/` and generated media.

### Configuration
Tune hyperparameters in `config.yaml`. Notable keys:
- `env.*`: number of agents, shaping weights, `render_mode`.
- `train.*`: RLlib PPO settings (workers, batch sizes, learning rate, network, stop criteria, log dir).

Environment variables to speed up training (used by CI):
- `FAST_STOP_ITERS`, `FAST_NUM_WORKERS`, `FAST_TRAIN_BATCH`, `FAST_ROLLOUT_LEN`.

Example:
```bash
FAST_STOP_ITERS=10 FAST_NUM_WORKERS=0 make train
```

### Docker

```bash
make docker-build
make docker-run   # mounts repo, runs demo inside container
```

### AWS (Ubuntu 22.04)

```bash
sudo apt update && sudo apt install -y python3.10-venv ffmpeg git
git clone <your-repo>
cd multi-goal-marl
make setup
make train
make video
make plot
```

### CI
GitHub Actions runs a CPU smoke test that:
- Installs deps
- Trains for 1 fast iteration (with `FAST_*` env vars)
- Plots the training curve
- Records a short random video

### Troubleshooting
- If video creation is slow on small instances, reduce `--max_steps` (e.g., 300) when recording.
- If `ffmpeg` is missing locally, install it via your OS package manager.
- If PettingZoo or RLlib versions shift APIs, pin versions in `requirements.txt` or update `eval/plot_training.py` column names.
