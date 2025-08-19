import argparse
import os
import time
from ray.rllib.algorithms.algorithm import Algorithm


def get_any(d: dict, keys):
    for k in keys:
        if k in d:
            return d[k]
        if "/" in k:
            a, b = k.split("/", 1)
            sub = d.get(a)
            if isinstance(sub, dict) and b in sub:
                return sub[b]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--fallback-batch", type=int, default=1000)
    args = ap.parse_args()

    print("Resuming from:", args.ckpt)
    algo = Algorithm.from_checkpoint(args.ckpt)

    cumulative = 0
    for i in range(args.iters):
        r = algo.train()
        rew = get_any(
            r,
            [
                "env_runners/episode_return_mean",
                "env_runners/episode_reward_mean",
                "episode_return_mean",
                "episode_reward_mean",
            ],
        )
        step = get_any(
            r,
            [
                "env_runners/num_env_steps_sampled_this_iter",
                "num_env_steps_sampled_this_iter",
            ],
        )
        if isinstance(step, (int, float)) and step > 0:
            cumulative += int(step)
            step_disp = int(step)
        else:
            cumulative += int(args.fallback_batch)
            step_disp = f"est:{args.fallback_batch}"
        print(
            f"iter {i+1} mean_reward={rew} env_steps_iter={step_disp} env_steps_total={cumulative}"
        )

    # Save to absolute, timestamped directory to avoid pyarrow URI issues
    base = os.path.abspath("runs")
    os.makedirs(base, exist_ok=True)
    target = os.path.join(base, f"checkpoint_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(target, exist_ok=True)
    new_ckpt = algo.save(checkpoint_dir=target).checkpoint.path
    print("Saved checkpoint:", new_ckpt)
    try:
        with open(os.path.join(base, "latest_checkpoint_path.txt"), "w") as f:
            f.write(new_ckpt + "\n")
        link = os.path.join(base, "latest_checkpoint")
        if os.path.islink(link) or os.path.exists(link):
            try:
                os.remove(link)
            except Exception:
                pass
        try:
            os.symlink(new_ckpt, link)
        except Exception:
            pass
    except Exception:
        pass


if __name__ == "__main__":
    main()


