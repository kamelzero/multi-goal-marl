import yaml, os, time
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from train.rllib_env import RLlibSpread

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def policy_mapping_fn(agent_id, *args, **kwargs):
    # Compatible with older/newer RLlib signatures regardless of extra params
    return "shared_policy"

if __name__ == "__main__":
    cfg = load_cfg()
    env_cfg = cfg["env"]
    trn = cfg["train"]

    # Speed toggles for CI via env vars
    stop_iter = int(os.getenv("FAST_STOP_ITERS", trn["stop_training_iteration"]))
    num_workers = int(os.getenv("FAST_NUM_WORKERS", trn["num_workers"]))
    train_batch_size = int(os.getenv("FAST_TRAIN_BATCH", trn["train_batch_size"]))
    rollout_fragment_length = int(os.getenv("FAST_ROLLOUT_LEN", trn["rollout_fragment_length"]))
    small_model = os.getenv("FAST_SMALL_MODEL", "0") == "1"
    time_limit_s = int(os.getenv("FAST_TIME_LIMIT", "0"))

    algo_cfg = (
        PPOConfig()
        .environment(env=RLlibSpread, env_config=env_cfg)
        .framework(trn["framework"])
        .env_runners(
            num_env_runners=num_workers,
            rollout_fragment_length=rollout_fragment_length,
            batch_mode="truncate_episodes"
        )
        .training(
            train_batch_size=train_batch_size,
            gamma=trn["gamma"],
            lr=trn["lr"],
            model={
                "fcnet_hiddens": ([64, 64] if small_model else trn["fcnet_hiddens"]),
                "vf_share_layers": trn["vf_share_layers"],
            },
        )
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_mapping_fn
        )
    )

    # Use Algorithm directly for simplicity with new API; save a checkpoint at the end
    from ray.rllib.algorithms import ppo as ppo_mod
    from ray.tune.logger import UnifiedLogger

    def _logger_creator(cfg):
        timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
        logdir = os.path.join(trn["local_dir"], f"PPO_{timestr}")
        os.makedirs(logdir, exist_ok=True)
        return UnifiedLogger(cfg, logdir, loggers=None)

    algo = ppo_mod.PPO(config=algo_cfg.to_dict(), logger_creator=_logger_creator)
    start_time = time.time()
    for i in range(stop_iter):
        result = algo.train()
        print(
            f"Iter {i+1}/{stop_iter}: episode_reward_mean="
            f"{result.get('episode_reward_mean', result.get('env_runners/episode_return_mean'))} "
            f"timesteps_total={result.get('timesteps_total', result.get('num_env_steps_sampled_lifetime'))}"
        )
        if time_limit_s and (time.time() - start_time) >= time_limit_s:
            print(f"FAST_TIME_LIMIT reached ({time_limit_s}s). Stopping early.")
            break
    # Ensure absolute checkpoint directory exists (avoid pyarrow URI issues)
    ckpt_base = os.path.abspath(trn["local_dir"]) if trn.get("local_dir") else os.path.abspath("runs")
    os.makedirs(ckpt_base, exist_ok=True)
    chkpt_dir = algo.save(checkpoint_dir=ckpt_base).checkpoint.path
    print(f"Saved checkpoint: {chkpt_dir}")


