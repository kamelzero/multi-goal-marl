try:
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv as _PZEnv
except Exception:  # fallback for older ray versions
    from ray.rllib.env import PettingZooEnv as _PZEnv
from envs.spread_wrapper import make_env

class RLlibSpread(_PZEnv):
    def __init__(self, env_config=None):
        cfg = env_config or {}
        self.raw_env = make_env(
            n_agents=cfg.get("n_agents", 3),
            collision_penalty=cfg.get("collision_penalty", 1.0),
            action_penalty=cfg.get("action_penalty", 0.01),
            cover_radius=cfg.get("cover_radius", 0.1),
            render_mode=cfg.get("render_mode", None),
        )
        super().__init__(self.raw_env)

    def render(self):
        # Proxy render to underlying env when possible
        try:
            return self.raw_env.render()
        except Exception:
            return None


