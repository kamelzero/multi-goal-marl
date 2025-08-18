import numpy as np
from pettingzoo.mpe import simple_spread_v3
import supersuit as ss

def make_env(
    n_agents=3,
    collision_penalty=1.0,
    action_penalty=0.01,
    cover_radius=0.1,
    render_mode=None,
):
    # Base parallel env (works with RLlib's PettingZooEnv)
    env = simple_spread_v3.parallel_env(
        N=n_agents,
        local_ratio=0.5,
        continuous_actions=True,
        render_mode=render_mode,
    )

    # Keep original step
    original_step = env.step

    def _step(actions):
        obs, rewards, terminations, truncations, infos = original_step(actions)

        # Positions from underlying AEC env used by parallel wrapper
        world = env.aec_env.world
        landmarks = np.array([l.state.p_pos for l in world.landmarks])
        agents = np.array([a.state.p_pos for a in world.agents])

        # Coverage
        if len(agents) > 0 and len(landmarks) > 0:
            dmat = np.linalg.norm(agents[:, None, :] - landmarks[None, :, :], axis=-1)
            covered = (dmat < cover_radius).any(axis=0).sum()
        else:
            covered = 0

        # Collisions (simple radius threshold)
        collisions = 0
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                if np.linalg.norm(agents[i] - agents[j]) < 0.05:
                    collisions += 1

        # Effort from action magnitudes
        effort = 0.0
        for a in actions.values():
            arr = np.asarray(a)
            effort += float(np.sum(arr * arr))

        shaped = float(covered) - collision_penalty * float(collisions) - action_penalty * effort
        per_agent = shaped / max(1, len(rewards) if rewards else 1)

        for agent_id in rewards.keys():
            rewards[agent_id] = float(rewards.get(agent_id, 0.0)) + per_agent

        return obs, rewards, terminations, truncations, infos

    env.step = _step
    return env


