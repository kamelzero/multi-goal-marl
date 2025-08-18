import argparse
import imageio.v2 as imageio
try:  # ensure ffmpeg writer plugin is registered
    import imageio_ffmpeg  # noqa: F401
except Exception:
    pass
from ray.rllib.algorithms.ppo import PPO
from train.rllib_env import RLlibSpread

def _render_frame(env):
    # Try direct render first
    if hasattr(env, "render"):
        try:
            frame = env.render()
            if frame is not None:
                return frame
        except Exception:
            pass
    # Fallbacks into wrapped envs
    for attr in ("raw_env", "par_env", "env"):
        inner = getattr(env, attr, None)
        if inner is not None and hasattr(inner, "render"):
            try:
                frame = inner.render()
                if frame is not None:
                    return frame
            except Exception:
                continue
    return None

def _agent_action_space(env, agent_id):
    # Prefer the underlying PettingZoo parallel env if available
    raw = getattr(env, "raw_env", None)
    if raw is not None and hasattr(raw, "action_space") and callable(raw.action_space):
        try:
            return raw.action_space(agent_id)
        except Exception:
            pass
    # Prefer explicit per-agent dict if available (RLlib wrappers)
    spd = getattr(env, "action_space_dict", None)
    if isinstance(spd, dict) and agent_id in spd:
        return spd[agent_id]

    sp = getattr(env, "action_space", None)
    if sp is not None:
        # Callable style: action_space(agent_id)
        if callable(sp):
            try:
                return sp(agent_id)
            except Exception:
                pass
        # Literal dict style: {agent_id: Space}
        if isinstance(sp, dict) and agent_id in sp:
            return sp[agent_id]
        # Gymnasium Dict space: use .spaces mapping
        spaces_attr = getattr(sp, "spaces", None)
        if isinstance(spaces_attr, dict) and agent_id in spaces_attr:
            return spaces_attr[agent_id]
    # Look inside other wrapped envs
    for attr in ("raw_env", "par_env", "env"):
        inner = getattr(env, attr, None)
        if inner is not None:
            try:
                return _agent_action_space(inner, agent_id)
            except Exception:
                continue
    raise RuntimeError(f"Could not resolve action space for agent: {agent_id}")

def rollout_frames(algo, env, policy_id="shared_policy", max_steps=500):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, _infos = reset_out
    else:
        obs, _infos = reset_out, {}
    frames = []
    steps = 0
    while steps < max_steps:
        # Render
        frame = _render_frame(env)
        if frame is not None:
            frames.append(frame)

        # Actions
        actions = {}
        for agent, ob in obs.items():
            if algo is None:
                space = _agent_action_space(env, agent)
                actions[agent] = space.sample()
            else:
                act, _, _ = algo.get_policy(policy_id).compute_single_action(ob, explore=False)
                actions[agent] = act

        step_out = env.step(actions)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, rewards, terms, truncs, infos = step_out
            done = len(obs) == 0
        else:
            obs, rewards, dones, infos = step_out
            done = bool(dones.get("__all__", False))
        steps += 1
        if done:
            break
    return frames

def save_mp4(frames, out_path, fps=20):
    if not frames:
        print("No frames to write; skipping save.")
        return
    try:
        with imageio.get_writer(out_path, format="FFMPEG", fps=fps) as writer:
            for f in frames:
                writer.append_data(f)
        return
    except Exception as e:
        print(f"imageio FFMPEG writer unavailable ({e}); falling back to imageio-ffmpeg.")
    # Fallback: direct imageio-ffmpeg
    try:
        import imageio_ffmpeg as ffm
        height, width = frames[0].shape[:2]
        proc = ffm.write_frames(out_path, size=(width, height), fps=fps, codec='libx264', pix_fmt='yuv420p')
        proc.send(None)
        for f in frames:
            proc.send(f)
        proc.close()
    except Exception as e:
        raise RuntimeError(f"Failed to write video via imageio-ffmpeg: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--out", type=str, default="rollout.mp4")
    ap.add_argument("--n_agents", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=500)
    args = ap.parse_args()

    env = RLlibSpread({"n_agents": args.n_agents, "render_mode": "rgb_array"})
    algo = PPO.from_checkpoint(args.ckpt) if args.ckpt else None
    frames = rollout_frames(algo, env, max_steps=args.max_steps)
    save_mp4(frames, args.out)
    print(f"Saved {args.out} with {len(frames)} frames")


