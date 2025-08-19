import glob, os, argparse
import pandas as pd
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--all", action="store_true", help="Aggregate all runs and plot cumulative curves")
args = ap.parse_args()

csvs = sorted(glob.glob("runs/**/progress.csv", recursive=True), key=os.path.getmtime)
assert csvs, "No progress.csv found under runs/"

if not args.all:
    df = pd.read_csv(csvs[-1])

    # Pick x-axis column
    x_candidates = [
        "training_iteration",
        "iteration",
        "episodes_total",
    ]
    x = None
    for col in x_candidates:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            break
    if x is None:
        x = pd.Series(range(len(df)))

    # Pick y-axis column (reward)
    y_candidates = [
        "episode_reward_mean",
        "episode_return_mean",
        "env_runners/episode_return_mean",
        "env_runners/episode_reward_mean",
    ]
    y = None
    for col in y_candidates:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce")
            break
    if y is None:
        y = pd.Series([0.0] * len(df))

    # Drop NaNs for plotting clarity
    mask = ~(x.isna() | y.isna())
    xp, yp = x[mask], y[mask]

    plt.figure()
    plt.plot(xp, yp, marker="o", label="Episode Reward (mean)")
    plt.xlabel("Iteration"); plt.ylabel("Reward"); plt.title("Training Progress")
    plt.legend(); plt.tight_layout(); plt.savefig("training_curve.png", dpi=180)
    print("Saved training_curve.png from:", csvs[-1])
else:
    # Aggregate all runs
    def pick_reward(df):
        for col in [
            "episode_reward_mean",
            "episode_return_mean",
            "env_runners/episode_return_mean",
            "env_runners/episode_reward_mean",
        ]:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
        return pd.Series([float('nan')] * len(df))

    def per_iter_steps(df):
        for col in [
            "env_runners/num_env_steps_sampled_this_iter",
            "num_env_steps_sampled_this_iter",
        ]:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(0)
        # Fallback: use configured batch-size if present in CSV, else 1000
        bs = 1000
        for c in ["train_batch_size", "config/train_batch_size"]:
            if c in df.columns:
                try:
                    bs = int(df[c].iloc[0])
                except Exception:
                    pass
                break
        return pd.Series([bs] * len(df))

    # Build cumulative-iteration plot
    global_iter = 0
    gi_x, gi_y = [], []
    for p in csvs:
        df = pd.read_csv(p)
        r = pick_reward(df)
        clean = r.dropna()
        n = len(clean)
        gi_x.extend(range(global_iter + 1, global_iter + n + 1))
        gi_y.extend(clean.values.tolist())
        global_iter += n
    plt.figure()
    plt.plot(gi_x, gi_y, marker="o", label="Episode Reward (mean)")
    plt.xlabel("Global Iteration"); plt.ylabel("Reward"); plt.title("Training Progress (All Runs)")
    plt.legend(); plt.tight_layout(); plt.savefig("training_curve_all_iters.png", dpi=180)
    print("Saved training_curve_all_iters.png across", len(csvs), "runs")

    # Build cumulative-steps plot
    steps = 0
    gs_x, gs_y = [], []
    for p in csvs:
        df = pd.read_csv(p)
        r = pick_reward(df).fillna(method="ffill")
        s = per_iter_steps(df)
        for d, val in zip(s, r):
            try:
                d = int(d)
            except Exception:
                d = 0
            steps += max(0, d)
            if pd.isna(val):
                continue
            gs_x.append(steps)
            gs_y.append(val)
    plt.figure()
    plt.plot(gs_x, gs_y, marker="o", label="Episode Reward (mean)")
    plt.xlabel("Global Env Steps"); plt.ylabel("Reward"); plt.title("Training Progress vs Steps (All Runs)")
    plt.legend(); plt.tight_layout(); plt.savefig("training_curve_all_steps.png", dpi=180)
    print("Saved training_curve_all_steps.png across", len(csvs), "runs")


