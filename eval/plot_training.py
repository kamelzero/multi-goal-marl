import glob, pandas as pd, matplotlib.pyplot as plt

csvs = sorted(glob.glob("runs/**/progress.csv", recursive=True))
assert csvs, "No progress.csv found under runs/"
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


