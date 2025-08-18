import glob, pandas as pd, matplotlib.pyplot as plt

csvs = sorted(glob.glob("runs/**/progress.csv", recursive=True))
assert csvs, "No progress.csv found under runs/"
df = pd.read_csv(csvs[-1])

plt.figure()
plt.plot(df.get("training_iteration", range(len(df))), df.get("episode_reward_mean", df.get("episode_return_mean", [0]*len(df))), label="Episode Reward (mean)")
plt.xlabel("Iteration"); plt.ylabel("Reward"); plt.title("Training Progress")
plt.legend(); plt.tight_layout(); plt.savefig("training_curve.png", dpi=180)
print("Saved training_curve.png")


