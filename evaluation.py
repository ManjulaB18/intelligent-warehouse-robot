"""Compare the trained Q-Learning robot with a random-movement baseline."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_policy(env, policy, episodes=500, seed=123):
    rng = np.random.default_rng(seed)
    res = {"reward": [], "steps": [], "success": [], "obstacle_hits": []}
    for _ in range(episodes):
        s, done, total, hits = env.reset(), False, 0, 0
        while not done:
            a = policy(s, rng)
            s, r, done, info = env.step(a)
            total += r
            hits += info["obstacle_hit"] or info["wall_hit"]
        res["reward"].append(total); res["steps"].append(env.steps)
        res["success"].append(int(env.delivered)); res["obstacle_hits"].append(hits)
    return res


def summarize(res):
    succ = np.array(res["success"]); steps = np.array(res["steps"])
    return {
        "Total successful deliveries": int(succ.sum()),
        "Success rate (%)": round(100 * succ.mean(), 1),
        "Avg steps (all episodes)": round(steps.mean(), 1),
        "Avg steps (successful only)": round(steps[succ == 1].mean(), 1) if succ.sum() else float("nan"),
        "Avg reward": round(float(np.mean(res["reward"])), 1),
        "Avg collisions per episode": round(float(np.mean(res["obstacle_hits"])), 1),
    }


def compare(env, agent, episodes=500):
    q_res = run_policy(env, lambda s, rng: agent.best_action(s), episodes)
    r_res = run_policy(env, lambda s, rng: int(rng.integers(4)), episodes)
    return summarize(q_res), summarize(r_res)


def print_table(q_sum, r_sum):
    print(f"{'Metric':35s}{'Q-Learning':>14s}{'Random':>14s}")
    for k in q_sum:
        print(f"{k:35s}{q_sum[k]:>14}{r_sum[k]:>14}")


def plot_training(history, path_dir):
    def smooth(x, w=50):
        return np.convolve(x, np.ones(w) / w, mode="valid")
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(smooth(history["reward"])); ax[0, 0].set_title("Reward per episode (moving avg)")
    ax[0, 1].plot(smooth(history["steps"])); ax[0, 1].set_title("Steps per episode (moving avg)")
    ax[1, 0].plot(smooth(history["success"]) * 100); ax[1, 0].set_title("Success rate % (moving avg)")
    ax[1, 1].plot(history["epsilon"]); ax[1, 1].set_title("Epsilon (exploration) decay")
    for a in ax.flat: a.set_xlabel("Episode"); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{path_dir}/training_progress.png", dpi=150); plt.close()


def plot_comparison(q_sum, r_sum, path_dir):
    keys = ["Success rate (%)", "Avg steps (all episodes)", "Avg reward", "Avg collisions per episode"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    for a, k in zip(ax, keys):
        a.bar(["Q-Learning", "Random"], [q_sum[k], r_sum[k]], color=["#2e8b57", "#c0392b"])
        a.set_title(k, fontsize=10)
        for i, v in enumerate([q_sum[k], r_sum[k]]):
            a.text(i, v, f"{v}", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(f"{path_dir}/q_learning_vs_random.png", dpi=150); plt.close()
