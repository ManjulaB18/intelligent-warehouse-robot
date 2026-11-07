"""Run the full pipeline: train -> save -> evaluate -> visualize."""
import json, os
from environment import WarehouseEnv
from training import train
from evaluation import compare, print_table, plot_training, plot_comparison
from visualization import plot_route, animate, plot_warehouse

os.makedirs("results/graphs", exist_ok=True); os.makedirs("models", exist_ok=True)

env = WarehouseEnv()
plot_warehouse(env, "results/graphs/warehouse_layout.png")

print("=== Training ===")
agent, history = train(env, episodes=5000)
agent.save("models/q_table.npy")
plot_training(history, "results/graphs")

print("\n=== Evaluation (500 episodes each) ===")
q_sum, r_sum = compare(env, agent, episodes=500)
print_table(q_sum, r_sum)
plot_comparison(q_sum, r_sum, "results/graphs")
json.dump({"q_learning": q_sum, "random": r_sum}, open("results/metrics.json", "w"), indent=2)

n = plot_route(env, agent, "results/graphs/learned_route.png")
print(f"\nLearned route length: {n} steps")
animate(env, agent, "results/graphs/robot_animation.gif")
print("Saved graphs, metrics, q_table and animation.")
