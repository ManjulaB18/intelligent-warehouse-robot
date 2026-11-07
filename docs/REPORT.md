# Intelligent Warehouse Robot Path Planning and Package Delivery Using Q-Learning

## 1. Abstract
A warehouse robot learns, with tabular Q-Learning, to pick up a package and deliver it to a station in a 10×10 grid warehouse. The state holds the robot position and a carrying flag. After 5000 training episodes the robot delivered the package in 500 of 500 evaluation episodes in 18 steps on average (the shortest possible for the layout); a random-movement baseline succeeded in 22 of 500.

## 2. Introduction
Warehouses use mobile robots to move goods. Hand-coded routes are inflexible when the layout changes; Reinforcement Learning lets a robot discover routes through interaction.

## 3. Problem Statement
Design an agent that, with no prior map, learns to reach a package, pick it up, avoid obstacles and deliver it in as few moves as possible.

## 4. Objectives
- Build a configurable grid warehouse environment.
- Implement Q-Learning from scratch.
- Train the robot and visualise its learned route.
- Compare against random movement using real simulation results.

## 5. Existing System
Fixed rule-based routes and classical planners (A*, Dijkstra) that need a complete known map.

## 6. Proposed System
A Q-Learning agent that learns a policy from rewards without an explicit map and can be retrained if the layout changes.

## 7. System Architecture
`environment.py` (world, rewards) ⇄ `q_learning.py` (agent, Q-table) → `training.py` (episodes) → `models/q_table.npy` → `evaluation.py` and `visualization.py` → `results/`. `main.py` runs the pipeline.

## 8. Technologies Used
Python, NumPy, Matplotlib, Pillow (GIF), Google Colab, GitHub.

## 9. Reinforcement Learning Fundamentals
Agent, environment, state, action, reward, policy, episode, return, discount factor, exploration vs exploitation.

## 10. Q-Learning Algorithm
Initialise Q to zeros. For each episode: reset; repeat { choose action (ε-greedy), act, observe r and s', update Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)] } until done; decay ε.

## 11. Implementation Methodology
State = (row, col, has_package): 200 states, 4 actions. Rewards: step −1, obstacle −10, wall −10, pickup +20, delivery +100, station without package −5. α=0.1, γ=0.95, ε 1.0→0.01 (decay 0.999), 5000 episodes, 200 max steps per episode.

## 12. Results and Discussion
Measured over 500 evaluation episodes (`results/metrics.json`):

| Metric | Q-Learning | Random |
|---|---|---|
| Successful deliveries | 500 | 22 |
| Success rate | 100 % | 4.4 % |
| Avg steps | 18.0 | 197.9 |
| Avg reward | 102.0 | −694.2 |
| Collisions / episode | 0.0 | 50.7 |

Training graphs show reward rising and steps falling, with 100 % success after about 1000 episodes. Q-Learning clearly outperforms random movement. Results are from one layout and a fixed seed; other layouts or seeds may differ.

## 13. Advantages
Learns without a map; simple and interpretable; adapts by retraining; no external RL libraries.

## 14. Limitations
Tabular method does not scale; static obstacles; one package; discrete grid; deterministic environment.

## 15. Future Enhancements
Multiple packages and robots, dynamic obstacles, battery constraints, Deep Q-Networks, comparison with A*.

## 16. Conclusion
The robot learned an optimal 18-step delivery route and strongly outperformed random movement.

## 17. References
1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279–292.
3. NumPy documentation: https://numpy.org/doc/
4. Matplotlib documentation: https://matplotlib.org/stable/
