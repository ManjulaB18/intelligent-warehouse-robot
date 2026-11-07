# Viva Questions and Simple Answers

1. **What is Reinforcement Learning?** A way for an agent to learn by interacting with an environment and receiving rewards or penalties, without being told the right answer.
2. **What is Q-Learning?** A model-free RL algorithm that learns a table of values Q(s,a) telling how good each action is in each state.
3. **What is the Q-table?** A 200 × 4 array (states × actions) holding the learned value of each action.
4. **What is the state in your project?** (row, col, has_package).
5. **Why include `has_package` in the state?** The right move depends on it: before pickup go to the package, after pickup go to the station. Without it the robot cannot tell the two situations apart.
6. **What actions are available?** Up, down, left, right. Pickup and delivery are automatic on entering the cell.
7. **Why automatic pickup/delivery?** It keeps the action space small; the learning goal is navigation.
8. **Explain the Q-Learning formula.** Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]: move the old estimate toward reward plus discounted best future value.
9. **What is the learning rate α?** How much new information overrides old (0.1 here).
10. **What is the discount factor γ?** How much future rewards matter (0.95 here); near 1 means far-sighted.
11. **What is exploration vs exploitation?** Trying new actions to discover better routes vs using what is already known to be best.
12. **What is epsilon-greedy?** With probability ε take a random action, otherwise the best known action.
13. **Why decay epsilon?** Explore a lot early, then rely on learned knowledge.
14. **Why is every step −1?** To make the robot prefer short routes.
15. **Why −10 for obstacles?** So the robot learns obstacles are bad; it stays in place when it hits one.
16. **Why +20 pickup and +100 delivery?** Pickup is a sub-goal that guides learning; delivery is the final goal so it is larger.
17. **What if the robot reaches the station without the package?** −5 and the episode continues.
18. **What is an episode?** One attempt, from reset to delivery or the 200-step limit.
19. **How do you know it learned?** Reward rose and steps fell during training, and evaluation shows 100 % success vs 4.4 % for random.
20. **What is the random baseline?** A robot choosing actions uniformly at random; it shows what no learning achieves.
21. **Is the 18-step route optimal?** Yes for this layout: Manhattan distance start→package→station is 18 and obstacles force no detour.
22. **Why doesn't Q-Learning scale to large problems?** The table grows with the number of states; large problems need function approximation (DQN).
23. **What is the Bellman equation's role?** It expresses a state's value as immediate reward plus the value of the next state; Q-Learning iteratively approximates it.
24. **Is Q-Learning on-policy or off-policy?** Off-policy: it updates with the max over next actions regardless of the action actually taken.
25. **How can the environment be changed?** Pass `size`, `obstacles`, `start`, `package`, `station` to `WarehouseEnv`.
26. **What are the limitations?** Single package, static obstacles, discrete grid, tabular method.
