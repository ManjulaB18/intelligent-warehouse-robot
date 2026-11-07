"""Q-Learning agent implemented from scratch with NumPy."""
import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999, seed=0):
        self.q = np.zeros((n_states, n_actions))   # the Q-table
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state):
        """Epsilon-greedy: explore with probability epsilon, else exploit."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        row = self.q[state]
        return int(self.rng.choice(np.flatnonzero(row == row.max())))  # random tie-break

    def update(self, s, a, r, s_next, done):
        """Q(s,a) = Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]"""
        target = r if done else r + self.gamma * np.max(self.q[s_next])
        self.q[s, a] += self.alpha * (target - self.q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def best_action(self, state):
        return int(np.argmax(self.q[state]))

    def save(self, path):
        np.save(path, self.q)

    def load(self, path):
        self.q = np.load(path)
