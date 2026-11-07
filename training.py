"""Training loop for the Q-Learning warehouse robot."""
import numpy as np
from q_learning import QLearningAgent


def train(env, episodes=5000, alpha=0.1, gamma=0.95, epsilon_decay=0.999,
          seed=0, log_every=500, verbose=True):
    agent = QLearningAgent(env.n_states, env.n_actions, alpha, gamma,
                           epsilon_decay=epsilon_decay, seed=seed)
    history = {"reward": [], "steps": [], "success": [], "epsilon": []}
    for ep in range(1, episodes + 1):
        s = env.reset()
        total, done = 0, False
        while not done:
            a = agent.choose_action(s)
            s2, r, done, info = env.step(a)
            agent.update(s, a, r, s2, done)
            s, total = s2, total + r
        agent.decay_epsilon()
        history["reward"].append(total)
        history["steps"].append(env.steps)
        history["success"].append(int(env.delivered))
        history["epsilon"].append(agent.epsilon)
        if verbose and ep % log_every == 0:
            w = slice(-log_every, None)
            print(f"Episode {ep:5d} | avg reward {np.mean(history['reward'][w]):8.2f} | "
                  f"success {np.mean(history['success'][w])*100:5.1f}% | "
                  f"avg steps {np.mean(history['steps'][w]):6.1f} | eps {agent.epsilon:.3f}")
    return agent, history
