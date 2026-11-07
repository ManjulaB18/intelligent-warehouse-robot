"""Warehouse drawing, learned-route plot and animation (works in Colab)."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

COL = {"empty": "#f4f1ea", "obstacle": "#34495e", "visited": "#aed6f1",
       "package": "#e67e22", "station": "#27ae60", "robot": "#e74c3c"}


def get_route(env, agent, max_steps=100):
    """Follow the greedy (best) action from the Q-table. Returns list of frames."""
    s = env.reset()
    frames = [(env.pos, env.has_package, env.delivered)]
    done = False
    while not done and env.steps < max_steps:
        s, r, done, info = env.step(agent.best_action(s))
        frames.append((env.pos, env.has_package, env.delivered))
    return frames


def draw_frame(ax, env, frames, k, title=""):
    ax.clear()
    n = env.size
    visited = {f[0] for f in frames[:k + 1]}
    for r in range(n):
        for c in range(n):
            color = COL["empty"]
            if (r, c) in env.obstacles: color = COL["obstacle"]
            elif (r, c) in visited: color = COL["visited"]
            ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, fc=color, ec="#999", lw=.8))
    pr, pc = env.package; sr, sc = env.station
    pos, has_pkg, delivered = frames[k]
    if not has_pkg and not delivered:
        ax.add_patch(Rectangle((pc + .15, n - 1 - pr + .15), .7, .7, fc=COL["package"], ec="k"))
        ax.text(pc + .5, n - 1 - pr + .5, "P", ha="center", va="center", weight="bold")
    ax.add_patch(Rectangle((sc + .05, n - 1 - sr + .05), .9, .9, fc=COL["station"], ec="k", alpha=.9))
    ax.text(sc + .5, n - 1 - sr + .5, "D", ha="center", va="center", color="w", weight="bold")
    ax.add_patch(plt.Circle((pos[1] + .5, n - 1 - pos[0] + .5), .32, fc=COL["robot"], ec="k"))
    if has_pkg:
        ax.add_patch(Rectangle((pos[1] + .38, n - 1 - pos[0] + .38), .24, .24, fc=COL["package"], ec="k"))
    status = "DELIVERED!" if delivered else ("Carrying package" if has_pkg else "Going to package")
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{title}Step {k} - {status}")


def plot_route(env, agent, path):
    frames = get_route(env, agent)
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_frame(ax, env, frames, len(frames) - 1, "Learned route | ")
    xs = [f[0][1] + .5 for f in frames]; ys = [env.size - 1 - f[0][0] + .5 for f in frames]
    ax.plot(xs, ys, "-", color="#c0392b", lw=2, alpha=.8)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return len(frames) - 1


def animate(env, agent, gif_path=None, interval=300):
    """Returns a FuncAnimation. In Colab: HTML(anim.to_jshtml())."""
    frames = get_route(env, agent)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    anim = FuncAnimation(fig, lambda k: draw_frame(ax, env, frames, k),
                         frames=len(frames), interval=interval, repeat=False)
    if gif_path:
        anim.save(gif_path, writer=PillowWriter(fps=3))
    plt.close(fig)
    return anim


def plot_warehouse(env, path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_frame(ax, env, [(env.start, False, False)], 0, "Warehouse layout | ")
    if path: plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
