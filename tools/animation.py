import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List

from math_methods import quaternion_rotation_matrix


class Animation:
    def __init__(self) -> None:
        self.ground_truth_states = None
        self.real_states = None

        self.states = None

    @staticmethod
    def plot_arrow(ax: plt.Axes, state: np.ndarray, color: str):
        ax.scatter(state[0], state[1], state[2], color=color, s=50)
        arrow_length = 0.05
        arrow = np.array(
            [
                [arrow_length],
                [0],
                [0],
            ]
        )
        R_full = quaternion_rotation_matrix(state[3:])
        arrow_vector = R_full @ arrow
        arrow_end = state[0:3][np.newaxis].T + arrow_vector
        ax.scatter(arrow_end[0], arrow_end[1], arrow_end[2], color=color, s=1)
        ax.plot(
            np.array([state[0], arrow_end[0]], dtype=object),
            np.array([state[1], arrow_end[1]], dtype=object),
            np.array([state[2], arrow_end[2]], dtype=object),
            color=color,
        )

    def animate_full(
        self,
        num: int,
        ax: plt.Axes,
        colors: List[str],
    ):
        Animation.plot_arrow(ax, self.ground_truth_states[num], colors[0])
        Animation.plot_arrow(ax, self.real_states[num], colors[1])
        return ()

    def make_full_animation(
        self, ground_truth_states: np.ndarray, real_states: np.ndarray
    ):
        self.ground_truth_states = ground_truth_states
        self.real_states = real_states

        figure = plt.figure(figsize=(10, 10))
        ax = figure.add_subplot(projection="3d")
        min_x = min(np.min(ground_truth_states[:, 0]), np.min(real_states[:, 0]))
        max_x = max(np.max(ground_truth_states[:, 0]), np.max(real_states[:, 0]))
        min_y = min(np.min(ground_truth_states[:, 1]), np.min(real_states[:, 1]))
        max_y = max(np.max(ground_truth_states[:, 1]), np.max(real_states[:, 1]))
        min_z = min(np.min(ground_truth_states[:, 2]), np.min(real_states[:, 2]))
        max_z = max(np.max(ground_truth_states[:, 2]), np.max(real_states[:, 2]))
        ax.set(xlim3d=(min_x, max_x), xlabel="X")
        ax.set(ylim3d=(min_y, max_y), ylabel="Y")
        ax.set(zlim3d=(min_z, max_z), zlabel="Z")
        num_steps = ground_truth_states.shape[0]
        colors = ["green", "red"]
        anim = FuncAnimation(
            figure,
            self.animate_full,
            frames=num_steps,
            fargs=(ax, colors),
            blit=True,
        )
        plt.show()

    def animate_one(
        self,
        num: int,
        ax: plt.Axes,
        color: str,
    ):
        Animation.plot_arrow(ax, self.states[num], color)
        return ()

    def make_single_animation(self, states: np.ndarray):
        self.states = states

        figure = plt.figure(figsize=(10, 10))
        ax = figure.add_subplot(projection="3d")
        min_x = np.min(states[:, 0])
        max_x = np.max(states[:, 0])
        min_y = np.min(states[:, 1])
        max_y = np.max(states[:, 1])
        min_z = np.min(states[:, 2])
        max_z = np.max(states[:, 2])
        ax.set(xlim3d=(min_x, max_x), xlabel="X")
        ax.set(ylim3d=(min_y, max_y), ylabel="Y")
        ax.set(zlim3d=(min_z, max_z), zlabel="Z")
        num_steps = states.shape[0]
        color = "green"
        anim = FuncAnimation(
            figure,
            self.animate_one,
            frames=num_steps,
            fargs=(ax, color),
            blit=True,
        )
        plt.show()


gt = np.array(
    [
        [1.2764, -0.9763, 0.6837, 0.8187, 0.3639, -0.1804, -0.4060],
        [1.1346, -0.8934, 0.6791, 0.7747, 0.4379, -0.2228, -0.3980],
    ]
)
real = np.array(
    [
        [1.1346, -0.8934, 0.6791, 0.7747, 0.4379, -0.2228, -0.3980],
        [1.2764, -0.9763, 0.6837, 0.8187, 0.3639, -0.1804, -0.4060],
    ]
)

Animation().make_single_animation(gt)
Animation().make_full_animation(gt, real)
