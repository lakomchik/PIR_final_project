import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from typing import List


class Animation:
    def __init__(self) -> None:
        self.ground_truth_states = None
        self.real_states = None

        self.states = None

    def plot_next_point(self, ax: plt.Axes, num: int, colors: List[str]):
        gt_scat = ax.scatter(
            self.ground_truth_states[num, 0, 3],
            self.ground_truth_states[num, 1, 3],
            self.ground_truth_states[num, 2, 3],
            color=colors[0],
            s=10,
        )
        real_scat = ax.scatter(
            self.real_states[num, 0, 3],
            self.real_states[num, 1, 3],
            self.real_states[num, 2, 3],
            color=colors[1],
            s=10,
        )
        if num == 0:
            ax.legend(
                [gt_scat, real_scat], ["Ground truth trajectory", "Real trajectory"]
            )
            return

        ax.plot(
            [
                self.ground_truth_states[num - 1, 0, 3],
                self.ground_truth_states[num, 0, 3],
            ],
            [
                self.ground_truth_states[num - 1, 1, 3],
                self.ground_truth_states[num, 1, 3],
            ],
            [
                self.ground_truth_states[num - 1, 2, 3],
                self.ground_truth_states[num, 2, 3],
            ],
            color=colors[0],
        )
        ax.plot(
            [self.real_states[num, 0, 3], self.real_states[num - 1, 0, 3]],
            [self.real_states[num, 1, 3], self.real_states[num - 1, 1, 3]],
            [self.real_states[num, 2, 3], self.real_states[num - 1, 2, 3]],
            color=colors[1],
        )

    def animate_full(
        self,
        num: int,
        ax: plt.Axes,
        colors: List[str],
    ):
        self.plot_next_point(ax, num, colors)
        return ()

    def make_full_animation(
        self, ground_truth_states: np.ndarray, real_states: np.ndarray
    ):
        self.ground_truth_states = ground_truth_states
        self.real_states = real_states

        figure = plt.figure(figsize=(10, 10))
        ax = figure.add_subplot(projection="3d")

        ax.set_box_aspect([1, 0.3, 0.3])
        ax.azim = 120
        # ax.elev = 10

        ax.set_title("Trajectory of camera")

        num_steps = ground_truth_states.shape[0]
        colors = ["green", "red"]
        anim = FuncAnimation(
            figure,
            self.animate_full,
            frames=num_steps,
            fargs=(ax, colors),
            blit=True,
        )
        # writer = FFMpegWriter(fps=5)
        # anim.save("animation.mp4", writer=writer, dpi=300)
        plt.show()
