import numpy as np
import matplotlib.pyplot as plt
from tools.math_methods import quaternion_rotation_matrix


def get_gt_mat():
    path = "datasets/rgbd_dataset_freiburg1_xyz_clear_test/ground_truth_sync.txt"
    trajectory_mat = np.zeros([0, 4, 4], dtype=float)
    with open(path) as file:
        next(file)
        for line in file:
            cur_mat = np.eye(4)
            line_arr = np.asarray([float(x) for x in line.split()][1:])
            cur_mat[:3, :3] = quaternion_rotation_matrix(line_arr[3:])
            cur_mat[:3, 3] = line_arr[:3]
            trajectory_mat = np.append(trajectory_mat, [cur_mat], axis=0)
    return trajectory_mat


def plot_3d_trajectory(traj, ax, max_step=10, color="gray", label="traj"):
    # print(traj[:, 1, 3])
    ax.plot3D(
        traj[:max_step, 0, 3],
        traj[:max_step, 1, 3],
        traj[:max_step, 2, 3],
        color,
        label=label,
    )


def plot_gt_and_est(est_traj, steps=10):
    gt = get_gt_mat()
    ax = plt.axes(projection="3d")
    plot_3d_trajectory(gt, ax, steps, "blue", label="Ground Truth")
    plot_3d_trajectory(est_traj, ax, steps, "red", label="Estimated Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


# plot_gt_and_est()
