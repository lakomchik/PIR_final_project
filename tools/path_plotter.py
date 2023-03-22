import numpy as np
import matplotlib.pyplot as plt
from tools.math_methods import quaternion_rotation_matrix


def get_gt_mat():
    path = "datasets/rgbd_dataset_freiburg1_xyz_clear_test/ground_truth_sync.txt"
    trajectory_mat = np.zeros([0, 4, 4], dtype=float)
    with open(path) as file:
        next(file)
        ts = np.empty(0, dtype=float)
        for line in file:
            cur_mat = np.eye(4)
            ts = np.append(ts, ([float(x) for x in line.split()][:1]))
            line_arr = np.asarray([float(x) for x in line.split()][1:])
            cur_mat[:3, :3] = quaternion_rotation_matrix(line_arr[3:])
            cur_mat[:3, 3] = line_arr[:3]
            trajectory_mat = np.append(trajectory_mat, [cur_mat], axis=0)
        avg_time = 0.0
        for i in range(1, ts.shape[0]):
            avg_time += ts[i] - ts[i - 1]
        avg_time /= ts.shape[0]
        print("AVG_TS =, ", avg_time)
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
    ax.scatter(
        traj[:max_step, 0, 3],
        traj[:max_step, 1, 3],
        traj[:max_step, 2, 3],
        color=color,
        linewidth=3,
    )
    # arrow_length = 0.1
    # arrow = np.array(
    #     [
    #         [arrow_length],
    #         [0],
    #         [0],
    #     ]
    # )
    # for i in range(max_step):
    #     arrow_vector = traj[i, :3, :3].dot(arrow)
    #     # print(traj[i, :3, :3])
    #     # arrow_vector = arrow_vector.reshape(-1)
    #     ax.quiver(
    #         traj[i, 0, 3],
    #         traj[i, 1, 3],
    #         traj[i, 2, 3],
    #         arrow_vector[0, 0],
    #         arrow_vector[1, 0],
    #         arrow_vector[2, 0],
    #         # pivot="middle",
    #         length=0.1,
    #         lw=2,
    #     )


def plot_gt_and_est(ax, est_traj, steps=10):
    gt = get_gt_mat()

    plot_3d_trajectory(gt, ax, steps, "blue", label="Ground Truth")
    plot_3d_trajectory(est_traj, ax, steps, "red", label="Estimated Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()


# plot_gt_and_est()
