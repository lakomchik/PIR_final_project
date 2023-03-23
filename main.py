import numpy as np
import cv2
import mrob


from tools.data_reader import get_observation
from tools.math_methods import point2xyz
from tools.math_methods import quaternion_rotation_matrix
from visual_odometry import get_translation_matrix
import math
from tools.math_methods import rotation_matrix
from tools.path_plotter import get_gt_mat


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class GraphSlam:
    def __init__(self, img, depth, init_mat) -> None:
        self.graph = mrob.FGraph()
        self.detected_features = (
            {}
        )  # dictionary of memorized features key: orb feature; value: idx in graph
        self.poses_id = []  # idx in graph for poses
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher()
        self.W_obs = np.eye(3) * 10e2
        n1 = self.graph.add_node_pose_3d(mrob.geometry.SE3(init_mat))
        self.poses_id.append(n1)
        self.W_0 = 10e6 * np.identity(6)  # covariation of pose
        self.graph.add_factor_1pose_3d(
            mrob.geometry.SE3(init_mat), self.poses_id[-1], self.W_0
        )
        kp, ds = self.get_img_features(img)
        self.add_3d_landmarks(kp, ds, depth)

        pass

    def get_img_features(self, img, type="ORB"):
        if type == "ORB":
            keypoints, descriptions = self.orb.detectAndCompute(img, None)
            return keypoints, descriptions

    def add_3d_landmarks(
        self, keypoints, descriptions, depth
    ):  # function for adding new points with unseen before features
        assert len(keypoints) == len(
            descriptions
        ), "Keypoints and descriptions must have the same length"
        for point, ds in zip(keypoints, descriptions):
            x_px, y_px = map(int, point.pt)
            if depth[y_px, x_px] == 0:
                continue
            node = self.graph.add_node_landmark_3d(np.zeros(3))
            self.detected_features[tuple(ds)] = node
            coords = point2xyz(np.array([x_px, y_px]), depth).to_np_array()
            self.graph.add_factor_1pose_1landmark_3d(
                coords, self.poses_id[-1], node, self.W_obs
            )

    def update_3d_landmarks(self, keypoints, descriptions, depth):
        assert len(keypoints) == len(
            descriptions
        ), "Keypoints and descriptions must have the same length"
        for point, ds in zip(keypoints, descriptions):
            x_px, y_px = map(int, point.pt)
            if depth[y_px, x_px] == 0:
                continue
            coords = point2xyz(np.array([x_px, y_px]), depth).to_np_array()
            self.graph.add_factor_1pose_1landmark_3d(
                coords, self.poses_id[-1], self.detected_features[tuple(ds)], self.W_obs
            )

    def step(self, img, depth):
        node = self.graph.add_node_pose_3d(mrob.geometry.SE3())
        self.poses_id.append(node)

        self.graph.add_factor_2poses_3d(
            mrob.geometry.SE3(),
            self.poses_id[-2],
            self.poses_id[-1],
            1e1 * np.identity(6),
        )
        ###add also odometry
        self.process_image_and_depth(img, depth)
        pass

    def step_odom(self, step_num, img, depth):
        node = self.graph.add_node_pose_3d(mrob.geometry.SE3())
        self.poses_id.append(node)
        trans_mat = get_translation_matrix(
            get_observation(step_num), get_observation(step_num - 1)
        )
        # trans_mat = np.eye(4)
        # trans_mat[0, 3] = 0.1e-2
        # trans_mat[1, 3] = 0.2e-2
        # trans_mat[2, 3] = 0.1e-2
        # trans_mat[:3, :3] = np.eye(3)
        print(trans_mat)
        self.graph.add_factor_2poses_3d(
            mrob.geometry.SE3(trans_mat),
            self.poses_id[-2],
            self.poses_id[-1],
            10e2 * np.identity(6),
        )
        # self.process_image_and_depth(img, depth)

    def process_image_and_depth(self, img, depth):
        kp, ds = self.get_img_features(img)

        features_list = list(self.detected_features.keys())
        features_arr = np.asarray(features_list)
        matches = self.bf.knnMatch(ds, features_arr, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append([m])

        similarity_mask = np.zeros_like(kp, dtype=bool)
        matched_idxs = []
        updated_features = []
        for match in good:
            similarity_mask[match[0].queryIdx] = True
            updated_features.append(features_list[match[0].trainIdx])
        _, counts = np.unique(similarity_mask, return_counts=True)
        kp_arr = np.asarray(kp)
        ds_arr = np.asarray(ds)
        updated_kp = kp_arr[similarity_mask]
        new_kp = kp_arr[np.logical_not(similarity_mask)]
        new_ds = ds_arr[np.logical_not(similarity_mask)]
        self.add_3d_landmarks(new_kp, new_ds, depth)
        self.update_3d_landmarks(updated_kp, updated_features, depth)

    def get_landmarks_coords(self):
        """Returns array with landmark poses

        Returns:
            _type_: _description_
        """
        est_states = self.graph.get_estimated_state()
        points = np.empty([0, 3], dtype=float)
        for id in self.detected_features.values():
            print(est_states[id])
            points = np.append(points, est_states[id].T, axis=0)

        return points


observation = get_observation(0)
# print(observation.image[0, 0])
# cv2.imshow("dad", observation.depth)
# cv2.waitKey(0)

pose = np.array([1.3452, 0.6273, 1.6627, 0.6582, 0.6109, -0.295, -0.3265])
init_mat = np.eye(4)
# init_mat[:3, :3] = quaternion_rotation_matrix(pose[3:])
rot_mat = rotation_matrix(np.pi, 10 / 180 * np.pi, 0)
# print(rot_mat)
init_mat[:3, :3] = rot_mat
init_mat[:3, 3] = pose[0:3]
print(init_mat)
graph_slam = GraphSlam(observation.image, observation.depth, init_mat)
print("Amount of descriptions in dictionary is", len(graph_slam.detected_features))
# cv2.imshow("dad", observation.image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
chi2 = []


num_steps = 60
for i in range(1, num_steps):
    graph_slam.step_odom(i, get_observation(i).image, get_observation(i).depth)
    print("Processing step", i)
    # cv2.imshow("dad", observation.image)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()

    # chi2.append(graph_slam.graph.chi2())

graph_slam.graph.solve(mrob.LM)
import matplotlib.pyplot as plt
import matplotlib

# print(graph_slam.poses_id)

matplotlib.use("TkAgg")
# plt.plot(np.arange(len(chi2)), chi2)
# plt.show()


est_states = graph_slam.graph.get_estimated_state()
# print(est_states)
est_trajectory = np.zeros([0, 4, 4])
for el in graph_slam.poses_id:
    est_trajectory = np.append(est_trajectory, [est_states[el]], axis=0)


from tools.path_plotter import plot_gt_and_est

ax = plt.axes(projection="3d")

plot_gt_and_est(ax, est_trajectory, steps=num_steps)
set_axes_equal(ax)
plt.show()

gt_mat = get_gt_mat()

# plotting error
fig, axes = plt.subplots(3, figsize=(10, 3))
titles = ["X error", "Y error", "Z error"]
ylabel = ["m", "m", "m"]
for i, ax in enumerate(axes):
    if i < 3:
        err = gt_mat[:num_steps, i, 3] - est_trajectory[:num_steps, i, 3]
        err = err.reshape(-1)
        ax.plot(np.arange(num_steps), err)
        ax.grid("on")

        ax.plot(np.arange(num_steps), err)
        ax.grid("on")
    ax.set_title(titles[i])
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel[i])
# fig.tight_layout(pad=5.0)
fig.subplots_adjust(hspace=0.5)
plt.show()


# plotting angular error
from scipy.spatial.transform import Rotation as R

fig, axes = plt.subplots(4)
err_mat = []
err_norm = []
angles_err = []
for j in range(num_steps):
    err_mat.append((np.linalg.inv(gt_mat[j, :3, :3]).dot(est_trajectory[j, :3, :3])))
    err_norm.append(np.linalg.norm(err_mat[-1]))
    angles_err.append(R.from_matrix(err_mat[j]).as_euler("xyz", degrees=True))
angles_err = np.asarray(angles_err)
titles = ["Norm of orientation error", "ROLL error", "PITCH error", "YAW error"]
ylabs = ["Rad", "Deg", "Deg", "Deg"]
for i, ax in enumerate(axes):
    if i == 0:
        ax.plot(np.arange(num_steps), err_norm)
    else:
        ax.plot(
            np.arange(num_steps), angles_err[:num_steps, i - 1] - angles_err[0, i - 1]
        )
    ax.set_title(titles[i])
    ax.set_xlabel("step")
    ax.set_ylabel(ylabs[i])
    ax.grid("on")

fig.subplots_adjust(hspace=0.5)
plt.show()
# plotting planes
fig, axes = plt.subplots(1, 3)
titles = ["XY trajectory", "YZ trajectory", "XZ trajectory"]

for i, ax in enumerate(axes):
    if i == 0:
        x = 0
        y = 1
        xlab = "X"
        ylab = "Y"
    elif i == 2:
        x = 1
        y = 2
        xlab = "Y"
        ylab = "Z"
    else:
        x = 0
        y = 2
        xlab = "X"
        ylab = "Z"
    ax.plot(gt_mat[:num_steps, x, 3], gt_mat[:num_steps, y, 3], label="Ground Truth")
    ax.plot(
        est_trajectory[:num_steps, x, 3],
        est_trajectory[:num_steps, y, 3],
        label="Estimated Trajectory",
    )
    ax.set_xlabel(ylab + ", m")
    ax.set_ylabel(xlab + ", m")
    ax.set_title(titles[i])
    ax.legend()
    ax.axis("equal")

plt.show()
