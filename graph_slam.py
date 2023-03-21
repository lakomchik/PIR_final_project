import numpy as np
import cv2
import mrob


from tools.data_reader import get_observation
from tools.tools import point2xyz
from tools.tools import quaternion_rotation_matrix
import math


class GraphSlam:
    def __init__(self, img, depth, init_mat) -> None:
        self.graph = mrob.FGraph()
        self.detected_features = (
            {}
        )  # dictionary of memorized features key: orb feature; value: idx in graph
        self.poses_id = []  # idx in graph for poses
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher()
        self.W_obs = np.eye(3) * 10e5
        n1 = self.graph.add_node_pose_3d(mrob.geometry.SE3())
        self.poses_id.append(n1)
        self.W_0 = 1e10 * np.identity(6)  # covariation of pose
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
            1e6 * np.identity(6),
        )
        self.process_image_and_depth(img, depth)
        pass

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


observation = get_observation(0)
# print(observation.image[0, 0])
# cv2.imshow("dad", observation.depth)
# cv2.waitKey(0)

pose = np.array([1.2788, 0.5815, 1.4563, 0.6652, 0.651, -0.2817, -0.2332])
init_mat = np.eye(4)
init_mat[:3, :3] = quaternion_rotation_matrix(pose[3:])
init_mat[:3, 3] = pose[0:3]
print(init_mat)
# init_pose = np.zeros(6)
# init_pose[0:3] = pose[0:3]
# init_pose[3:] = quaternion_to_euler(pose[3:])
# print(init_pose)
mrob.geometry.SE3(init_mat).print()
graph_slam = GraphSlam(observation.image, observation.depth, init_mat)
print("Amount of descriptions in dictionary is", len(graph_slam.detected_features))
cv2.imshow("dad", observation.image)
cv2.waitKey(0)
cv2.destroyAllWindows()
chi2 = []
for i in range(1, 10):
    observation = get_observation(i)
    # cv2.imshow("dad", observation.image)
    # cv2.waitKey(10)
    # cv2.destroyAllWindows()
    # print(len(graph_slam.detected_features))
    graph_slam.step(observation.image, observation.depth)
    print("Amount of descriptions in dictionary is", len(graph_slam.detected_features))
    graph_slam.graph.solve(mrob.LM)
    chi2.append(graph_slam.graph.chi2())


import matplotlib.pyplot as plt
import matplotlib

print(graph_slam.poses_id)
matplotlib.use("TkAgg")
plt.plot(np.arange(len(chi2)), chi2)
plt.show()


est_states = graph_slam.graph.get_estimated_state()
for el in graph_slam.poses_id:
    print(est_states[el])
