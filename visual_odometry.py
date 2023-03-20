import numpy as np
import cv2
import mrob


from tools.data_reader import get_observation
from tools.tools import point2xyz


class VisualOdometry:
    def __init__(self, img, depth) -> None:
        self.graph = mrob.FGraph()
        self.detected_features = (
            {}
        )  # dictionary of memorized features key: orb feature; value: idx in graph
        self.poses_id = []  # idx in graph for poses
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher()
        xi = np.array(
            [-0.0282668, -0.05882867, 0.0945925, -0.02430618, 0.01794402, -0.06549129]
        )  # initial pose TODO: ADD INITIAL POSE
        n1 = self.graph.add_node_pose_3d(mrob.geometry.SE3(xi))
        self.poses_id.append(n1)
        self.W_0 = 1e6 * np.identity(6)  # covariation of pose
        self.graph.add_factor_1pose_3d(mrob.geometry.SE3(), self.poses_id[-1], self.W_0)
        kp, ds = self.get_img_features(img)
        self.add_3d_landmarks(kp, ds, depth)
        pass

    def add_visual_landmark(self, rgb_img, depth_img, timestamp=0.0):
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
            # print(ds)
            self.detected_features[tuple(ds)] = node
            coords = point2xyz(np.array([x_px, y_px]), depth).to_np_array()
            W = np.identity(3)
            self.graph.add_factor_1pose_1landmark_3d(coords, self.poses_id[-1], node, W)

    def update_3d_landmarks(self, keypoints, descriptions, depth):
        assert len(keypoints) == len(
            descriptions
        ), "Keypoints and descriptions must have the same length"
        for point, ds in zip(keypoints, descriptions):
            x_px, y_px = map(int, point.pt)
            if depth[y_px, x_px] == 0:
                continue
            coords = point2xyz(np.array([x_px, y_px]), depth).to_np_array()
            W = np.identity(3)
            self.graph.add_factor_1pose_1landmark_3d(
                coords, self.poses_id[-1], self.detected_features[ds], W
            )

    def step(self, img, depth):
        # self.graph.add
        pass

    def process_image_and_depth(self, img, depth):
        kp, ds = self.get_img_features(img)
        features_list = list(self.detected_features.keys())
        matches = self.bf.knnMatch(features_list, kp)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.3 * n.distance:
                good.append([m])

        similarity_mask = np.zeros_like(kp, dtype=bool)
        matched_idxs = []
        updated_features = []
        for match in good:
            similarity_mask[match[0].trainIdx] = True
            updated_features.append(features_list[match[0].queryIdx])
        kp_arr = np.asarray(kp)
        ds_arr = np.asarray(ds)
        updated_kp = kp_arr[similarity_mask]
        new_kp = kp_arr[np.logical_not(similarity_mask)]
        new_ds = ds_arr[np.logical_not(similarity_mask)]
        self.add_3d_landmarks(new_kp, new_ds, depth)
        self.update_3d_landmarks(updated_kp, updated_features)


observation = get_observation(1)
print(observation.depth[0, 0])
cv2.imshow("dad", observation.depth)
cv2.waitKey(0)
vis_odom = VisualOdometry(observation.image, observation.depth)
cv2.imshow("dad", observation.image)
cv2.waitKey(0)
cv2.destroyAllWindows()
