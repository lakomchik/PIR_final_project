import numpy as np
import cv2
import mrob


from tools.data_reader import get_observation
from tools.tools import point2xyz


class VisualOdometry:
    def __init__(self) -> None:
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
            x_px, y_px = map(int, point.kp)
            if depth[y_px, x_px] == 0:
                continue
            node = self.graph.add_node_landmark_3d(np.zeros(3))
            self.detected_features[ds] = node
            coords = point2xyz([x_px, y_px], depth).to_np_array()
            W = np.identity(3)
            self.graph.add_factor_1pose_1landmark_3d(coords, self.poses_id[-1], node, W)

    def observe_3d_landmarks(self, keypoints, descriptions, depth):
        pass

    # def compare_features(self, img):
    #     kp, ds = self.get_img_features(img)
    #     matches = self.bf.knnMatch(np.asarray(list(self.detected_features.keys())), kp)
    #     # Apply ratio test
    #     good = []
    #     for m, n in matches:
    #         if m.distance < 0.3 * n.distance:
    #             good.append([m])


observation = get_observation(1)
cv2.imshow("dad", observation.image)
cv2.waitKey(0)
cv2.destroyAllWindows()
