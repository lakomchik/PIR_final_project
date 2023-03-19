import numpy as np
import cv2
import mrob


from tools.data_reader import get_observations


class VisualOdometry:
    def __init__(self) -> None:
        self.features = []  # list of memorized features
        self.graph = mrob.FGraph()
        self.detected_features = {}
        self.orb = cv2.ORB_create()
        pass

    def add_visual_landmark(self, rgb_img, depth_img, timestamp=0.0):
        pass

    def get_img_features(self, img, type="ORB"):
        if type == "ORB":
            keypoints, descriptions = self.orb.detectAndCompute(img, None)


stamp, img, depth = get_observations(1)
cv2.imshow("dad", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
