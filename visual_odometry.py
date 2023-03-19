import numpy as np
import cv2
import mrob


class VisualOdometry:
    def __init__(self) -> None:
        self.features = []  # list of memorized features
        self.graph = mrob.FGraph()
        pass

    # def
