class CameraParams:
    """Camera intrinsic parameters"""

    def __init__(self) -> None:
        # Camera intrinsics; Obtained from
        # https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
        self.fx = 525.0
        self.fy = 525.0
        self.cx = 319.5
        self.cy = 239.5
