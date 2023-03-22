import open3d as o3d
import numpy as np

from tools.camera_params import CameraParams
from tools.data_reader import Observation


def get_translation_matrix(obs_start: Observation, obs_goal: Observation) -> np.ndarray:
    camera_params = CameraParams()
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    pinhole_camera_intrinsic.set_intrinsics(
        640, 480, camera_params.fx, camera_params.fy, camera_params.cx, camera_params.cy
    )
    start_color = o3d.geometry.Image(obs_start.image)
    start_depth = o3d.geometry.Image(obs_start.depth)
    goal_color = o3d.geometry.Image(obs_goal.image)
    goal_depth = o3d.geometry.Image(obs_goal.depth)
    start_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        start_color, start_depth, depth_scale=0.05
    )
    goal_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        goal_color, goal_depth, depth_scale=0.05
    )

    option = o3d.pipelines.odometry.OdometryOption()
    odom_init = np.identity(4)

    _, trans_hybrid_term, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
        start_rgbd,
        goal_rgbd,
        pinhole_camera_intrinsic,
        odom_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
        option,
    )

    return trans_hybrid_term
