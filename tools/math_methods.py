import dataclasses
import numpy as np
from camera_params import CameraParams
import math

cam_params = CameraParams()


@dataclasses.dataclass
class Position:
    x: float
    y: float
    z: float

    def to_np_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


def point2xyz(
    point_coords: np.ndarray, depth: np.ndarray, factor: int = 5000
) -> Position | None:
    """
    Calculate XYZ point coordinates from the depth map

    Args:
        point_coords (np.ndarray): XY point coordinates on pixelmap
        depth (np.ndarray): Image depth map

    Returns:
        Position | None: XYZ coordinates or None in case if no depth available
    """
    assert point_coords.shape == (
        2,
    ), f"Wrong shape of point_coords: {point_coords.shape}"
    x_pix = point_coords.reshape(-1)[0]
    y_pix = point_coords.reshape(-1)[1]
    height, width = depth.shape
    assert (
        x_pix < width and y_pix < height
    ), f"Point coordinates ({x_pix}, {y_pix}) are out of range!"

    if (depth[y_pix][x_pix]) == 0:
        return None
    z = depth[y_pix][x_pix] / factor
    x = (x_pix - cam_params.cx) * z / cam_params.fx
    y = (y_pix - cam_params.cy) * z / cam_params.fy
    return Position(x, y, z)


def quaternion_to_euler(input_quat):
    x, y, z, w = input_quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return np.array([X, Y, Z])


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

    return rot_matrix
