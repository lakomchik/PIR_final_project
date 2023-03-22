import dataclasses
import numpy as np
from tools.camera_params import CameraParams
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
    point_coords: np.ndarray, depth: np.ndarray, factor: int = 40
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
    return Position(-y, -x, z)


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
    # q0 = Q[0]
    # q1 = Q[1]
    # q2 = Q[2]
    # q3 = Q[3]
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


def rotation_matrix(theta1, theta2, theta3, order="xyz"):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        oreder = rotation order of x,y,z　e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    c3 = np.cos(theta3)
    s3 = np.sin(theta3)

    if order == "xzx":
        matrix = np.array(
            [
                [c2, -c3 * s2, s2 * s3],
                [c1 * s2, c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3],
                [s1 * s2, c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3],
            ]
        )
    elif order == "xyx":
        matrix = np.array(
            [
                [c2, s2 * s3, c3 * s2],
                [s1 * s2, c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1],
                [-c1 * s2, c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3],
            ]
        )
    elif order == "yxy":
        matrix = np.array(
            [
                [c1 * c3 - c2 * s1 * s3, s1 * s2, c1 * s3 + c2 * c3 * s1],
                [s2 * s3, c2, -c3 * s2],
                [-c3 * s1 - c1 * c2 * s3, c1 * s2, c1 * c2 * c3 - s1 * s3],
            ]
        )
    elif order == "yzy":
        matrix = np.array(
            [
                [c1 * c2 * c3 - s1 * s3, -c1 * s2, c3 * s1 + c1 * c2 * s3],
                [c3 * s2, c2, s2 * s3],
                [-c1 * s3 - c2 * c3 * s1, s1 * s2, c1 * c3 - c2 * s1 * s3],
            ]
        )
    elif order == "zyz":
        matrix = np.array(
            [
                [c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                [-c3 * s2, s2 * s3, c2],
            ]
        )
    elif order == "zxz":
        matrix = np.array(
            [
                [c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2],
                [c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
                [s2 * s3, c3 * s2, c2],
            ]
        )
    elif order == "xyz":
        matrix = np.array(
            [
                [c2 * c3, -c2 * s3, s2],
                [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2],
            ]
        )
    elif order == "xzy":
        matrix = np.array(
            [
                [c2 * c3, -s2, c2 * s3],
                [s1 * s3 + c1 * c3 * s2, c1 * c2, c1 * s2 * s3 - c3 * s1],
                [c3 * s1 * s2 - c1 * s3, c2 * s1, c1 * c3 + s1 * s2 * s3],
            ]
        )
    elif order == "yxz":
        matrix = np.array(
            [
                [c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3, c2 * s1],
                [c2 * s3, c2 * c3, -s2],
                [c1 * s2 * s3 - c3 * s1, c1 * c3 * s2 + s1 * s3, c1 * c2],
            ]
        )
    elif order == "yzx":
        matrix = np.array(
            [
                [c1 * c2, s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3],
                [s2, c2 * c3, -c2 * s3],
                [-c2 * s1, c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3],
            ]
        )
    elif order == "zyx":
        matrix = np.array(
            [
                [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
                [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
                [-s2, c2 * s3, c2 * c3],
            ]
        )
    elif order == "zxy":
        matrix = np.array(
            [
                [c1 * c3 - s1 * s2 * s3, -c2 * s1, c1 * s3 + c3 * s1 * s2],
                [c3 * s1 + c1 * s2 * s3, c1 * c2, s1 * s3 - c1 * c3 * s2],
                [-c2 * s3, s2, c2 * c3],
            ]
        )

    return matrix
