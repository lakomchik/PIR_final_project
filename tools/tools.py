import dataclasses
import numpy as np
from tools.camera_params import CameraParams

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
