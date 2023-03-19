import numpy as np
from tools.camera_params import CameraParams

cam_params = CameraParams()


def point2xyz(point_coords, depth):
    """Returns xyz point coords calculated by depth map

    Args:
        point_coords (1d np.array): xy point pixel coordinates
        depth (cv2 Image): Image depth map

    Returns:
        1d np.array: xyz coordinates or None in case if no depth available
    """
    x_pix = point_coords.reshape(-1)[0]
    y_pix = point_coords.reshape(-1)[1]
    height, width = depth.shape
    assert x_pix < width and y_pix < height, "Point coordinates are out of Range!"
    factor = 5000  # for the 16-bit PNG files
    if (depth[y_pix][x_pix]) == 0:
        return None
    z = depth[y_pix][x_pix] / factor
    x = (x_pix - cam_params.cx) * z / cam_params.fx
    y = (y_pix - cam_params.cy) * z / cam_params.fy
    return np.array([x, y, z])
