import linecache

dataset_path = "/home/lakomchik/sata/slam_dataset/rgbd_dataset_freiburg1_xyz/"

accelerometer_name = "accelerometer.txt"
depth_name = "depth.txt"
rgb_name = "rgb.txt"
gt_name = "groundtruth.txt"

import cv2

initial_offset = 4


def get_observations(idx):
    """Provides timestamp, rgb_image and depth image by step

    Args:
        idx (int): step number

    Returns:
        float, cv_img, cv_img: timestamp, rgb_img, depth_img
    """
    timestamp, img_loc = linecache.getline(
        dataset_path + rgb_name, initial_offset + idx
    ).split()
    _, depth_loc = linecache.getline(
        dataset_path + depth_name, initial_offset + idx
    ).split()
    img = cv2.imread(dataset_path + img_loc)
    depth = cv2.imread(dataset_path + depth_loc)
    timestamp = float(timestamp)
    return timestamp, img, depth
