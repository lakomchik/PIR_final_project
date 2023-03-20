import dataclasses
import linecache
import cv2
import numpy as np


@dataclasses.dataclass
class Observation:
    timestamp: float
    image: np.ndarray
    depth: np.ndarray


def get_filepath(folder: str, filename: str) -> str:
    return f"{folder}/{filename}"


def get_observation(
    idx: int,
    dataset_folder: str = "datasets/rgbd_dataset_freiburg1_xyz",
    image_filename: str = "rgb.txt",
    depth_filename: str = "depth.txt",
    starting_offset: int = 4,
) -> Observation:
    """
    Gets an observation by step number

    Args:
        idx (int): Step number
        dataset_folder (str, optional): Path to dataset folder. Defaults to "datasets/rgbd_dataset_freiburg1_xyz"
        image_filename (str, optional): Name of file with image RGB data. Defaults to "depth.txt"
        depth_filename (str, optional): Name of file with image depth data. Defaults to "rgb.txt"
        starting_offset (int, optional): Index of initial observation. Defaults to 4

    Returns:
        Observation: Observation at the given step
    """
    timestamp, img_loc = linecache.getline(
        filename=get_filepath(dataset_folder, image_filename),
        lineno=starting_offset + idx,
    ).split()
    _, depth_loc = linecache.getline(
        filename=get_filepath(dataset_folder, depth_filename),
        lineno=starting_offset + idx,
    ).split()
    image = cv2.imread(get_filepath(dataset_folder, img_loc))
    depth = cv2.imread(get_filepath(dataset_folder, depth_loc))
    timestamp = float(timestamp)
    return Observation(timestamp, image, depth)
