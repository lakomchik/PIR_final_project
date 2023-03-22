import cv2
from matplotlib import pyplot as plt
import mrob
import numpy as np
import math
import matplotlib

matplotlib.use("TkAgg")
img_1 = cv2.imread("datasets/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png")
depth_1 = cv2.imread(
    "datasets/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png",
    cv2.IMREAD_GRAYSCALE,
)
img_2 = cv2.imread("datasets/rgbd_dataset_freiburg1_xyz/rgb/1305031102.211214.png")


FX_DEPTH = 525.0  # focal length x
FY_DEPTH = 525.0  # focal length y
CX_DEPTH = 319.5  # optical center x
CY_DEPTH = 239.5  # optical center y
# compute point cloud:
factor = 5000  # for the 16-bit PNG files
pcd = []
height, width = depth_1.shape
for i in range(height):
    for j in range(width):
        if depth_1[i][j] == 0:
            continue
        z = depth_1[i][j] / factor
        x = (j - CX_DEPTH) * z / FX_DEPTH
        y = (i - CY_DEPTH) * z / FY_DEPTH
        pcd.append([x, -y, -z])

pcd = np.asarray(pcd)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
step = 30
ax.scatter(pcd[::step, 0], pcd[::step, 1], pcd[::step, 2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
