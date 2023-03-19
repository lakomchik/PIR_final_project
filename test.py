import cv2
from matplotlib import pyplot as plt

img = cv2.imread("/home/lakomchik/sata/slam_dataset/train/0/50/depth/0.png")
cv2.imshow("1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
