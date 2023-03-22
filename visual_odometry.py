import sys
import numpy as np
import cv2 as cv
import random as rand
import dataclasses
from tools.data_reader import get_observation

from typing import List, Tuple, Dict
from tools.camera_params import CameraParams


class Cells:
    def __init__(self):
        self.pts = []
        self.pairs = {}

    def rand_pt(self) -> Tuple | None:
        try:
            res = rand.choice(self.pts)
            return res
        except IndexError:
            return None


@dataclasses.dataclass
class CameraOffsets:
    R: Dict[int, np.ndarray]
    C: Dict[int, np.ndarray]

    def make_positive_det(self) -> None:
        for i in self.R.keys():
            if np.linalg.det(self.R[i]) < 0:
                self.R[i] *= -1
                self.C[i] *= -1


@dataclasses.dataclass
class Measurement:
    omega: np.ndarray
    v: np.ndarray

    def to_np_array(self):
        return np.array(
            [
                self.omega[0],
                self.omega[1],
                self.omega[2],
                self.v[0],
                self.v[1],
                self.v[2],
            ]
        )


class VisualOdometry:
    @staticmethod
    def get_rand8(grid: np.ndarray) -> Tuple[np.ndarray]:
        """
        Get random 8 points from different regions in a Image using Zhang's 8x8 Grid

        Args:
            grid (np.ndarray): Zhang's Grid

        Returns:
            Tuple[np.ndarray]: Point indexes in grid and corresponding points
        """
        cells = []
        for i in range(8):
            for j in range(8):
                cells.append((i, j))
        rand_grid_index = rand.choices(cells, k=8)
        rand8 = []
        rand8_ = []
        for index in rand_grid_index:
            if grid[index].pts:
                pt = grid[index].rand_pt()
                rand8.append(pt)
            else:
                index = rand.choice(cells)
                while not grid[index].pts or index in rand_grid_index:
                    index = rand.choice(cells)
                pt = grid[index].rand_pt()
                rand8.append(pt)

            # -----> find the correspondence given point <----- #
            rand8_.append(grid[index].pairs[pt])
        return np.array(rand8), np.array(rand8_)

    @staticmethod
    def calculate_fundamental_matrix(
        pts_cf: np.ndarray, pts_nf: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Fundamental Matrix for the given points from RANSAC

        Args:
            pts_cf (np.ndarray): Point indexes in grid
            pts_nf (np.ndarray): Corresponding points

        Returns:
            np.ndarray: Calculated fundamental matrix
        """
        origin = [0.0, 0.0]
        origin_ = [0.0, 0.0]
        origin = np.mean(pts_cf, axis=0)
        origin_ = np.mean(pts_nf, axis=0)
        k = np.mean(np.sum((pts_cf - origin) ** 2, axis=1, keepdims=True) ** 0.5)
        k_ = np.mean(np.sum((pts_nf - origin_) ** 2, axis=1, keepdims=True) ** 0.5)
        k = np.sqrt(2.0) / k
        k_ = np.sqrt(2.0) / k_
        x = (pts_cf[:, 0].reshape((-1, 1)) - origin[0]) * k
        y = (pts_cf[:, 1].reshape((-1, 1)) - origin[1]) * k
        x_ = (pts_nf[:, 0].reshape((-1, 1)) - origin_[0]) * k_
        y_ = (pts_nf[:, 1].reshape((-1, 1)) - origin_[1]) * k_
        A = np.hstack(
            (x_ * x, x_ * y, x_, y_ * x, y_ * y, y_, x, y, np.ones((len(x), 1)))
        )
        U, S, V = np.linalg.svd(A)
        F = V[-1]
        F = np.reshape(F, (3, 3))
        U, S, V = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ V
        T1 = np.array([[k, 0, -k * origin[0]], [0, k, -k * origin[1]], [0, 0, 1]])
        T2 = np.array([[k_, 0, -k_ * origin_[0]], [0, k_, -k_ * origin_[1]], [0, 0, 1]])
        F = T2.T @ F @ T1
        F = F / F[-1, -1]
        return F

    @staticmethod
    def estimate_fundamental_matrix_RANSAC(
        pts1: np.ndarray,
        pts2: np.ndarray,
        grid: np.ndarray,
        epsilon: float = 0.05,
    ) -> np.ndarray:
        """
        Estimate Fundamental Matrix from the given correspondences using RANSAC

        Args:
            pts1 (np.ndarray): Point indexes in grid
            pts2 (np.ndarray): Corresponding points
            grid (np.ndarray): Zhang's Grid
            epsilon (float, optional): Error threshold. Defaults to 0.05.

        Returns:
            np.ndarray: Estimated fundamental matrix
        """
        max_inliers = 0
        F_best = []
        confidence = 0.99
        N = sys.maxsize
        count = 0
        while N > count:
            counter = 0
            x_1, x_2 = VisualOdometry.get_rand8(grid)
            F = VisualOdometry.calculate_fundamental_matrix(x_1, x_2)
            ones = np.ones((len(pts1), 1))
            x = np.hstack((pts1, ones))
            x_ = np.hstack((pts2, ones))
            e, e_ = x @ F.T, x_ @ F
            error = np.sum(e_ * x, axis=1, keepdims=True) ** 2 / np.sum(
                np.hstack((e[:, :-1], e_[:, :-1])) ** 2, axis=1, keepdims=True
            )
            inliers = error <= epsilon
            counter = np.sum(inliers)
            if max_inliers < counter:
                max_inliers = counter
                F_best = F
            I_O_ratio = counter / len(pts1)
            if np.log(1 - (I_O_ratio**8)) == 0:
                continue
            N = np.log(1 - confidence) / np.log(1 - (I_O_ratio**8))
            count += 1
        return F_best

    @staticmethod
    def estimate_essential_matrix(K: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Estimate essential matrix

        Args:
            K (np.ndarray): Camera calibration matrix
            F (np.ndarray): Fundamental matrix

        Returns:
            np.ndarray: Estimated essential matrix
        """
        E = K.T @ F @ K
        U, S, V = np.linalg.svd(E)
        S = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
        E = U @ S @ V
        return E

    @staticmethod
    def linear_triangulation(
        K: np.ndarray,
        C1: np.ndarray,
        R1: np.ndarray,
        C2: np.ndarray,
        R2: np.ndarray,
        pt: np.ndarray,
        pt_: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Calculate Linear Triangulation

        Args:
            K (np.ndarray): Camera calibration matrix
            C1 (np.ndarray): Translation vector for initial position
            R1 (np.ndarray): Rotation matrix for initial position
            C2 (np.ndarray): Translation vector for final position
            R2 (np.ndarray): Rotation matrix for final position
            pt (np.ndarray): Point indexes in grid
            pt_ (np.ndarray): Corresponding points

        Returns:
            List[np.ndarray]: _description_
        """
        P1 = K @ np.hstack((R1, -R1 @ C1))
        P2 = K @ np.hstack((R2, -R2 @ C2))
        X = []
        for i in range(len(pt)):
            x1 = pt[i]
            x2 = pt_[i]
            A1 = x1[0] * P1[2, :] - P1[0, :]
            A2 = x1[1] * P1[2, :] - P1[1, :]
            A3 = x2[0] * P2[2, :] - P2[0, :]
            A4 = x2[1] * P2[2, :] - P2[1, :]
            A = [A1, A2, A3, A4]
            _, _, V = np.linalg.svd(A)
            V = V[3]
            V = V / V[-1]
            X.append(V)
        return X

    @staticmethod
    def get_camera_offsets(E: np.array) -> CameraOffsets:
        """
        Estimate the camera Pose

        Args:
            K (np.array): Camera calibration matrix
            E (np.array): Essential matrix

        Returns:
            CameraOffsets: Camera offsets
        """
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        U, _, V = np.linalg.svd(E)
        camera_offsets = CameraOffsets(
            R={
                1: U @ W @ V,
                2: U @ W @ V,
                3: U @ W.T @ V,
                4: U @ W.T @ V,
            },
            C={
                1: U[:, 2].reshape(3, 1),
                2: -U[:, 2].reshape(3, 1),
                3: U[:, 2].reshape(3, 1),
                4: -U[:, 2].reshape(3, 1),
            },
        )
        camera_offsets.make_positive_det()
        return camera_offsets

    @staticmethod
    def extract_rot_and_trans(
        R: np.ndarray,
        T: np.ndarray,
        pt: np.ndarray,
        pt_: np.ndarray,
        K: np.ndarray,
    ) -> int:
        """
        Find the rotation and translation parameters

        Args:
            R (np.ndarray): Rotation matrix from camera offsets
            T (np.ndarray): Translation vector from camera offsets
            pt (np.ndarray): Point indexes in grid
            pt_ (np.ndarray): Corresponding points
            K (np.ndarray): Camera calibration matrix

        Returns:
            int: _description_
        """
        C = [[0], [0], [0]]
        R = np.eye(3, 3)
        X1 = VisualOdometry.linear_triangulation(K, C, R, T, R, pt, pt_)
        X1 = np.array(X1)
        count = 0
        for i in range(X1.shape[0]):
            x = X1[i, :].reshape(-1, 1)
            if R[2] @ (x[0:3] - T) > 0 and x[2] > 0:
                count += 1
        return count


def get_measurement(
    key_frame1: np.ndarray,
    key_frame2: np.ndarray,
    time_stamp: float,
) -> Measurement:
    """
    Measure velocities between two frames

    Args:
        time_stamp (int): Timestamp
        key_frame1 (np.ndarray): Image of initial position
        key_frame2 (np.ndarray): Image of final position

    Returns:
        Measurement: Linear and angular velocities
    """
    frame1 = key_frame1.copy()
    frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    img_dim = key_frame1.shape
    y_bar, x_bar = np.array(img_dim[:-1]) / 8

    frame2 = key_frame2.copy()
    frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    sift = cv.xfeatures2d.SIFT_create()
    kp_cf, des_current = sift.detectAndCompute(frame1, None)
    kp_nf, des_next = sift.detectAndCompute(frame2, None)

    best_matches = []
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_current, des_next, k=2)
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            best_matches.append(m)

    point_correspondence_cf = np.zeros((len(best_matches), 2))
    point_correspondence_nf = np.zeros((len(best_matches), 2))
    grid = np.empty((8, 8), dtype=object)
    grid[:, :] = Cells()

    for i, match in enumerate(best_matches):
        j = int(kp_cf[match.queryIdx].pt[0] / x_bar)
        k = int(kp_cf[match.queryIdx].pt[1] / y_bar)
        grid[j, k].pts.append(kp_cf[match.queryIdx].pt)
        grid[j, k].pairs[kp_cf[match.queryIdx].pt] = kp_nf[match.trainIdx].pt

        point_correspondence_cf[i] = (
            kp_cf[match.queryIdx].pt[0],
            kp_cf[match.queryIdx].pt[1],
        )
        point_correspondence_nf[i] = (
            kp_nf[match.trainIdx].pt[0],
            kp_nf[match.trainIdx].pt[1],
        )
    K = CameraParams().calibration_matrix()

    F = VisualOdometry.estimate_fundamental_matrix_RANSAC(
        pts1=point_correspondence_cf,
        pts2=point_correspondence_nf,
        grid=grid,
        epsilon=0.05,
    )
    E = VisualOdometry.estimate_essential_matrix(K, F)
    camera_offsets = VisualOdometry.get_camera_offsets(E)

    flag = 0
    for p in camera_offsets.R.keys():
        R = camera_offsets.R[p]
        T = camera_offsets.C[p]
        Z = VisualOdometry.extract_rot_and_trans(
            R,
            T,
            point_correspondence_cf,
            point_correspondence_nf,
            K,
        )
        if flag < Z:
            flag, reg = Z, p

    R = camera_offsets.R[reg]
    (rvec, _) = cv.Rodrigues(R)
    tvec = camera_offsets.C[reg]
    v = tvec / time_stamp
    omega = rvec / time_stamp
    print(omega)
    print(v)
    return Measurement(omega, v)


old_frame = []


frame1 = cv.imread("datasets/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png")
frame2 = cv.imread("datasets/rgbd_dataset_freiburg1_xyz/rgb/1305031102.211214.png")
frame_1 = get_observation(0).image
for i in range(1, 20):
    frame2 = get_observation(i).image
    print("ON iteration ", i)
    get_measurement(frame2, frame2, 0.033)
    frame1 = frame2.copy()
