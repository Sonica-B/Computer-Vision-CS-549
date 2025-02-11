import numpy as np
import cv2
from scipy.optimize import least_squares
from dataclasses import dataclass
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict


@dataclass
class CalibrationResult:
    """Stores calibration results"""
    K: np.ndarray  # Camera intrinsic matrix
    k: np.ndarray  # Distortion coefficients [k1, k2]
    R: List[np.ndarray]  # Rotation matrices
    t: List[np.ndarray]  # Translation vectors
    reprojection_error: float  # Overall RMS error
    per_view_errors: List[float]  # Per-image RMS errors


class CameraCalibration:
    def __init__(self, square_size: float = 21.5, pattern_size: Tuple[int, int] = (5, 8)):
        """
        Initialize camera calibration using Zhang's method
        Args:
            square_size: Size of checkerboard squares in mm
            pattern_size: Number of corners (rows, cols)
        """
        self.square_size = square_size
        self.pattern_size = pattern_size
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize camera parameters
        self.K = None
        self.k = np.zeros(2)  # [k1, k2] radial distortion
        self.R = []  # Rotation matrices
        self.t = []  # Translation vectors

    def create_object_points(self) -> np.ndarray:
        """Create 3D points for inner corners (3x6 grid)"""
        objp = np.zeros((3 * 6, 3), np.float32)
        # Create grid for inner corners (skipping outer squares)
        inner_points = np.mgrid[1:4, 1:7].T.reshape(-1, 2)
        objp[:, :2] = inner_points
        objp *= self.square_size
        return objp

    def find_corners(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """Find checkerboard corners using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # Enhance image for better corner detection
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Main pattern dimensions - looking for inner corners 6x7
        pattern_points = (7, 6)  # width x height

        # Find corners
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_points,
            flags=(cv2.CALIB_CB_ADAPTIVE_THRESH +
                   cv2.CALIB_CB_NORMALIZE_IMAGE +
                   cv2.CALIB_CB_FILTER_QUADS)
        )

        if ret and corners is not None and len(corners) > 0:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners, True

        return None, False

    def verify_corner_positions(self, corners: np.ndarray) -> bool:
        """Verify corners form a valid pattern"""
        if corners is None or len(corners) != 42:  # 6x7 inner corners
            return False

        try:
            # Reshape corners to grid
            grid = corners.reshape(6, 7, 1, 2)

            # Check row and column spacing consistency
            row_diffs = np.diff(grid, axis=0)
            col_diffs = np.diff(grid, axis=1)

            # Calculate distances between adjacent corners
            row_distances = np.linalg.norm(row_diffs.reshape(-1, 2), axis=1)
            col_distances = np.linalg.norm(col_diffs.reshape(-1, 2), axis=1)

            # Calculate statistics
            mean_row_dist = np.mean(row_distances)
            mean_col_dist = np.mean(col_distances)
            std_row = np.std(row_distances) / mean_row_dist
            std_col = np.std(col_distances) / mean_col_dist

            # Thresholds for validity
            max_std = 0.2  # Maximum allowed normalized standard deviation
            min_dist = 10  # Minimum distance between corners in pixels

            if (std_row > max_std or std_col > max_std or
                    mean_row_dist < min_dist or mean_col_dist < min_dist):
                return False

            return True

        except Exception:
            return False

    def estimate_homography(self, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """Estimate homography matrix using normalized DLT"""
        # Normalize points
        src_mean = np.mean(src_points, axis=0)
        dst_mean = np.mean(dst_points, axis=0)
        src_scale = np.sqrt(2) / np.std(src_points - src_mean)
        dst_scale = np.sqrt(2) / np.std(dst_points - dst_mean)

        T_src = np.array([
            [src_scale, 0, -src_scale * src_mean[0]],
            [0, src_scale, -src_scale * src_mean[1]],
            [0, 0, 1]
        ])

        T_dst = np.array([
            [dst_scale, 0, -dst_scale * dst_mean[0]],
            [0, dst_scale, -dst_scale * dst_mean[1]],
            [0, 0, 1]
        ])

        # Build homogeneous system of equations
        normalized_src = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), T_src).reshape(-1, 2)
        normalized_dst = cv2.perspectiveTransform(dst_points.reshape(-1, 1, 2), T_dst).reshape(-1, 2)

        A = []
        for (x, y), (u, v) in zip(normalized_src, normalized_dst):
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        A = np.array(A)

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H = np.linalg.inv(T_dst) @ H @ T_src

        return H / H[2, 2]

    def extract_intrinsics(self, homographies: List[np.ndarray]) -> np.ndarray:
        """Extract camera intrinsic matrix K from homographies"""
        # Build V matrix for solving linear system
        V = []
        for H in homographies:
            h1, h2 = H[:, 0], H[:, 1]
            V.append(self._create_v_row(h1, h2))
            V.append(self._create_v_row(h1, h1) - self._create_v_row(h2, h2))
        V = np.array(V)

        # Solve Vb = 0 using SVD
        _, _, Vt = np.linalg.svd(V)
        b = Vt[-1]
        B11, B12, B22, B13, B23, B33 = b

        # Extract intrinsic parameters
        v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 ** 2)
        lambda_ = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
        alpha = np.sqrt(lambda_ / B11)
        beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12 ** 2))
        gamma = -B12 * alpha ** 2 * beta / lambda_
        u0 = gamma * v0 / beta - B13 * alpha ** 2 / lambda_

        return np.array([[alpha, gamma, u0],
                         [0, beta, v0],
                         [0, 0, 1]])

    def _create_v_row(self, h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
        """Create a row of V matrix for intrinsic parameter extraction"""
        return np.array([
            h1[0] * h2[0], h1[0] * h2[1] + h1[1] * h2[0], h1[1] * h2[1],
            h1[2] * h2[0] + h1[0] * h2[2], h1[2] * h2[1] + h1[1] * h2[2], h1[2] * h2[2]
        ])

    def estimate_extrinsics(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate extrinsic parameters (R,t) from homography"""
        # Get rotation matrix columns
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        K_inv = np.linalg.inv(self.K)

        lambda_ = 1 / np.linalg.norm(K_inv @ h1)
        r1 = lambda_ * K_inv @ h1
        r2 = lambda_ * K_inv @ h2
        r3 = np.cross(r1, r2)

        R = np.column_stack([r1, r2, r3])

        # Ensure proper rotation matrix
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        t = lambda_ * K_inv @ h3

        return R, t.reshape(3, 1)

    def project_points(self, object_points: np.ndarray, R: np.ndarray,
                       t: np.ndarray, K: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Project 3D points to image plane with distortion"""
        # Transform points to camera coordinates
        points_cam = (R @ object_points.T + t).T

        # Project to normalized image coordinates
        points_norm = points_cam[:, :2] / points_cam[:, 2:]

        # Apply radial distortion
        r2 = np.sum(points_norm ** 2, axis=1)
        distortion = 1 + k[0] * r2 + k[1] * r2 ** 2
        points_dist = points_norm * distortion.reshape(-1, 1)

        # Apply camera matrix
        points_img = (K[:2, :2] @ points_dist.T + K[:2, 2:]).T

        return points_img

    def optimize_parameters(self, object_points: np.ndarray,
                            image_points: List[np.ndarray]) -> None:
        """Optimize all parameters using Levenberg-Marquardt"""

        def objective(params):
            n_views = len(image_points)
            K_vec = params[:4]
            k_vec = params[4:6]
            rt_vecs = params[6:].reshape(n_views, 6)

            # Reconstruct matrices
            K = np.array([[K_vec[0], 0, K_vec[2]],
                          [0, K_vec[1], K_vec[3]],
                          [0, 0, 1]])
            k = k_vec

            errors = []
            for i in range(n_views):
                R, _ = cv2.Rodrigues(rt_vecs[i, :3])
                t = rt_vecs[i, 3:].reshape(3, 1)
                projected = self.project_points(object_points, R, t, K, k)
                errors.extend((projected - image_points[i].reshape(-1, 2)).ravel())

            return errors

        # Initial parameter vector
        initial_params = []
        initial_params.extend([self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]])
        initial_params.extend(self.k)

        for R, t in zip(self.R, self.t):
            r, _ = cv2.Rodrigues(R)
            initial_params.extend(r.ravel())
            initial_params.extend(t.ravel())

        # Optimize
        result = least_squares(objective, initial_params, method='lm')

        # Update parameters
        params = result.x
        self.K = np.array([[params[0], 0, params[2]],
                           [0, params[1], params[3]],
                           [0, 0, 1]])
        self.k = params[4:6]

        n_views = len(self.R)
        rt_vecs = params[6:].reshape(n_views, 6)
        self.R = []
        self.t = []
        for i in range(n_views):
            R, _ = cv2.Rodrigues(rt_vecs[i, :3])
            t = rt_vecs[i, 3:].reshape(3, 1)
            self.R.append(R)
            self.t.append(t)

    def calibrate(self, images: List[np.ndarray]) -> CalibrationResult:
        """Perform complete camera calibration"""
        object_points = self.create_object_points()
        image_points = []
        valid_images = []

        # Find corners
        for img in images:
            corners, success = self.find_corners(img)
            if success:
                image_points.append(corners)
                valid_images.append(img)

        if len(valid_images) < 2:
            raise ValueError("At least 2 valid calibration images required")

        # Estimate homographies
        homographies = []
        for points in image_points:
            H = self.estimate_homography(object_points[:, :2], points.reshape(-1, 2))
            homographies.append(H)

        # Initial parameter estimation
        self.K = self.extract_intrinsics(homographies)

        # Estimate extrinsics
        self.R = []
        self.t = []
        for H in homographies:
            R, t = self.estimate_extrinsics(H)
            self.R.append(R)
            self.t.append(t)

        # Optimize all parameters
        self.optimize_parameters(object_points, image_points)

        # Calculate reprojection errors
        total_error = 0
        per_view_errors = []

        for i, points_2d in enumerate(image_points):
            projected = self.project_points(object_points, self.R[i], self.t[i],
                                            self.K, self.k)
            error = np.sqrt(np.mean((projected - points_2d.reshape(-1, 2)) ** 2))
            per_view_errors.append(error)
            total_error += error

        total_error /= len(image_points)

        return CalibrationResult(
            K=self.K.copy(),
            k=self.k.copy(),
            R=[R.copy() for R in self.R],
            t=[t.copy() for t in self.t],
            reprojection_error=total_error,
            per_view_errors=per_view_errors
        )

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion from image"""
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.K, np.hstack((self.k, [0, 0, 0])), (w, h), 1, (w, h))

        dst = cv2.undistort(image, self.K, np.hstack((self.k, [0, 0, 0])),
                            None, newcameramtx)

        x, y, w, h = roi
        return dst[y:y + h, x:x + w]