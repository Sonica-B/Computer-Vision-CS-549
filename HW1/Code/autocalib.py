import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
from Utils.corner_detection import *


@dataclass
class CalibrationResult:
    """Stores calibration results"""
    K: np.ndarray  # Camera matrix
    k: np.ndarray  # Distortion coefficients [k1, k2]
    R: List[np.ndarray]  # Rotation matrices
    t: List[np.ndarray]  # Translation vectors
    reprojection_error: float
    per_view_errors: List[float]
    homographies: List[np.ndarray]


class CameraCalibration:
    def __init__(self, square_size: float = 21.5, pattern_size: Tuple[int, int] = (3, 6)):
        """
        Initialize camera calibration using Zhang's method
        Args:
            square_size: Size of checkerboard squares in mm
            pattern_size: Number of inner corners (rows, cols)
        """
        self.square_size = square_size
        self.pattern_size = pattern_size
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize camera parameters
        self.K = None  # Camera matrix
        self.k = np.zeros(2)  # Distortion coefficients [k1, k2]
        self.R = []  # Rotation matrices
        self.t = []  # Translation vectors
        self.homographies = []

    def create_object_points(self) -> np.ndarray:
        """Create 3D points of checkerboard corners in world coordinate system"""
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0],
                      0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    def find_corners(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """
        Fast and robust corner detection optimized for Google Pixel XL images
        Args:
            image: Input image
        Returns:
            corners: Corner coordinates if found
            success: Whether corners were found
        """
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # Resize if image is too large (speeds up detection)
        height, width = gray.shape[:2]
        max_dimension = 1000
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        # Basic preprocessing
        # Normalize image to reduce lighting variations
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)

        # Find corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                  cv2.CALIB_CB_FAST_CHECK
        )

        if ret:
            # If image was resized, scale corners back
            if max(height, width) > max_dimension:
                corners *= 1.0 / scale

            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )

            return corners, True

        return None, False

    def draw_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Draw detected corners on image
        Args:
            image: Input image
            corners: Detected corner coordinates
        Returns:
            Image with drawn corners
        """
        vis = image.copy()
        cv2.drawChessboardCorners(vis, self.pattern_size, corners, True)

        # Draw corner numbers for debugging
        for i, corner in enumerate(corners):
            x, y = corner.ravel()
            cv2.putText(vis, str(i), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return vis

    def estimate_homography(self, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """
        Estimate homography matrix using normalized DLT
        Args:
            src_points: Source points (nx2)
            dst_points: Destination points (nx2)
        Returns:
            3x3 homography matrix
        """
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

        # Normalize coordinates
        src_normalized = cv2.perspectiveTransform(
            src_points.reshape(-1, 1, 2), T_src).reshape(-1, 2)
        dst_normalized = cv2.perspectiveTransform(
            dst_points.reshape(-1, 1, 2), T_dst).reshape(-1, 2)

        # Construct equation matrix
        A = []
        for (x, y), (u, v) in zip(src_normalized, dst_normalized):
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        A = np.array(A)

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)

        # Denormalize
        H = np.linalg.inv(T_dst) @ H @ T_src
        return H / H[2, 2]

    def extract_camera_matrix(self) -> np.ndarray:
        """
        Extract camera matrix K from homographies using Zhang's method
        Returns:
            3x3 camera matrix
        """
        # Construct V matrix based on Zhang's paper
        V = []
        for H in self.homographies:
            h1, h2 = H[:, 0], H[:, 1]
            V.append(self._create_v_matrix(h1, h2))
            V.append(self._create_v_matrix(h1, h1) - self._create_v_matrix(h2, h2))
        V = np.array(V)

        # Solve Vb = 0 using SVD
        _, _, Vt = np.linalg.svd(V)
        b = Vt[-1]

        # Extract intrinsic parameters
        B11, B12, B22, B13, B23, B33 = b

        # Calculate camera matrix elements
        v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 ** 2)
        lambda_ = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
        alpha = np.sqrt(lambda_ / B11)
        beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12 ** 2))
        gamma = -B12 * alpha ** 2 * beta / lambda_
        u0 = gamma * v0 / beta - B13 * alpha ** 2 / lambda_

        return np.array([[alpha, gamma, u0],
                         [0, beta, v0],
                         [0, 0, 1]])

    def _create_v_matrix(self, h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
        """Helper function to create v matrix for Zhang's method"""
        return np.array([
            h1[0] * h2[0], h1[0] * h2[1] + h1[1] * h2[0], h1[1] * h2[1],
            h1[2] * h2[0] + h1[0] * h2[2], h1[2] * h2[1] + h1[1] * h2[2], h1[2] * h2[2]
        ])

    def estimate_extrinsics(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate extrinsic parameters from homography
        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        """
        K_inv = np.linalg.inv(self.K)
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

        # Calculate rotation matrix columns
        lambda_ = 1 / np.linalg.norm(K_inv @ h1)
        r1 = lambda_ * K_inv @ h1
        r2 = lambda_ * K_inv @ h2
        r3 = np.cross(r1, r2)

        R = np.column_stack([r1, r2, r3])

        # Ensure proper rotation matrix
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        # Calculate translation vector
        t = lambda_ * K_inv @ h3

        return R, t.reshape(3, 1)

    def project_points(self, object_points: np.ndarray, R: np.ndarray,
                       t: np.ndarray, K: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Project 3D points to image plane considering distortion"""
        # Transform points to camera coordinates
        points_camera = (R @ object_points.T + t).T

        # Project to normalized image coordinates
        points_normalized = points_camera[:, :2] / points_camera[:, 2:]

        # Apply distortion
        r2 = np.sum(points_normalized ** 2, axis=1)
        radial_distortion = 1 + k[0] * r2 + k[1] * r2 ** 2
        points_distorted = points_normalized * radial_distortion.reshape(-1, 1)

        # Apply camera matrix
        points_image = (K[:2, :2] @ points_distorted.T + K[:2, 2:]).T

        return points_image

    def optimize_parameters(self, object_points: np.ndarray,
                            image_points: List[np.ndarray]) -> None:
        """Optimize all parameters using Levenberg-Marquardt"""

        def objective(params):
            # Extract parameters
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

        # Create initial parameter vector
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
        """
        Perform complete camera calibration
        Args:
            images: List of input images
        Returns:
            CalibrationResult object
        """
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
        self.homographies = []
        for points in image_points:
            H = self.estimate_homography(object_points[:, :2], points.reshape(-1, 2))
            self.homographies.append(H)

        # Initial parameter estimation
        self.K = self.extract_camera_matrix()

        # Estimate extrinsics
        self.R = []
        self.t = []
        for H in self.homographies:
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
            per_view_errors=per_view_errors,
            homographies=[H.copy() for H in self.homographies]
        )

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove lens distortion from image
        Args:
            image: Input distorted image
        Returns:
            Undistorted image
        """
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.K,
            np.hstack((self.k, [0, 0, 0])),  # OpenCV expects 5 distortion coefficients
            (w, h),
            1,
            (w, h)
        )

        # Undistort
        dst = cv2.undistort(
            image,
            self.K,
            np.hstack((self.k, [0, 0, 0])),
            None,
            newcameramtx
        )

        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        return dst

    def check_calibration_quality(self, image_points: List[np.ndarray],
                                  object_points: np.ndarray) -> Dict:
        """
        Compute various calibration quality metrics
        Args:
            image_points: List of detected corner coordinates
            object_points: 3D object points
        Returns:
            Dictionary containing quality metrics
        """
        metrics = {}

        # Compute reprojection error for each view
        errors = []
        for i, points_2d in enumerate(image_points):
            projected = self.project_points(object_points, self.R[i],
                                            self.t[i], self.K, self.k)
            error = np.sqrt(np.mean((projected - points_2d.reshape(-1, 2)) ** 2))
            errors.append(error)

        metrics['per_view_errors'] = errors
        metrics['mean_error'] = np.mean(errors)
        metrics['max_error'] = np.max(errors)
        metrics['std_error'] = np.std(errors)

        # Check camera matrix properties
        metrics['focal_length_ratio'] = self.K[0, 0] / self.K[1, 1]  # Should be close to 1
        metrics['skew'] = self.K[0, 1]  # Should be close to 0

        # Check distortion coefficients
        metrics['k1'] = self.k[0]
        metrics['k2'] = self.k[1]

        return metrics

    def check_degeneracy(self, R1: np.ndarray, R2: np.ndarray) -> bool:
        """
        Check if two views form a degenerate configuration
        Args:
            R1: First rotation matrix
            R2: Second rotation matrix
        Returns:
            True if configuration is degenerate
        """
        # Get plane normals
        n1 = R1[:, 2]
        n2 = R2[:, 2]

        # Compute angle between normals
        cos_angle = np.dot(n1, n2)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle)

        # Check for near-parallel planes (within 5 degrees)
        if abs(angle_deg) < 5 or abs(angle_deg - 180) < 5:
            return True

        return False

    def save_debug_images(self, index: int, original: np.ndarray,
                          processed: np.ndarray, corners: Optional[np.ndarray]):
        """Save debug images to understand corner detection failures"""
        debug_dir = self.output_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)

        # Save original
        cv2.imwrite(str(debug_dir / f'original_{index}.jpg'), original)

        # Save processed
        cv2.imwrite(str(debug_dir / f'processed_{index}.jpg'), processed)

        if corners is not None:
            # Draw corners on copy
            vis = original.copy()
            cv2.drawChessboardCorners(vis, self.pattern_size, corners, True)
            cv2.imwrite(str(debug_dir / f'corners_{index}.jpg'), vis)

    def save_calibration(self, filename: str) -> None:
        """
        Save calibration parameters to file
        Args:
            filename: Output filename (.npz)
        """
        np.savez(filename,
                 K=self.K,
                 k=self.k,
                 R=self.R,
                 t=self.t,
                 homographies=self.homographies)

    def load_calibration(self, filename: str) -> None:
        """
        Load calibration parameters from file
        Args:
            filename: Input filename (.npz)
        """
        data = np.load(filename)
        self.K = data['K']
        self.k = data['k']
        self.R = data['R']
        self.t = data['t']
        self.homographies = data['homographies']