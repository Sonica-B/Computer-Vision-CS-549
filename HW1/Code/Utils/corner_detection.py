import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy.optimize import least_squares
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Store calibration results."""
    K: np.ndarray  # Camera intrinsic matrix
    k: np.ndarray  # Distortion coefficients
    R: List[np.ndarray]  # Rotation matrices for each view
    t: List[np.ndarray]  # Translation vectors for each view
    reprojection_error: float  # Overall reprojection error
    per_view_errors: List[float]  # Reprojection error for each view


class CameraCalibration:
    def __init__(self, square_size: float = 21.5):
        """Initialize camera calibration with checkerboard square size."""
        self.square_size = square_size
        self.K = None  # Camera intrinsic matrix
        self.k = np.zeros((5, 1), dtype=np.float32)  # Updated to 5 coefficients
        self.R = []  # List of rotation matrices
        self.t = []  # List of translation vectors

    def find_corners(self, image: np.ndarray, pattern_size: Tuple[int, int] = (9, 6)) -> Tuple[
        Optional[np.ndarray], bool]:
        """
        Detect checkerboard corners in an image.

        Args:
            image: Input image
            pattern_size: Size of the checkerboard pattern (inner corners)

        Returns:
            corners: Detected corner coordinates
            found: Whether corners were successfully found
        """
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # Preprocess for better detection
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Find corners
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=(cv2.CALIB_CB_ADAPTIVE_THRESH +
                   cv2.CALIB_CB_NORMALIZE_IMAGE +
                   cv2.CALIB_CB_FILTER_QUADS)
        )

        if ret and corners is not None:
            # Refine corners to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners, True
        return None, False

    def create_object_points(self, pattern_size: Tuple[int, int]) -> np.ndarray:
        """
        Generate 3D object points for the checkerboard pattern.

        Args:
            pattern_size: Size of the checkerboard pattern (inner corners)

        Returns:
            3D object points array with shape (N,3) in float64 format
        """
        # Create the object points grid
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float64)

        # Fill in X,Y coordinates
        mgrid = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T
        objp[:, :2] = mgrid.reshape(-1, 2)

        # Scale by square size
        objp *= self.square_size

        # Ensure array is contiguous
        objp = np.ascontiguousarray(objp)

        return objp

    def estimate_homography(self, obj_points: np.ndarray, img_points: np.ndarray) -> np.ndarray:
        """
        Estimate homography matrix using Zhang's method.

        Args:
            obj_points: 3D object points
            img_points: 2D image points

        Returns:
            3x3 homography matrix
        """
        # Normalize points for better numerical stability
        obj_mean = np.mean(obj_points[:, :2], axis=0)
        img_mean = np.mean(img_points, axis=0)
        obj_std = np.std(obj_points[:, :2])
        img_std = np.std(img_points)

        norm_obj_points = (obj_points[:, :2] - obj_mean) / obj_std
        norm_img_points = (img_points - img_mean) / img_std

        # Build homogeneous equation system
        n_points = len(obj_points)
        A = np.zeros((2 * n_points, 9))

        for i in range(n_points):
            X, Y = norm_obj_points[i]
            u, v = norm_img_points[i]
            A[2 * i] = [-X, -Y, -1, 0, 0, 0, X * u, Y * u, u]
            A[2 * i + 1] = [0, 0, 0, -X, -Y, -1, X * v, Y * v, v]

        # Solve using SVD
        _, _, Vh = np.linalg.svd(A)
        H = Vh[-1].reshape(3, 3)

        # Denormalize
        T_obj = np.array([[1 / obj_std, 0, -obj_mean[0] / obj_std],
                          [0, 1 / obj_std, -obj_mean[1] / obj_std],
                          [0, 0, 1]])
        T_img = np.array([[1 / img_std, 0, -img_mean[0] / img_std],
                          [0, 1 / img_std, -img_mean[1] / img_std],
                          [0, 0, 1]])

        H = np.linalg.inv(T_img) @ H @ T_obj
        return H / H[2, 2]

    def calibrate(self, images: List[np.ndarray], pattern_size: Tuple[int, int] = (9, 6)) -> CalibrationResult:
        """
        Perform camera calibration using Zhang's method.

        Args:
            images: List of calibration images
            pattern_size: Size of the checkerboard pattern

        Returns:
            CalibrationResult object containing all calibration parameters
        """
        # Collect corner points and estimate homographies
        object_points = []
        image_points = []
        homographies = []
        objp = self.create_object_points(pattern_size)

        for img in images:
            corners, found = self.find_corners(img, pattern_size)
            if found:
                object_points.append(objp)
                image_points.append(corners)
                H = self.estimate_homography(objp, corners.reshape(-1, 2))
                homographies.append(H)

        if len(homographies) < 2:
            raise ValueError("At least 2 valid calibration images required")

        # Initial parameter estimation using Zhang's method
        V = np.zeros((2 * len(homographies), 6))
        for i, H in enumerate(homographies):
            h1, h2 = H[:, 0], H[:, 1]
            V[2 * i] = [h1[0] * h2[0], h1[0] * h2[1] + h1[1] * h2[0], h1[1] * h2[1],
                        h1[2] * h2[0] + h1[0] * h2[2], h1[2] * h2[1] + h1[1] * h2[2], h1[2] * h2[2]]
            V[2 * i + 1] = [h1[0] * h1[0] - h2[0] * h2[0],
                            2 * (h1[0] * h1[1] - h2[0] * h2[1]),
                            h1[1] * h1[1] - h2[1] * h2[1],
                            2 * (h1[0] * h1[2] - h2[0] * h2[2]),
                            2 * (h1[1] * h1[2] - h2[1] * h2[2]),
                            h1[2] * h1[2] - h2[2] * h2[2]]

        # Solve Vb = 0
        _, _, Vh = np.linalg.svd(V)
        b = Vh[-1]

        # Extract intrinsic parameters
        B11, B12, B22, B13, B23, B33 = b
        v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12)
        lambda_ = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11
        alpha = np.sqrt(lambda_ / B11)
        beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12 * B12))
        gamma = -B12 * alpha * alpha / lambda_
        u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda_

        # Initialize camera matrix
        self.K = np.array([[alpha, gamma, u0],
                           [0, beta, v0],
                           [0, 0, 1]], dtype=np.float64)

        # Calculate extrinsic parameters for each view
        self.R = []
        self.t = []
        for H in homographies:
            h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
            lambda_ = 1 / np.linalg.norm(np.linalg.inv(self.K) @ h1)

            r1 = lambda_ * np.linalg.inv(self.K) @ h1
            r2 = lambda_ * np.linalg.inv(self.K) @ h2
            r3 = np.cross(r1, r2)

            R = np.column_stack([r1, r2, r3])
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt

            t = lambda_ * np.linalg.inv(self.K) @ h3

            self.R.append(R)
            self.t.append(t.reshape(3, 1))

        # Refine all parameters through optimization
        return self.refine_parameters(object_points, image_points)

    def refine_parameters(self, object_points: List[np.ndarray], image_points: List[np.ndarray]) -> CalibrationResult:
        """
        Refine calibration parameters using maximum likelihood estimation.
        Added proper convergence criteria and bounds to prevent infinite loops.
        """

        def compute_residuals(params):
            """Compute reprojection error for optimization."""
            K_new = np.array([[params[0], params[1], params[2]],
                              [0, params[3], params[4]],
                              [0, 0, 1]], dtype=np.float64)

            # Use first two distortion coefficients only for stability
            k_new = np.array([[params[5], params[6], 0., 0., 0.]], dtype=np.float64)

            residuals = []
            param_idx = 7

            for i in range(len(object_points)):
                obj_pts = np.ascontiguousarray(object_points[i], dtype=np.float64)
                rvec = np.array(params[param_idx:param_idx + 3], dtype=np.float64).reshape(3, 1)
                tvec = np.array(params[param_idx + 3:param_idx + 6], dtype=np.float64).reshape(3, 1)
                param_idx += 6

                try:
                    proj_points, _ = cv2.projectPoints(
                        objectPoints=obj_pts,
                        rvec=rvec,
                        tvec=tvec,
                        cameraMatrix=K_new,
                        distCoeffs=k_new
                    )
                    img_pts = image_points[i].reshape(-1, 2).astype(np.float64)
                    current_residuals = (img_pts - proj_points.reshape(-1, 2)).ravel()
                    residuals.extend(current_residuals)
                except Exception as e:
                    print(f"Error in projectPoints: {str(e)}")
                    # Return large residuals on error to guide optimization away
                    residuals.extend([1000.0] * (object_points[i].shape[0] * 2))

            return np.array(residuals, dtype=np.float64)

        # Initialize parameters
        initial_params = [
            self.K[0, 0], self.K[0, 1], self.K[0, 2],  # K matrix elements
            self.K[1, 1], self.K[1, 2],  # K matrix elements
            0.0, 0.0  # Initial distortion coefficients (k1, k2)
        ]

        # Add extrinsic parameters
        for R, t in zip(self.R, self.t):
            rvec, _ = cv2.Rodrigues(R.astype(np.float64))
            initial_params.extend(rvec.ravel())
            initial_params.extend(t.ravel())

        initial_params = np.array(initial_params, dtype=np.float64)

        try:
            # Run optimization with proper parameters
            result = least_squares(
                compute_residuals,
                initial_params,
                method='lm',  # Levenberg-Marquardt algorithm
                ftol=1e-8,  # Function tolerance
                xtol=1e-8,  # Parameter tolerance
                gtol=1e-8,  # Gradient tolerance
                max_nfev=1000,  # Maximum number of function evaluations
                verbose=1  # Reduced verbosity
            )

            if not result.success:
                print(f"Warning: Optimization did not converge: {result.message}")

            optimized_params = result.x

            # Update parameters
            self.K = np.array([[optimized_params[0], optimized_params[1], optimized_params[2]],
                               [0, optimized_params[3], optimized_params[4]],
                               [0, 0, 1]], dtype=np.float64)

            # Update distortion coefficients (k1, k2 only)
            self.k = np.array([[optimized_params[5], optimized_params[6], 0., 0., 0.]], dtype=np.float64).T

            # Update extrinsic parameters
            param_idx = 7
            self.R = []
            self.t = []
            for _ in range(len(object_points)):
                rvec = optimized_params[param_idx:param_idx + 3].reshape(3, 1)
                tvec = optimized_params[param_idx + 3:param_idx + 6].reshape(3, 1)
                param_idx += 6

                R, _ = cv2.Rodrigues(rvec)
                self.R.append(R)
                self.t.append(tvec)

            # Calculate errors
            residuals = compute_residuals(optimized_params)
            total_error = np.sqrt(np.mean(residuals ** 2))
            points_per_view = len(object_points[0]) * 2
            per_view_errors = [
                np.sqrt(np.mean(residuals[i:i + points_per_view] ** 2))
                for i in range(0, len(residuals), points_per_view)
            ]

            return CalibrationResult(
                K=self.K.copy(),
                k=self.k.copy(),
                R=[R.copy() for R in self.R],
                t=[t.copy() for t in self.t],
                reprojection_error=total_error,
                per_view_errors=per_view_errors
            )

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            print(f"Initial params shape: {initial_params.shape}")
            print(f"Initial params dtype: {initial_params.dtype}")
            raise

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove lens distortion from an image.

        Args:
            image: Input distorted image

        Returns:
            Undistorted image
        """
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.k, (w, h), 1, (w, h))
        dst = cv2.undistort(image, self.K, self.k, None, newcameramtx)
        x, y, w, h = roi
        return dst[y:y + h, x:x + w]

    def save_rectified_image(self, image: np.ndarray, object_points: np.ndarray, corners: np.ndarray,
                             output_path: str = "rectified_reprojected.png") -> None:
        """
        Save rectified image with detected and reprojected corners visualization.

        Args:
            image: Input image
            object_points: 3D object points
            corners: Detected corner points
            output_path: Path to save the visualization
        """
        # First undistort the image
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.k, (w, h), 1, (w, h))
        undist = cv2.undistort(image, self.K, self.k, None, newcameramtx)

        # Make visualization image (copy of undistorted image)
        vis_img = undist.copy()

        # Project object points using current parameters
        projected_points, _ = cv2.projectPoints(object_points,
                                                cv2.Rodrigues(self.R[0])[0],
                                                self.t[0],
                                                self.K,
                                                self.k)

        # Draw detected corners in red
        for corner in corners:
            point = tuple(corner.ravel().astype(int))
            cv2.circle(vis_img, point, 5, (0, 0, 255), -1)  # Red circles

        # Draw reprojected points in green
        for point in projected_points:
            point = tuple(point.ravel().astype(int))
            cv2.circle(vis_img, point, 3, (0, 255, 0), -1)  # Green circles

        # Add legend
        legend_height = 60
        legend = np.full((legend_height, w, 3), 255, dtype=np.uint8)

        # Add legend text and samples
        cv2.circle(legend, (30, 20), 5, (0, 0, 255), -1)
        cv2.putText(legend, "Detected Corners", (50, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.circle(legend, (30, 45), 3, (0, 255, 0), -1)
        cv2.putText(legend, "Reprojected Points", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Combine visualization and legend
        final_img = np.vstack((vis_img, legend))

        # Add error information
        error = np.mean(np.linalg.norm(corners - projected_points, axis=2))
        cv2.putText(final_img, f"Reprojection Error: {error:.3f} pixels",
                    (w - 300, h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Save the final visualization
        cv2.imwrite(output_path, final_img)

        # Optional: Also save undistorted image without visualization
        undist_path = output_path.replace('.png', '_undistorted.png')
        cv2.imwrite(undist_path, undist)